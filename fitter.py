__author__ = 'gjones'
import emcee
from corner import corner
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import models

reload(models)


def mc_red_white_rolloff(fr, pxx, dof, red_exponent_limits=(.1, 3), burnin=100, samples=500, nwalkers=32, threads=1,
                         initial=None):
    logprior, limits = models.make_logprior_red_white_rolloff(fr, pxx, red_exponent_limits=red_exponent_limits)
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=5, lnpostfn=models.logprob_red_white_rolloff,
                                    args=(fr, pxx, dof, logprior), threads=threads)
    if initial is None:
        initials = []
        for k, (low, high) in enumerate(limits):
            initials.append(np.random.uniform(low=low, high=high, size=nwalkers))
        initials = np.array(initials).T
    else:
        initials = np.random.randn(nwalkers, len(limits)) * initial / 100 + initial
    # print "burning in"
    #pos,_,_ = sampler.run_mcmc(initials,burnin)
    print "sampling"
    sampler.reset()
    sampler.run_mcmc(initials, samples)
    return sampler


class MCPowerSpectrumFitter(object):
    def _setup_sampler(self):
        self.sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers, dim=self.dim, lnpostfn=self.logprob)

    def _uniform_logprior(self, params):
        for value, (min, max) in zip(params, self.limits):
            if value < min or value > max:
                return -np.inf
        return 0

    def logprob(self, params):
        prior = self.logprior(params)
        if not np.isfinite(prior):
            return -np.inf
        ll, model_psd = self.loglikelihood(params)
        return ll.sum()

    def loglikelihood(self, params):
        model_psd = self.model.psd(self.frequency_data, *params)
        dof = self.dof
        if self.debias:
            scale = (dof / 2.) * (dof - 2.) / (dof)
        else:
            scale = dof / 2.
        return scipy.stats.gamma.logpdf(scale * self.psd_data / model_psd, dof / 2.), model_psd

    def run_mcmc(self, samples=500, burnin=200):
        self.sampler.run_mcmc(self.initials, samples)
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, self.dim))
        self.burnin = burnin
        self.get_map_params()

    def get_map_params(self):
        mapx = self.sampler.lnprobability.max(0)
        mapy = mapx.argmax()
        mapx = self.sampler.lnprobability[:, mapy].argmax()
        print "MAP:", self.sampler.lnprobability[mapx, mapy]
        self.map_params = self.sampler.chain[mapx, mapy, :]
        for k in range(self.dim):
            print "%s : %f" % (self.model.labels[k], self.map_params[k])

    def update_initials_from_posterior(self):
        spread = self.sampler.lnprobability.ptp(axis=0)
        usable_sample_mask = spread / spread[-1] < 4
        usable_samples = self.sampler.chain[:, usable_sample_mask, :].reshape((-1, self.dim))
        sample_indexes = np.random.random_integers(0, usable_samples.shape[0] - 1, size=(self.nwalkers,))
        self.initials = usable_samples[sample_indexes, :]

    def update_initials_from_map(self, scale=10.):
        self.set_initial_ball(self.map_params, scale=scale)

    def set_initial_ball(self, values, scale=10.):
        self.initials = values[None, :] * (1 + np.random.randn(self.nwalkers, self.dim) / scale)
        for k in range(self.dim):
            self.initials[:, k] = self.initials[:, k].clip(*self.limits[k])

    def auto_run_mcmc(self, update_from_map=True):
        self.sampler.reset()
        self.run_mcmc()
        if update_from_map:
            self.update_initials_from_map()
        else:
            self.update_initials_from_posterior()
        self.sampler.reset()
        self.run_mcmc()

    def corner(self, **kwargs):
        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [np.exp(-3), np.exp(-1), 0.5, 1 - np.exp(-1), 1 - np.exp(-3)]
        try:
            plot_map_params = kwargs.pop('plot_map_params')
        except KeyError:
            plot_map_params = False
        fig = corner(self.samples.reshape((-1, self.dim)), labels=self.model.labels, **kwargs)
        if plot_map_params:
            if self.dim == 1:
                ax = fig.axes[0]
                ax.axvline(self.map_params[0], color='r', linestyle='--')
            else:
                for k in range(self.dim):
                    for m in range(k + 1):
                        ax = fig.axes[k * self.dim + m]
                        # ax.text(0.5,0.5,('%d,%d' % (k,m)),transform=ax.transAxes)
                        ax.axvline(self.map_params[m], color='r', linestyle='--')
                        if k != m:
                            ax.axhline(self.map_params[k], color='r', linestyle='--')
                            ax.plot(self.map_params[m], self.map_params[k], 'o', color='r', mew=0, markersize=2)
        return fig

    def plot_lnprob(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.sampler.lnprobability.T)

    def plot_results(self, plot_map_result=True, num_samples=50, alpha=0.02, plot_limits=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.loglog(self.frequency_data, self.psd_data, lw=2)
        for k in np.random.random_integers(0, self.samples.shape[0], size=num_samples):
            psd_sample = self.model.psd(self.frequency_data, *self.samples[k, :])
            ax.loglog(self.frequency_data, psd_sample, 'k', alpha=alpha)
            if plot_limits:
                ax.loglog(self.frequency_data, psd_sample * scipy.stats.gamma.ppf(0.05, self.dof) / (self.dof), 'r',
                          alpha=alpha)
                ax.loglog(self.frequency_data, psd_sample * scipy.stats.gamma.ppf(0.95, self.dof) / (self.dof), 'r',
                          alpha=alpha)

        if plot_map_result:
            psd_map = self.model.psd(self.frequency_data, *self.map_params)
            ax.loglog(self.frequency_data, psd_map, '--', lw=2, color='orange')
            if plot_limits:
                ax.loglog(self.frequency_data, psd_map * scipy.stats.gamma.ppf(0.05, self.dof) / (self.dof), '--', lw=2,
                          color='orange')
                ax.loglog(self.frequency_data, psd_map * scipy.stats.gamma.ppf(0.95, self.dof) / (self.dof), '--', lw=2,
                          color='orange')

    def data_likelihood_in_sigma(self, params=None):
        if params is None:
            params = self.map_params
        model_psd = self.model.psd(self.frequency_data, *params)
        sigmas = -np.log(.5 - np.abs(scipy.stats.gamma.cdf(self.psd_data * self.dof / model_psd, self.dof) - .5)) / 2
        return sigmas

    def suggest_outliers_to_prune(self, max_sigma=5, params=None, verbose=False):
        sigmas = self.data_likelihood_in_sigma(params)
        mask = sigmas < max_sigma
        if verbose:
            sorter = sigmas.argsort()
            freq = self.frequency_data[sorter]
            sigmas = sigmas[sorter]
            for index in range(freq.shape[0])[::-1]:
                sigma = sigmas[index]
                if sigma < max_sigma:
                    break
                print "%8.3f Hz : %4.1f sigma" % (freq[index], sigma)
        return mask


    def plot_data_likelihood(self, params=None, outlier_threshold_in_sigma=5):
        if params is None:
            params = self.map_params
        loglikelihood, model_psd = self.loglikelihood(params)
        sigmas = self.data_likelihood_in_sigma(params)
        mask = sigmas > outlier_threshold_in_sigma
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        ax1 = axs[0]
        ax1.loglog(self.frequency_data, self.psd_data)
        ax1.loglog(self.frequency_data, model_psd, '--', color='orange')
        ax1.loglog(self.frequency_data, model_psd * scipy.stats.gamma.ppf(0.05, self.dof) / (self.dof), '--', lw=2,
                   color='orange')
        ax1.loglog(self.frequency_data, model_psd * scipy.stats.gamma.ppf(0.95, self.dof) / (self.dof), '--', lw=2,
                   color='orange')
        ax1.loglog(self.frequency_data[mask], self.psd_data[mask], 'ro')
        # ax1.loglog(self.frequency_data[mask],self.psd_data[mask],'ro')
        ax2 = axs[1]
        cdf = scipy.stats.gamma.cdf(self.psd_data / (model_psd * self.dof), self.dof)
        ax2.semilogx(self.frequency_data, sigmas)
        ax2.semilogx(self.frequency_data[mask], sigmas[mask], 'ro')

        #for fr in self.frequency_data[mask]:
        #    ax2.axvline(fr,color='r',alpha=0.5,linestyle='--')


# def __init__(self,frequency_data,psd_data,dof,psd_model):

class MCWhiteNoisePowerSpectrumFitter(MCPowerSpectrumFitter):
    def __init__(self, frequency_data, psd_data, dof, nwalkers=32, debias=True):
        self.frequency_data = frequency_data
        self.psd_data = psd_data
        self.dof = dof
        self.nwalkers = nwalkers
        self.dim = 1
        self._setup_sampler()
        self.estimate_parameter_limits()
        self.logprior = self._uniform_logprior
        self.model = models.WhiteNoisePowerSpectrumModel(1.0)
        self.debias = debias

    def estimate_parameter_limits(self):
        min_psd = self.psd_data.min()
        max_psd = self.psd_data.max()
        self.limits = [(min_psd, max_psd)]  # limit on white noise density

    def make_initial_guesses(self):
        initials = []
        for k, (low, high) in enumerate(self.limits):
            initials.append(np.random.uniform(low=low, high=high, size=self.nwalkers))
        self.initials = np.array(initials).T


class MCRedWhiteNoisePowerSpectrumFitter(MCPowerSpectrumFitter):
    def __init__(self, frequency_data, psd_data, dof, nwalkers=32, debias=True):
        mask = frequency_data > 0
        self.frequency_data = frequency_data[mask]
        self.psd_data = psd_data[mask]
        if np.isscalar(dof):
            self.dof = dof
        else:
            self.dof = dof[mask]
        self.nwalkers = nwalkers
        self.dim = 3
        self._setup_sampler()
        self.estimate_parameter_limits()
        self.logprior = self._uniform_logprior
        self.model = models.RedWhiteNoisePowerSpectrumModel(1.0, 100, 1)
        self.debias = debias

    def estimate_parameter_limits(self):
        min_psd = self.psd_data.min()
        max_psd = self.psd_data.max()
        self.limits = [(min_psd, max_psd),  # limit on white noise density
                       (self.frequency_data.min(), self.frequency_data.max()),
                       (.1, 3)]

    def make_initial_guesses(self):
        initials = []
        for k, (low, high) in enumerate(self.limits):
            initials.append(np.random.uniform(low=low, high=high, size=self.nwalkers))
        self.initials = np.array(initials).T


class MCRedWhiteNoiseRolloffPowerSpectrumFitter(MCPowerSpectrumFitter):
    def __init__(self, frequency_data, psd_data, dof, nwalkers=32, debias=True):
        mask = frequency_data > 0
        self.frequency_data = frequency_data[mask]
        self.psd_data = psd_data[mask]
        if np.isscalar(dof):
            self.dof = dof
        else:
            self.dof = dof[mask]
        self.nwalkers = nwalkers
        self.dim = 5
        self._setup_sampler()
        self.estimate_parameter_limits()
        self.logprior = self._uniform_logprior
        self.model = models.RedWhiteRolloffPowerSpectrumModel(2.0, 1., 10, 1., 2000)
        self.debias = debias

    def estimate_parameter_limits(self):
        min_psd = self.psd_data.min()
        max_psd = self.psd_data.max()
        self.limits = [(min_psd / 10., max_psd),  # limit on in band noise density
                       (0, max_psd),  # limit on out of band noise density
                       (self.frequency_data.min(), self.frequency_data.max()),  # fknee
                       (.1, 3),  # alpha
                       (self.frequency_data.min(), self.frequency_data.max()), ]  # f3db

    def make_initial_guesses(self):
        initials = []
        for k, (low, high) in enumerate(self.limits):
            initials.append(np.random.uniform(low=low, high=high, size=self.nwalkers))
        self.initials = np.array(initials).T


def logbin(fr, pxx, scale=128):
    mask = fr > 0
    fr = fr[mask]
    pxx = pxx[mask]
    edges = np.logspace(np.log10(fr[0]), np.log10(fr[-1]), num=scale)
    indexes = np.digitize(fr, edges)
    pxx_binned = np.bincount(indexes, weights=pxx)
    fr_binned = np.bincount(indexes, weights=fr)
    counts = np.bincount(indexes)
    pxx_binned = pxx_binned[counts > 0]
    fr_binned = fr_binned[counts > 0]
    counts = counts[counts > 0]
    return fr_binned / counts, pxx_binned / counts, counts
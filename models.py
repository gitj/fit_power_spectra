__author__ = 'gjones'
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

def red_white_model(f,fknee,noise,alpha):
    return noise*(1+(fknee/f)**alpha)

def red_white_rolloff_model(f,fknee,in_band_noise,out_band_noise,fcutoff,red_exponent):
    in_band = red_white_model(f,fknee,noise=in_band_noise,alpha=red_exponent)
    return in_band/(1+(f/fcutoff)**2) + out_band_noise

def make_logprior_red_white_rolloff(fr,pxx,red_exponent_limits=(0.1,3)):
    mask = fr > 0
    fr = fr[mask]
    pxx = pxx[mask]
    fknee_min = fcutoff_min = fr.min()
    fknee_max = fcutoff_max = fr.max()*2
    in_band_noise_min = 0
    in_band_noise_max = out_band_noise_max = pxx.max()*2
    out_band_noise_min = pxx.min()/2.
    print "%.3f < fknee < %.3f" % (fknee_min,fknee_max)
    print "%.3f < fcutoff < %.3f" % (fcutoff_min,fcutoff_max)
    print "%.3f < in_band_noise < %.3f" % (in_band_noise_min,in_band_noise_max)
    print "%.3f < out_band_noise < %.3f" % (out_band_noise_min,out_band_noise_max)
    print "%.3f < red_exponent < %.3f" % red_exponent_limits
    def logprior(params):
        fknee,in_band_noise,out_band_noise,fcutoff,red_exponent = params
        #print fknee_min,fknee, (fknee_min>fknee)
        if fknee_min > fknee or fknee > fknee_max:
            return -np.inf
        if fcutoff_min > fcutoff or fcutoff > fcutoff_max:
            return -np.inf
        if in_band_noise_min > in_band_noise or in_band_noise > in_band_noise_max:
            return -np.inf
        if out_band_noise_min > out_band_noise or out_band_noise > out_band_noise_max:
            return -np.inf
        if red_exponent_limits[0] > red_exponent or red_exponent > red_exponent_limits[1]:
            return -np.inf
        return 0
    limits = [(fknee_min,fknee_max),(in_band_noise_min,in_band_noise_max),
        (out_band_noise_min,out_band_noise_max),(fcutoff_min,fcutoff_max),red_exponent_limits]
    return logprior,limits

def logprob_red_white_rolloff(params,fr,pxx,dof,logprior,debias=True):
    prior = logprior(params)
    #print prior,params
    if not np.isfinite(prior):
        return -np.inf
    model = red_white_rolloff_model(fr,*params)
    if debias:
        scale = (dof/2.)*(dof-2.)/(dof)
    else:
        scale = dof/2.
    return scipy.stats.gamma.logpdf(scale*pxx/model,dof/2.).sum()

def model_timeseries(nsamp,fs,power_spectrum):
    nfreq = nsamp//2+1
    freq = np.linspace(0,fs/2.,nfreq,endpoint=True)
    amps = np.exp(-2j*np.pi*np.random.rand(nfreq)) * np.sqrt(nsamp)
    amps *= np.sqrt(power_spectrum)
    amps[-1] = np.abs(amps[-1])*2
    amps[0] = 0
    ts = np.fft.irfft(amps)*np.sqrt(fs/2.)
    return ts

def model_psd(power_spectrum,dof):
    if np.isscalar(dof):
        size = None
    else:
        size = power_spectrum.shape
    return power_spectrum*scipy.stats.gamma.rvs(dof/2.,size=size)/(dof/2.)

def red_white_rolloff_timeseries(nsamp,fknee,in_band_noise,out_band_noise,fcutoff,red_exponent,fs):
    power_spectrum = red_white_rolloff_model(freq,fknee,in_band_noise,out_band_noise,fcutoff,red_exponent)
    return model_timeseries(nsamp,fs=fs,power_spectrum=power_spectrum)

def red_white_rolloff_sim_psd(freq,fknee,in_band_noise,out_band_noise,fcutoff,red_exponent,dof):
    power = red_white_rolloff_model(freq,fknee,in_band_noise,out_band_noise,fcutoff,red_exponent)
    return model_psd(power,dof)


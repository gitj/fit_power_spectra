__author__ = 'gjones'
import emcee
import numpy as np
from matplotlib import pyplot as plt
import models
reload(models)

def mc_red_white_rolloff(fr,pxx,dof,red_exponent_limits=(.1,3),burnin=100,samples=500,nwalkers=32,threads=1,
                         initial = None):
    logprior,limits = models.make_logprior_red_white_rolloff(fr,pxx,red_exponent_limits=red_exponent_limits)
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers,dim=5,lnpostfn=models.logprob_red_white_rolloff,
                                    args=(fr,pxx,dof,logprior),threads=threads)
    if initial is None:
        initials = []
        for k,(low,high) in enumerate(limits):
            initials.append(np.random.uniform(low=low,high=high,size=nwalkers))
        initials = np.array(initials).T
    else:
        initials = np.random.randn(nwalkers,len(limits))*initial/100 + initial
    #print "burning in"
    #pos,_,_ = sampler.run_mcmc(initials,burnin)
    print "sampling"
    sampler.reset()
    sampler.run_mcmc(initials,samples)
    return sampler
"""Exact Frobenius moments and a fixed-coefficient operator-norm control variate.

Moment coordinates are (G0, G1, svec(G2)/sqrt(2)). All covariances include
the numerical jitter used by the actual auxiliary sampler. The raw corrected
mean is unbiased conditional on the released moments; the square root and
the explicitly recorded nonpositive safeguard are not unbiased estimators.
"""
import numpy as np
from modules.md.moments import unpack
from modules.md.structured_shared_field import JetMarginal


class FrobeniusMoment:
    """Cache public covariance contractions for E||H||_F^2."""

    def __init__(self, marginal, D):
        self.D=int(D)
        p=(D+1)*(D+2)//2
        if isinstance(marginal,JetMarginal):
            if marginal.D!=D:
                raise ValueError('marginal dimension differs from decoder')
            self.isotropic=(float(marginal.scalar[0,0]),float(marginal.vector_variance),
                float(marginal.scalar[1,1]),float(marginal.traceless_variance),
                float(marginal.scalar[0,1]))
        elif np.ndim(marginal)==0:
            var=float(marginal)
            if not np.isfinite(var) or var<=0:
                raise ValueError('positive finite marginal variance required')
            self.isotropic=(var,var,var,var,0.)
        else:
            cov=np.asarray(marginal,dtype=float)
            if cov.shape!=(p,p) or not np.isfinite(cov).all():
                raise ValueError('finite normalized moment covariance required')
            self.isotropic=None
            self.a=float(cov[0,0])
            self.first_cov=cov[1:1+D,1:1+D].copy()
            self.second_energy=2*float(np.trace(cov[1+D:,1+D:]))
            self.mass_second=unpack(cov[0],D)[2]
            self.mass_first=cov[0,1:1+D].copy()
            basis=unpack(np.eye(p)[1+D:],D)[2]
            self.second_first=np.einsum('kij,ik->j',basis,cov[1:1+D,1+D:])

    def expectation(self,mean,covariance,bandwidth):
        m=np.asarray(mean,dtype=float);C=np.asarray(covariance,dtype=float)
        D=self.D;h=float(bandwidth)
        if m.shape[-1]!=D or C.shape!=m.shape+(D,):
            raise ValueError('mean and covariance shapes must match')
        if h<=0 or not np.isfinite(h) or not np.isfinite(m).all() or not np.isfinite(C).all():
            raise ValueError('finite means/covariances and positive bandwidth required')
        P=m[..., :,None]*m[...,None,:]-C
        norm_m=np.sum(m*m,axis=-1)
        norm_P=np.sum(P*P,axis=(-2,-1))
        if self.isotropic is not None:
            a,b,t,s,c=self.isotropic
            M=D*(D+1)//2
            terms=[2*h**4*(t+(M-1)*s),2*h*h*b*(D+1)*norm_m,
                   a*norm_P,2*h*h*np.sqrt(2/D)*c*np.trace(P,axis1=-2,axis2=-1)]
        else:
            terms=[h**4*self.second_energy,
                2*h*h*(np.trace(self.first_cov)*norm_m+np.einsum('...i,ij,...j->...',m,self.first_cov,m)),
                self.a*norm_P,
                2*h*h*np.einsum('ij,...ij->...',self.mass_second,P),
                -4*h**3*np.einsum('j,...j->...',self.second_first,m),
                -4*h*np.einsum('i,...ij,...j->...',self.mass_first,P,m)]
        answer=sum(terms)
        scale=sum(np.abs(term) for term in terms)
        if np.any(answer < -1e-12*np.maximum(scale,np.finfo(float).tiny)):
            raise ArithmeticError('negative analytic Frobenius second moment')
        return np.maximum(answer,0.)


def operator_second_moment_cv(operator_squared,frobenius_squared,exact_frobenius,*,D):
    """Fixed beta=4/D; safeguard nonpositive estimates with the original MC mean.

    The raw correction retains the original expectation target. Returning the
    raw value and safeguard flag prevents an unqualified unbiasedness claim.
    """
    Y=np.asarray(operator_squared,float);Z=np.asarray(frobenius_squared,float)
    mu=np.asarray(exact_frobenius,float)
    if Y.shape!=Z.shape or Y.ndim<1 or Y.shape[-1]<1 or mu.shape!=Y.shape[:-1]:
        raise ValueError('matching sample arrays and exact expectation required')
    if D<2 or not all(np.isfinite(v).all() for v in (Y,Z,mu)) or any(np.any(v<0) for v in (Y,Z,mu)):
        raise ValueError('nonnegative finite second moments required')
    plain=Y.mean(axis=-1)
    raw=plain-(4./D)*(Z.mean(axis=-1)-mu)
    guarded=raw<=0
    return dict(estimate=np.where(guarded,plain,raw),raw_corrected=raw,
                safeguarded=guarded,beta=4./D,plain_mean=plain)

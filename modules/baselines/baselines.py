"""Central-DP and nonprivate PCA comparators, sharing input and evaluation protocol.

Each mechanism is calibrated on the complete released vector under replace-one
adjacency, not on individual coordinates. All queries are public.
"""
import numpy as np
from modules.md.moments import empirical, unpack
from modules.privacy import analytic_gaussian_sigma

def _clip(y,radius):
    y=np.asarray(y,float)
    if y.ndim!=2 or len(y)<2 or not np.isfinite(y).all() or radius<=0:
        raise ValueError("finite sample matrix and positive clipping radius required")
    return y*np.minimum(1.,radius/np.maximum(np.linalg.norm(y,axis=-1),1e-300))[:,None]

def pca(reference,queries,*,d,epsilon=None,delta=None,rng=None,clip_radius=1.3):
    """Global affine PCA. DP joint release of (mean, svec(second)/sqrt(2)).

    Conservative replace-one L2 sensitivity sqrt(4*c^2+c^4)/n.
    This bound includes the second moment; it is NOT the uncorrected 2/n claim.
    """
    y=_clip(reference,clip_radius);x=np.asarray(queries,float)
    n,D=y.shape
    if x.ndim!=2 or x.shape[1]!=D or not 0<d<D:raise ValueError("invalid query/rank")
    mean=y.mean(axis=0); second=y.T@y/n
    if epsilon is not None:
        if rng is None or delta is None:raise ValueError("DP PCA needs RNG and delta")
        sd=float(analytic_gaussian_sigma(epsilon,delta,1.)[0]) * np.sqrt(4*clip_radius**2+clip_radius**4)/n
        mean=mean+sd*rng.standard_normal(D)
        upper=np.triu_indices(D)
        triangular=(second[upper]/np.where(upper[0]==upper[1],np.sqrt(2),1.))
        triangular+=sd*rng.standard_normal(len(triangular))
        rec=np.zeros((D,D))
        rec[upper]=triangular*np.where(upper[0]==upper[1],np.sqrt(2),1.)
        second=rec+np.triu(rec,1).T
    C=(second-np.outer(mean,mean))
    C=(C+C.T)*.5
    _,U=np.linalg.eigh(C);T=U[:,-d:]
    return mean+(x-mean)@T@T.T

def local_pca(reference,queries,*,d,bandwidth=0.1,epsilon=None,delta=None,rng=None):
    """Public-query local affine PCA. Fixed h; private mode adds IID joint noise.

    Gaussian moment feature F=(m0,m1/h,svec(M2/h^2)/sqrt2) has norm <=1
    for every record, because exp(-r^2)(1+r^2+r^4/2)<=1.
    Releasing all Q independent moment blocks has replace-one sensitivity
    bounded by 2*sqrt(Q)/n. This is conservative and contains no query privacy.
    """
    x=np.asarray(queries,float);ref=np.asarray(reference,float)
    if x.ndim!=2 or ref.ndim!=2 or x.shape[1]!=ref.shape[1] or not 0<d<x.shape[1]:
        raise ValueError("invalid local PCA inputs")
    n,D=ref.shape;Q=len(x)
    t=empirical(ref,x,float(bandwidth))
    if epsilon is not None:
        if rng is None or delta is None:raise ValueError("private local PCA needs RNG/delta")
        sd=float(analytic_gaussian_sigma(epsilon,delta,1.)[0])*2*np.sqrt(Q)/n
        t=t+sd*rng.standard_normal(t.shape)
    q,v,A=unpack(t,D)
    den=np.maximum(q,1/n)
    m=bandwidth*v/den[:,None]
    C=bandwidth**2*A/den[:,None,None]-m[:,:,None]*m[:,None,:]
    C=(C+C.swapaxes(-1,-2))*.5
    _,U=np.linalg.eigh(C)
    normal=U[:,:,:D-d]
    return x+np.einsum("qij,qkj,qk->qi",normal,normal,m)

def evaluate_baselines(reference,queries,*,d,epsilon,delta,rng):
    """Return comparable predictions. Each DP arm receives independent noise."""
    return {
        "NP-PCA":pca(reference,queries,d=d),
        "DP-PCA":pca(reference,queries,d=d,epsilon=epsilon,delta=delta,rng=rng),
        "NP-LocalPCA":local_pca(reference,queries,d=d),
        "DP-LocalPCA":local_pca(reference,queries,d=d,epsilon=epsilon,delta=delta,rng=rng),
    }

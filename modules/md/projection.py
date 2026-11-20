"""Exact weighted moment PSD projection (Appendix A.2), opt-in backend.

For X0=[[a,b.T],[b,C]], minimize .5*||X-X0||_F^2+.5*(X00-a)^2
subject to X>=0 and X00<=1. Its unique solution is (X0-t*e0*e0.T)_+
where min(a+t,1)=[(X0-t*e0*e0.T)_+]00, t>=0. This is NOT
ordinary eigenvalue clipping of X0, nor clipping the bottom block alone.
Only scalar root termination and eigensolver roundoff are approximate.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.linalg.lapack import dsytrd, dormqr


@dataclass
class ProjectionResult:
    matrix: np.ndarray
    shift: float
    upper_dual: float
    iterations: int
    converged: bool
    residual: float


@dataclass
class BatchProjectionResult:
    matrix: np.ndarray
    shift: np.ndarray
    upper_dual: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    residual: np.ndarray


def _validate(X, tol, max_iter):
    X=np.asarray(X,dtype=float)
    if X.ndim<2 or X.shape[-1]!=X.shape[-2] or X.shape[-1]<2:
        raise ValueError('square moment matrix of order at least 2 required')
    if not np.isfinite(X).all():
        raise ValueError('finite moment matrices required')
    if not np.isfinite(tol) or tol<=0 or isinstance(max_iter,bool) or int(max_iter)!=max_iter or max_iter<1:
        raise ValueError('positive tolerance and integer iteration limit required')
    scale=np.linalg.norm(X,axis=(-2,-1))
    if np.any(np.max(np.abs(X-X.swapaxes(-1,-2)),axis=(-2,-1))>1e-12*np.maximum(scale,np.finfo(float).tiny)):
        raise ValueError('symmetric moment matrices required')
    return (X+X.swapaxes(-1,-2))*.5,np.maximum(scale,np.finfo(float).tiny)


def _phi_alpha(w,U):
    """Phi and negative derivative; zero eigenvalues use derivative zero."""
    p=U[0]**2;pos=w>0;neg=~pos
    phi=float(p@np.maximum(w,0.))
    alpha=float(np.sum(p[pos])**2)
    if np.any(pos) and np.any(neg):
        ratios=w[pos,None]/(w[pos,None]-w[None,neg])
        alpha+=2*float(np.sum((p[pos,None]*p[None,neg])*ratios))
    return phi,min(1.,max(0.,alpha))


def project_matrix(X0, *, tol=1e-11, max_iter=80, method='newton', initial_shift=None):
    """Solve one original-unit problem, returning an explicit convergence flag.

    `newton`: dense symmetric eigensolver. `tridiagonal`: one Householder
    reduction fixing e0, followed by repeated tridiagonal eigensolves.
    `bisect`: independent conservative scalar iteration for diagnostics.
    `initial_shift` is only an initialization; no eigenvectors are reused
    across different input matrices. The returned residual is scale-relative.
    """
    X0,scale=_validate(X0,tol,max_iter)
    if X0.ndim!=2:raise ValueError('project_matrix expects one matrix')
    if method not in ('newton','tridiagonal','bisect'):
        raise ValueError('unknown scalar projection method')
    if initial_shift is not None and (not np.isfinite(initial_shift) or initial_shift<0):
        raise ValueError('initial_shift must be finite and nonnegative')
    # The fixed q<=1 box needs an absolute guard even if A dominates ||X0||.
    # Use both relative and absolute scalar accuracy; never clip q afterwards.
    stop=tol*min(float(scale),1.)
    a=float(X0[0,0]);N=len(X0);map_vectors=None
    if method=='tridiagonal':
        if N==2:
            # Already tridiagonal; no trailing Householder transform exists.
            diag=np.diag(X0).copy();off=np.array([X0[1,0]])
        else:
            reduced,diag,off,tau,info=dsytrd(X0,lower=1,overwrite_a=False)
            if info!=0:
                raise np.linalg.LinAlgError(f'dsytrd failed with info={info}')
            # Lower SYTRD has Q=diag(1,Qsub). Qsub's QR-format reflectors
            # are reduced[1:,:-1]; its upper triangle is ignored by ORMQR.
            # Thus Q fixes e0, so every scalar shift uses this same reduction.
            reflectors=np.asfortranarray(reduced[1:,:-1])
            def map_vectors(vectors):
                tail=np.asfortranarray(vectors[1:,:])
                _,work,info=dormqr('L','N',reflectors,tau,tail,lwork=-1)
                if info!=0:
                    raise np.linalg.LinAlgError(f'dormqr workspace query failed with info={info}')
                transformed,_,info=dormqr('L','N',reflectors,tau,tail,lwork=int(work[0]))
                if info!=0:
                    raise np.linalg.LinAlgError(f'dormqr failed with info={info}')
                return np.vstack((vectors[0,:],transformed))
        def spectral(t):
            d=diag.copy();d[0]-=t
            return eigh_tridiagonal(d,off,check_finite=False,lapack_driver='stevd')
    else:
        def spectral(t):
            H=X0.copy();H[0,0]-=t
            return eigh(H,check_finite=False,driver='evd',overwrite_a=True)
    w,U=spectral(0.);phi,alpha=_phi_alpha(w,U);evaluations=1
    def finish(t,w,U,residual,converged):
        vectors=U if map_vectors is None else map_vectors(U)
        P=(vectors*np.maximum(w,0.))@vectors.T
        P=(P+P.T)*.5
        return ProjectionResult(P,float(t),float(max(0.,a+t-1.)),evaluations,bool(converged),float(residual/scale))
    if 0<=a<=1 and w[0]>=0:
        return ProjectionResult(X0.copy(),0.,0.,1,True,0.)
    f=min(a,1.)-phi
    if abs(f)<=stop:
        return finish(0.,w,U,abs(f),True)
    beta=max(0.,phi-a)
    if phi<=1.:
        lo=.5*beta;hi=beta
    else:
        lo=0.;hi=max(beta,a-1.,float(scale),np.finfo(float).tiny)
        for _ in range(80):
            wh,Uh=spectral(hi);evaluations+=1
            ph,_=_phi_alpha(wh,Uh)
            if min(a+hi,1.)-ph>=0:break
            lo=hi;hi*=2.
        else:
            return finish(hi,wh,Uh,abs(min(a+hi,1.)-ph),False)
    t=float(np.clip(beta/(1.+alpha),lo,hi)) if initial_shift is None else float(np.clip(initial_shift,lo,hi))
    if method=='bisect':t=.5*(lo+hi)
    for _ in range(max_iter):
        w,U=spectral(t);evaluations+=1
        phi,alpha=_phi_alpha(w,U);f=min(a+t,1.)-phi
        if abs(f)<=stop:
            return finish(t,w,U,abs(f),True)
        if f<0:lo=t
        else:hi=t
        derivative=(1. if a+t<1. else 0.)+alpha
        proposal=t-f/derivative if derivative>np.finfo(float).eps and method!='bisect' else .5*(lo+hi)
        if not lo<=proposal<=hi or proposal==t:proposal=.5*(lo+hi)
        t=float(proposal)
    # Keep the matrix and shift from the same last evaluation, not the proposal.
    evaluated_shift=float(X0[0,0]-np.sum(w*(U[0]**2)))
    return finish(evaluated_shift,w,U,abs(f),False)


def _batch_phi_alpha(w,U):
    p=U[:,0,:]**2;pos=w>0
    phi=np.sum(p*np.maximum(w,0.),axis=-1)
    alpha=np.sum(np.where(pos,p,0.),axis=-1)**2
    # This bounded O(batch*N^2) buffer is chunked by project_batch.
    denom=w[:,:,None]-w[:,None,:]
    mask=pos[:,:,None]&~pos[:,None,:]
    ratio=np.divide(w[:,:,None],denom,out=np.zeros_like(denom),where=mask)
    alpha+=2*np.sum(p[:,:,None]*p[:,None,:]*ratio,axis=(-2,-1))
    return phi,np.clip(alpha,0.,1.)


def project_batch(X0, *, tol=1e-11, max_iter=80, chunk_size=128, initial_shift=None, method='newton'):
    """Vectorized independent projections; arbitrary leading dimensions.

    Batching changes only scheduling, never joint noise generation. D>=32 can
    use method='tridiagonal' for a sequential per-input alternative. Inputs
    are never mutated. Unconverged entries remain explicitly marked false.
    """
    X0,scales=_validate(X0,tol,max_iter)
    if isinstance(chunk_size,bool) or int(chunk_size)!=chunk_size or chunk_size<1:raise ValueError('positive integer chunk_size required')
    lead=X0.shape[:-2];N=X0.shape[-1];flat=X0.reshape(-1,N,N);scales=scales.reshape(-1)
    if initial_shift is not None:
        starts=np.broadcast_to(np.asarray(initial_shift,float),lead).reshape(-1)
        if np.any(~np.isfinite(starts)) or np.any(starts<0):raise ValueError('invalid initial shifts')
    else:starts=None
    out=np.empty_like(flat);shifts=np.zeros(len(flat));duals=np.zeros(len(flat));iterations=np.zeros(len(flat),int)
    success=np.zeros(len(flat),bool);residuals=np.full(len(flat),np.inf)
    if method!='newton':
        for i,M in enumerate(flat):
            r=project_matrix(M,tol=tol,max_iter=max_iter,method=method,initial_shift=None if starts is None else starts[i])
            out[i],shifts[i],duals[i],iterations[i],success[i],residuals[i]=r.matrix,r.shift,r.upper_dual,r.iterations,r.converged,r.residual
        return BatchProjectionResult(out.reshape(X0.shape),shifts.reshape(lead),duals.reshape(lead),iterations.reshape(lead),success.reshape(lead),residuals.reshape(lead))
    for begin in range(0,len(flat),chunk_size):
        end=min(begin+chunk_size,len(flat));M=flat[begin:end];scale=scales[begin:end];a=M[:,0,0];K=len(M)
        stop=tol*np.minimum(scale,1.)
        w,U=np.linalg.eigh(M);phi,alpha=_batch_phi_alpha(w,U)
        t=np.zeros(K);lo=np.zeros(K);beta=np.maximum(phi-a,0.);hi=beta.copy()
        evals=np.ones(K,int);f=np.minimum(a,1.)-phi
        done=np.abs(f)<=stop
        feasible=(a>=0)&(a<=1)&(w[:,0]>=0)
        out[begin:end]=(U*np.maximum(w,0.)[:,None,:])@U.swapaxes(-1,-2)
        out[begin:end][feasible]=M[feasible]
        residuals[begin:end]=np.abs(f)/scale;success[begin:end]=done|feasible
        active=~(done|feasible)
        # Capped exceptional inputs use the scalar certified implementation.
        cap_possible=active&(phi>1.)
        for i in np.flatnonzero(cap_possible):
            r=project_matrix(M[i],tol=tol,max_iter=max_iter,initial_shift=None if starts is None else starts[begin+i])
            out[begin+i],shifts[begin+i],duals[begin+i],iterations[begin+i],success[begin+i],residuals[begin+i]=r.matrix,r.shift,r.upper_dual,r.iterations,r.converged,r.residual
        active[cap_possible]=False
        lo=.5*beta
        t=np.clip(beta/(1.+alpha),lo,hi) if starts is None else np.clip(starts[begin:end],lo,hi)
        for _ in range(max_iter):
            idx=np.flatnonzero(active)
            if len(idx)==0:break
            H=M[idx].copy();H[:,0,0]-=t[idx]
            wi,Ui=np.linalg.eigh(H);ph,al=_batch_phi_alpha(wi,Ui);fi=a[idx]+t[idx]-ph
            evals[idx]+=1
            out[begin+idx]=(Ui*np.maximum(wi,0.)[:,None,:])@Ui.swapaxes(-1,-2)
            shifts[begin+idx]=t[idx];residuals[begin+idx]=np.abs(fi)/scale[idx]
            converged=np.abs(fi)<=stop[idx]
            success[begin+idx]=converged
            active[idx[converged]]=False
            lo[idx]=np.where(fi<0,t[idx],lo[idx]);hi[idx]=np.where(fi>=0,t[idx],hi[idx])
            proposal=t[idx]-fi/(1.+al)
            bad=(proposal<lo[idx])|(proposal>hi[idx])|(proposal==t[idx])
            proposal[bad]=.5*(lo[idx[bad]]+hi[idx[bad]])
            t[idx]=proposal
        normal=~cap_possible
        iterations[begin:end][normal]=evals[normal]
    return BatchProjectionResult(out.reshape(X0.shape),shifts.reshape(lead),duals.reshape(lead),iterations.reshape(lead),success.reshape(lead),residuals.reshape(lead))


@lru_cache(None)
def _indices(D):
    pairs=[(i,j) for j in range(D) for i in range(j+1)]
    i,j=np.array(pairs,dtype=int).T
    return i,j,np.where(i==j,np.sqrt(2.),1.)


def normalized_to_matrix(u,D):
    """u=(u0,u1,svec(A)/sqrt(2)), upper triangle column order."""
    u=np.asarray(u,float);width=1+D+D*(D+1)//2
    if u.shape[-1]!=width:raise ValueError('wrong normalized moment width')
    X=np.zeros(u.shape[:-1]+(D+1,D+1));X[...,0,0]=u[...,0]
    X[...,0,1:]=u[...,1:1+D];X[...,1:,0]=u[...,1:1+D]
    i,j,factor=_indices(D);X[...,1+i,1+j]=u[...,1+D:]*factor;X[...,1+j,1+i]=X[...,1+i,1+j]
    return X


def matrix_to_normalized(X):
    X=np.asarray(X,float);D=X.shape[-1]-1;i,j,factor=_indices(D)
    return np.concatenate((X[...,0,0,None],X[...,0,1:],X[...,1+i,1+j]/factor),axis=-1)


class WeightedMomentProjector:
    """Opt-in drop-in psd_box adapter. Other cones are deliberately unsupported."""
    def __init__(self,D,*,tol=1e-11,method='newton'):
        if isinstance(D,bool) or int(D)!=D or D<1:raise ValueError('positive integer D required')
        self.D=int(D);self.tol=tol;self.method=method
    def _project_arm(self,name,q,v,B,sd):
        if name!='psd_box':raise ValueError('weighted solver supports only the paper psd_box cone')
        if not np.isfinite(sd) or sd<=0:raise ValueError('positive finite sd required')
        X=np.empty((self.D+1,self.D+1));X[0,0]=q;X[0,1:]=v;X[1:,0]=v;X[1:,1:]=B
        r=project_matrix(X,tol=self.tol,method=self.method)
        return dict(q=r.matrix[0,0],v=r.matrix[1:,0],B=r.matrix[1:,1:],ok=r.converged,
                    status='Solved' if r.converged else 'MaxIterations',iterations=r.iterations,
                    residual=r.residual,shift=r.shift,upper_dual=r.upper_dual)
    def project_normalized(self,u,*,chunk_size=128):
        r=project_batch(normalized_to_matrix(u,self.D),tol=self.tol,method=self.method,chunk_size=chunk_size)
        return matrix_to_normalized(r.matrix),r

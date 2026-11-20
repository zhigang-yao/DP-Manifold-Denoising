"""Exact finite-spectral Gaussian mechanisms with different computational factorizations.

joint: one shared spectrum, all bandwidths coupled; dense initial Cholesky.
per_scale: independent h-specific RKHS releases, sqrt(H) noise multiplier;
           one Cholesky/conditional union per bandwidth; no dense cross-h block.
Neither path uses Vecchia, inducing points, truncated Fourier features or independent-query noise.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from modules.md.finite_field import FiniteSpectrumKernel
from modules.md.highd_kernel import HighDFiniteKernel
from modules.md.gaussian_field import FullField, cross_density_readout
from modules.md.moments import empirical, empirical_density_grid
from modules.md.structured_shared_field import SharedStructuredField, JetMarginal

@dataclass(frozen=True)
class FieldPlan:
    backend: str
    D: int
    Q: int
    H: int
    p: int
    factor_bytes: int
    marginal_sd_multiplier: float
    largest_factor_dimension: int

def plan_field(D, Q, H, backend):
    p=(D+1)*(D+2)//2
    if backend == "joint":
        size=Q*H*p
        bytes_=8*size*size
        multiplier=1.0
    elif backend == "per_scale":
        size=Q*p
        bytes_=8*H*size*size
        multiplier=math.sqrt(H)
    elif backend=="structured_per_query":
        # Q independent one-query fields (span rank 0), each shared over all H.
        core,vec,tensor=2*H,H,H
        size=core
        bytes_=8*Q*(core**2+vec**2+tensor**2)
        multiplier=math.sqrt(Q) if Q else 1.0
    elif backend in ("structured_shared","structured_independent"):
        r=min(D,max(0,Q-1))
        q=D-r
        core=Q*H*(1+r+r*(r+1)//2+int(q>0))
        vec=Q*H*(r+1) if q else 0
        tensor=Q*H if q>=2 else 0
        size=max(core,vec,tensor)
        bytes_=8*(core**2+vec**2+tensor**2)
        multiplier=math.sqrt(H) if backend=="structured_independent" else 1.0
    else:
        raise ValueError("unknown field backend")
    return FieldPlan(backend,D,Q,H,p,bytes_,multiplier,size)

def scalar_isotropic_variance(covariance, *, rtol=1e-12, row_chunk=128):
    """Recognize a true numerical c*I covariance, including declared jitter.

    This checks EVERY matrix entry without allocating a second full identity.
    If the finite-quadrature covariance is anisotropic, the caller must retain
    its original dense factor and must not change the marginal Gaussian law.
    """
    cov=np.asarray(covariance,dtype=float)
    if cov.ndim!=2 or cov.shape[0]!=cov.shape[1] or cov.shape[0]<1:
        raise ValueError("expected square marginal covariance")
    variance=float(cov[0,0])
    if not np.isfinite(variance) or variance<=0:
        raise ValueError("marginal variance must be positive and finite")
    threshold=rtol*variance
    p=len(cov)
    for lo in range(0,p,row_chunk):
        hi=min(p,lo+row_chunk)
        residual=cov[lo:hi,:].copy()
        residual[np.arange(hi-lo),np.arange(lo,hi)]-=variance
        if not np.isfinite(residual).all() or np.max(np.abs(residual))>threshold:
            raise ValueError("marginal is not a verified scalar times identity")
    return variance


class Field:
    def __init__(self, reference, hs, base_sd, rng, *, backend, order=128,
                 cache_dir=None, memory_limit_gib=8.0,public_factor_cache=None):
        self.reference=np.asarray(reference,float)
        self.hs=np.asarray(hs,float)
        self.D=self.reference.shape[1]
        self.backend=backend
        self.plan=plan_field(self.D,0, len(hs), backend)
        self.rng=rng
        self.base_sd=float(base_sd)
        self.sd=self.base_sd*(math.sqrt(len(hs))
           if backend in ("per_scale","structured_independent") else 1.)
        self.order=order
        self.cache_dir=Path(cache_dir or "results/cache")
        self.memory_limit_bytes=int(memory_limit_gib*2**30)
        self.children=[]
        self.local_noise_modes=[]
        self.impl=None
        self.structured=None
        self.public_factor_cache=public_factor_cache

    def initial(self, queries):
        x=np.asarray(queries,float)
        Q=len(x); H=len(self.hs)
        self.plan=plan_field(self.D,Q,H,self.backend)
        if self.plan.factor_bytes > self.memory_limit_bytes:
            raise MemoryError(f"initial Cholesky factors alone need {self.plan.factor_bytes/2**30:.2f} GiB")
        if self.backend in ("structured_shared","structured_independent"):
            factory=(SharedStructuredField if self.public_factor_cache is None
                     else self.public_factor_cache.new_field)
            self.structured=factory(
                x,self.hs,order=self.order,memory_limit_gib=self.memory_limit_bytes/2**30,
                bandwidth_correlation=("independent" if self.backend=="structured_independent"
                                       else "shared"))
            moments=self.structured.initial(self.reference,self.sd,self.rng)
            self.local_covariances=self.structured.marginals
            self.local_noise_modes=["structured_jet"]*H
            self.factor_dimensions=[self.structured.S,self.structured.V,self.structured.T]
            self.initial_timings=dict(structured_setup_seconds=self.structured.plan["setup_seconds"])
            self.design=dict(
                             mode=("finite_joint_structured_shared"
                                   if self.backend=="structured_shared"
                                   else "independent_bandwidth_structured"),
                             spectrum=("uniform finite mixture g=h"
                                       if self.backend=="structured_shared"
                                       else "independent single-atom spectra g=h"),
                             covariance=("joint across all public queries and all bandwidths"
                                         if self.backend=="structured_shared"
                                         else "joint across public queries within each h; zero cross-h covariance"),
                             gaussian_noise_multiplier=self.sd/self.base_sd,
                             public_query_span_rank=self.structured.r,
                             factor_plan=self.structured.plan)
            self.plan=plan_field(self.D,Q,H,self.backend)
        elif self.backend=="structured_per_query":
            # Case 4 comparator: one independent release per public query, each a
            # shared-over-H structured field; Q-fold Gaussian composition gives sqrt(Q).
            self.sd=self.base_sd*math.sqrt(Q)
            self.children=[SharedStructuredField(x[i:i+1],self.hs,order=self.order,
                               memory_limit_gib=self.memory_limit_bytes/2**30) for i in range(Q)]
            moments=np.concatenate([child.initial(self.reference,self.sd,self.rng)
                                    for child in self.children],axis=0)
            first=self.children[0].marginals
            if any(not np.array_equal(m.scalar,f.scalar) or m.vector_variance!=f.vector_variance
                   or m.traceless_variance!=f.traceless_variance
                   for child in self.children[1:] for m,f in zip(child.marginals,first)):
                raise ArithmeticError("per-query marginals must not depend on the query")
            self.local_covariances=first
            self.local_noise_modes=["structured_jet"]*H
            self.factor_dimensions=[child.S for child in self.children]
            self.initial_timings=dict(structured_setup_seconds=sum(
                child.plan["setup_seconds"] for child in self.children))
            self.design=dict(mode="independent_per_query_structured",
                             spectrum="uniform finite mixture g=h",
                             covariance="joint across all bandwidths within each query; zero cross-query covariance",
                             gaussian_noise_multiplier=self.sd/self.base_sd)
        elif self.backend=="joint":
            kernel=(FiniteSpectrumKernel(self.D,self.hs,self.order) if self.D in (2,3)
                    else HighDFiniteKernel(self.D,self.hs,self.order))
            self.impl=FullField(self.reference,self.hs,float(self.hs[0]),self.sd,
                                self.rng,order=self.order,cache_dir=self.cache_dir,kernel=kernel)
            moments=self.impl.initial(x)
            self.local_covariances=self.impl.local_covariances
            self.local_noise_modes=["dense"]*H
            self.design=self.impl.kernel.design
            self.factor_dimensions=[len(self.impl.L)]
            self.initial_timings=self.impl.initial_timings
        else:
            outputs=[]
            self.local_covariances=[]
            self.factor_dimensions=[]
            self.initial_timings=[]
            for h in self.hs:
                kernel=HighDFiniteKernel(self.D,[float(h)],self.order,scales=[float(h)],weights=[1.])
                child=FullField(self.reference,[float(h)],float(h),self.sd,self.rng,
                                order=self.order,cache_dir=self.cache_dir,kernel=kernel)
                values=child.initial(x)
                outputs.append(values[:,0,:])
                self.children.append(child)
                # Exact single-h Hermite coordinates are marginally isotropic.
                # Finite quadrature may introduce numerical anisotropy, so verify
                # all entries. Fall back to the original dense covariance if not.
                local=child.local_covariances[0]
                try:
                    variance=scalar_isotropic_variance(local)
                except ValueError:
                    self.local_covariances.append(local)
                    self.local_noise_modes.append("dense_fallback")
                else:
                    self.local_covariances.append(variance)
                    child.local_covariances[0]=variance
                    self.local_noise_modes.append("isotropic_verified")
                self.factor_dimensions.append(len(child.L))
                self.initial_timings.append(child.initial_timings)
            moments=np.stack(outputs,axis=1)
            self.design=dict(mode="independent_per_scale",g_equals_h=True,
                             H=H,sd_multiplier=math.sqrt(H),
                             covariance="block-diagonal in bandwidth; joint across queries within bandwidth")
        self.moments=moments
        return moments

    def terminal_masses(self,candidates):
        """One terminal joint Gaussian conditional extension; all candidate coordinates included."""
        values=np.asarray(candidates,float)
        Q,C,D=values.shape
        if self.backend in ("structured_shared","structured_independent"):
            noise,metadata=self.structured.conditional_masses(values)
            out=empirical_density_grid(self.reference,values.reshape(Q*C,D),self.hs).reshape(Q,C,len(self.hs))
            out+=self.sd*noise
            return out,metadata
        if self.backend=="structured_per_query":
            # Query i's candidates are read from query i's own release only.
            noise=np.concatenate([child.conditional_masses(values[i:i+1])[0]
                                  for i,child in enumerate(self.children)],axis=0)
            out=np.empty_like(noise)
            for k,h in enumerate(self.hs):
                out[:,:,k]=empirical(self.reference,values.reshape(Q*C,D),
                                       float(h),"density")[:,0].reshape(Q,C)+self.sd*noise[:,:,k]
            return out,dict(mode="structured_per_query",queries=Q)
        if self.backend=="joint":
            result=cross_density_readout(self.impl,values,empirical=empirical,
                draw_identity="final_joint_soft_hard_v1",allow_candidate_count=True,
                seal_history=True,memory_limit_bytes=self.memory_limit_bytes)
            return result["values"],dict(new_scalar_count=result["metadata"]["new_scalar_count"],
                                         mode="joint",max_conditional_jitter=result["metadata"].get("conditional_extra_jitter"))
        output=np.empty((Q,C,len(self.hs)))
        counts=[]
        for j,child in enumerate(self.children):
            result=cross_density_readout(child,values,empirical=empirical,
                draw_identity=f"final_per_scale_h{j}_v1",allow_candidate_count=True,
                seal_history=True,memory_limit_bytes=self.memory_limit_bytes)
            output[:,:,j]=result["values"][:,:,0]
            counts.append(result["metadata"]["new_scalar_count"])
        return output,dict(new_scalar_count_by_bandwidth=counts,mode="per_scale")

    def release_metadata(self):
        return dict(backend=self.backend,design=self.design,sd=self.sd,
                    plan=self.plan.__dict__,factor_dimensions=self.factor_dimensions,
                    initial_timings=self.initial_timings,
                    local_noise_modes=self.local_noise_modes,
                    auxiliary_sampling="independent marginal per query-bandwidth cell; no joint field draws",
                    numerical_ideal_mechanism_not_machine_certified=True)

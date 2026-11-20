"""Four-method paired simulation implementation: NP/DP x Hard/Soft.

Only private references and public queries enter the estimator. Latent points and
the true manifold are never arguments. Research results include NP controls and
are NOT themselves safe DP release bundles.
"""
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
import hashlib
import json
import numpy as np

from modules.config import MethodConfig
from modules.privacy import analytic_gaussian_sigma
from modules.md.moments import empirical, unpack, empirical_density_grid
from modules.md.projection import WeightedMomentProjector
from modules.md.spectral import canonical_eigenpairs
from modules.md.bandwidth import evaluate, select_estimate
from modules.md.backends import Field
from modules.md.structured_shared_field import JetMarginal, SharedStructuredField
from modules.md.shared_memory import decoder_workspace,require_memory
from modules.md.inner_cv import FrobeniusMoment, operator_second_moment_cv

METHODS=("RAW","NP-MD-H","NP-MD-S","DP-MD-H","DP-MD-S")

def _rng(config,unit_id,tag,*indices):
    digest=np.frombuffer(hashlib.sha256(str(unit_id).encode()).digest()[:16],dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([
        int(config.auxiliary_seed), *map(int,digest),int(tag),*map(int,indices)]))

def _gap_weights(values, nu, alpha0):
    s=np.maximum(values,0)
    b=np.maximum(s,nu[:,None])
    T=np.maximum(np.sqrt(b[:,-1]*b[:,0]),alpha0*b[:,-1])
    star=np.max(np.where(b<=T[:,None],b,-np.inf),axis=-1)
    if not np.isfinite(star).all():
        raise ArithmeticError("empty spectral cluster")
    lam2=b[:,-1]*star
    den=s*s+lam2[:,None]
    weights=np.divide(lam2[:,None],den,out=np.ones_like(s),where=den>0)
    return np.sqrt(lam2),weights

class Decoder:
    def __init__(self,D,d,hs,n,sd,covs,config,unit_id,*,private,
                 auxiliary_covariances=None,auxiliary_sd=None):
        self.D,self.d,self.hs,self.n,self.sd=D,d,np.asarray(hs),n,sd
        self.auxiliary_sd=float(sd if auxiliary_sd is None else auxiliary_sd)
        if private and (not np.isfinite(self.auxiliary_sd) or self.auxiliary_sd<=0):
            raise ValueError("auxiliary Gaussian scale must be positive")
        self.covs,self.cfg,self.unit_id,self.private=covs,config,unit_id,private
        self.psd=None
        self.noise_factors=[]
        self.frobenius_moments=[]
        self.cv_diagnostics=dict(cells=0,nonpositive_safeguards=0)
        if private:
            mode=config.psd_solver
            mode="newton" if (mode=="auto" and D<32) or mode=="dense" else "tridiagonal"
            self.psd=WeightedMomentProjector(D,tol=config.psd_tol,method=mode)
            auxiliary_marginals=(covs if auxiliary_covariances is None
                                 else auxiliary_covariances)
            if len(auxiliary_marginals)!=len(self.hs):
                raise ValueError("auxiliary marginal covariance grid does not match bandwidths")
            for c in auxiliary_marginals:
                if config.inner_estimator=="frobenius_cv":
                    self.frobenius_moments.append(FrobeniusMoment(c,D))
                if isinstance(c,JetMarginal):
                    self.noise_factors.append(c)
                elif np.ndim(c)==0:
                    variance=float(c)
                    if not np.isfinite(variance) or variance<=0:
                        raise ValueError("invalid isotropic marginal variance")
                    self.noise_factors.append(np.sqrt(variance))
                else:
                    self.noise_factors.append(np.linalg.cholesky(c))
        self.threshold=None
        if config.pointwise_fallback:
            if config.gate_threshold is not None:
                self.threshold=float(config.gate_threshold)
            else:
                gates=json.loads(Path(__file__).with_name("gate_thresholds.json").read_text())
                values=[float(r["threshold"]) for r in gates if r["D"]==D]
                if not values:raise ValueError("missing public mass-gate threshold for this D/release")
                self.threshold=values[0]

    def sample_marginal_noise(self,rng,k,count):
        """Independent N(0,C_k) draws; no cross-query or cross-h covariance needed."""
        if not self.private:
            raise ValueError("marginal perturbations are private-decoder only")
        p=(self.D+1)*(self.D+2)//2
        factor=self.noise_factors[k]
        if isinstance(factor,JetMarginal):
            return factor.sample(rng,count)
        z=rng.standard_normal((count,p))
        return z*factor if np.ndim(factor)==0 else z@factor.T

    def decode(self,t,x,k,*,query_ids=None,outer_reps=None):
        t,x=np.asarray(t,float),np.asarray(x,float)
        N,p=t.shape
        if x.shape!=(N,self.D) or not np.isfinite(t).all() or not np.isfinite(x).all():
            raise ValueError("nonfinite or shape-invalid moment decode input")
        if self.private:
            projected, certificate=self.psd.project_normalized(t,chunk_size=self.cfg.psd_chunk_size)
            if not bool(np.all(certificate.converged)):
                raise ArithmeticError(f"weighted PSD projection failed {int(np.sum(~certificate.converged))} cells")
            q,v,A=unpack(projected,self.D)
        else:
            q,v,A=unpack(t,self.D)
        den=np.maximum(q,1/self.n)
        m=self.hs[k]*v/den[:,None]
        C=self.hs[k]**2*A/den[:,None,None]-m[:,:,None]*m[:,None,:]
        C=(C+C.swapaxes(-1,-2))*.5
        values,U=np.linalg.eigh(C)
        nu=np.zeros(N)
        if self.private:
            query_ids=np.arange(N) if query_ids is None else np.asarray(query_ids)
            outer_reps=np.full(N,-1) if outer_reps is None else np.asarray(outer_reps)
            for lo in range(0,N,16):
                hi=min(N,lo+16)
                noises=np.stack([self.sample_marginal_noise(
                    _rng(self.cfg,self.unit_id,101,int(query_ids[i]),k,int(outer_reps[i])+1),
                    k,self.cfg.inner_B) for i in range(lo,hi)])
                g0,g1,g2=unpack(noises,self.D)
                mm,cc=m[lo:hi],C[lo:hi]
                h=self.hs[k]
                pert=(h*h*g2-h*(g1[:,:,:,None]*mm[:,None,None,:]+mm[:,None,:,None]*g1[:,:,None,:])
                      +(mm[:,:,None]*mm[:,None,:]-cc)[:,None]*g0[:,:,None,None])
                op=np.max(np.abs(np.linalg.eigvalsh(pert)),axis=-1)
                if self.cfg.inner_estimator=="frobenius_cv":
                    exact=self.frobenius_moments[k].expectation(mm,cc,h)
                    corrected=operator_second_moment_cv(op*op,np.sum(pert*pert,axis=(-2,-1)),exact,D=self.D)
                    mu=corrected['estimate']
                    self.cv_diagnostics['cells']+=hi-lo
                    self.cv_diagnostics['nonpositive_safeguards']+=int(np.sum(corrected['safeguarded']))
                else:
                    mu=np.mean(op*op,axis=1)
                nu[lo:hi]=self.auxiliary_sd/den[lo:hi]*np.sqrt(mu)
        lam,w=_gap_weights(values,nu,self.cfg.alpha0)
        soft=x+np.einsum("nij,nj->ni",U,np.einsum("nji,nj->ni",U,m)*w)
        eig,V=canonical_eigenpairs(C)
        Nbasis=V[:,:,:self.D-self.d]
        hard=x+np.einsum("nij,nkj,nk->ni",Nbasis,Nbasis,m)
        if self.threshold is not None:
            if self.private:
                marginal_q_variance=(self.covs[k].q_variance if isinstance(self.covs[k],JetMarginal)
                   else float(self.covs[k]) if np.ndim(self.covs[k])==0
                   else float(self.covs[k][0,0]))
                ok=(q>self.threshold*self.sd*np.sqrt(marginal_q_variance))&(q>=1/self.n)
            else:
                ok=q>=1/self.n
            soft=np.where(ok[:,None],soft,x)
            hard=np.where(ok[:,None],hard,x)
        else:
            ok=q>=1/self.n  # diagnostic only: never overwrites an output
        if not np.isfinite(soft).all() or not np.isfinite(hard).all():
            raise FloatingPointError("nonfinite decoder; not a RAW fallback")
        return soft,hard,dict(gate_eligible=ok,lambda_=lam,nu=nu,q=q)


class _RunningPositionVariance:
    """Stable vector-valued Welford merging, with sample ddof=1."""

    def __init__(self, D):
        self.count=0
        self.mean=np.zeros(D,dtype=float)
        self.m2=np.zeros(D,dtype=float)

    def add(self, values):
        a=np.asarray(values,dtype=float)
        if a.ndim!=2 or a.shape[1]!=len(self.mean) or len(a)<1 or not np.isfinite(a).all():
            raise ValueError("finite auxiliary output batch expected")
        count=len(a)
        batch_mean=np.mean(a,axis=0)
        batch_m2=np.sum((a-batch_mean)**2,axis=0)
        if self.count==0:
            self.count=count
            self.mean=batch_mean
            self.m2=batch_m2
            return
        total=self.count+count
        shift=batch_mean-self.mean
        self.m2+=batch_m2+(shift*shift)*(self.count*count/total)
        self.mean+=shift*(count/total)
        self.count=total

    def sample_trace(self):
        if self.count<2:
            raise ValueError("at least two auxiliary draws required")
        return float(np.sum(self.m2)/(self.count-1))


def auxiliary_variances(decoder,moments,queries,*,config,unit_id):
    """Per-(query,h) independent marginal MC; streamed in bounded R chunks.

    This does NOT sample the private field. It is auxiliary post-processing of
    already released moments. The same cell's draw drives both Soft and Hard,
    while different query/h cells use independent deterministic SeedSequences.
    In particular, no R x J x Q x p field tensor or joint Cholesky is used.
    """
    x=np.asarray(queries,float)
    moments=np.asarray(moments,float)
    Q,H,p=moments.shape
    D=x.shape[1]
    soft_variances=np.empty((Q,H))
    hard_variances=np.empty((Q,H))
    max_batch=0
    for k in range(H):
        for i in range(Q):
            generator=_rng(config,unit_id,202,i,k)
            soft_stat=_RunningPositionVariance(D)
            hard_stat=_RunningPositionVariance(D)
            for start in range(0,config.outer_R,config.auxiliary_chunk_size):
                count=min(config.auxiliary_chunk_size,config.outer_R-start)
                noise=decoder.sample_marginal_noise(generator,k,count)
                batch=moments[i,k][None,:]+decoder.auxiliary_sd*noise
                query=np.broadcast_to(x[i],(count,D))
                soft,hard,_=decoder.decode(
                    batch,query,k,
                    query_ids=np.full(count,i,dtype=int),
                    outer_reps=np.arange(start,start+count,dtype=int))
                soft_stat.add(soft)
                hard_stat.add(hard)
                max_batch=max(max_batch,count)
            soft_variances[i,k]=soft_stat.sample_trace()
            hard_variances[i,k]=hard_stat.sample_trace()
    return soft_variances,hard_variances,dict(
        mode="independent_marginal_query_bandwidth_cells",
        field_joint_noise_sampled=False,
        replicas=config.outer_R,
        chunk_size=config.auxiliary_chunk_size,
        maximum_actual_chunk=max_batch,
        per_cell_covariance="same numerical C_k as the initial release marginal",
        seed_namespace=202,
        auxiliary_noise_model=config.auxiliary_noise_model,
        auxiliary_sd=decoder.auxiliary_sd,
        released_moment_sd=decoder.sd,
        shared_soft_hard_within_cell=True,
        marginal_variance_unbiased_ddof=1,
        intercell_finite_R_correlation_changed=True)


@dataclass
class PairedResult:
    estimates: dict
    selected_bandwidths: dict
    scores: dict
    diagnostics: dict

def run(reference,queries,*,epsilon,intrinsic_dimension,rng=None,config=None,
        unit_id="experiment",cache_dir=None,public_field_cache=None):
    """Produce all 2x2 estimates with paired private field and own bandwidth rules."""
    reference=np.asarray(reference,float);x=np.asarray(queries,float)
    if (reference.ndim!=2 or x.ndim!=2 or len(reference)<2 or len(x)<1
            or reference.shape[1]!=x.shape[1] or not np.isfinite(reference).all()
            or not np.isfinite(x).all()):
        raise ValueError("expected finite reference [n,D] and query [Q,D] arrays")
    D=x.shape[1];n=len(reference);Q=len(x)
    if isinstance(intrinsic_dimension,bool) or not isinstance(intrinsic_dimension,(int,np.integer)) or not 0<intrinsic_dimension<D:
        raise ValueError("Hard requires public integer intrinsic_dimension in (0,D)")
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError("epsilon must be positive")
    cfg=(config or MethodConfig()).validate(D)
    hs=np.asarray(cfg.bandwidths,float);H=len(hs)
    rng=np.random.default_rng() if rng is None else rng
    delta=n**(-1.1)
    s=float(np.sqrt(2)*analytic_gaussian_sigma(epsilon,delta,1.)[0]/n)
    started=perf_counter()
    field=Field(reference,hs,s,rng,backend=cfg.backend,order=cfg.quadrature_order,
                cache_dir=cache_dir,memory_limit_gib=cfg.memory_limit_gib,
                public_factor_cache=public_field_cache)
    moments=field.initial(x)
    initial_release_sha256=hashlib.sha256(
        np.ascontiguousarray(moments,dtype=np.float64).tobytes()).hexdigest()
    initial_finished=perf_counter()
    shared_reference_marginals=None
    auxiliary_sd=field.sd
    if cfg.auxiliary_noise_model=="shared_marginal":
        # The real private release stays independent with scale sqrt(H)*s.
        # Only the B/R postprocessing noises use covariance C_k and scale s.
        # Construct the public one-query finite shared spectrum without
        # touching private data or changing the actual field history.
        public_reference=SharedStructuredField(
            np.zeros((1,D),dtype=float),hs,order=cfg.quadrature_order,
            memory_limit_gib=cfg.memory_limit_gib)
        shared_reference_marginals=public_reference.marginals
        auxiliary_sd=s
    if cfg.backend in ("structured_shared","structured_independent","structured_per_query"):
        require_memory(decoder_workspace(D,Q,cfg.inner_B,cfg.auxiliary_chunk_size),
            memory_limit_gib=cfg.memory_limit_gib,stage="structured Hard/Soft decoder")
    dp=Decoder(D,intrinsic_dimension,hs,n,field.sd,field.local_covariances,cfg,unit_id,
               private=True,auxiliary_covariances=shared_reference_marginals,
               auxiliary_sd=auxiliary_sd)
    np_dec=Decoder(D,intrinsic_dimension,hs,n,0,[],cfg,unit_id,private=False)
    dp_soft=np.empty((Q,H,D));dp_hard=np.empty_like(dp_soft)
    np_soft=np.empty_like(dp_soft);np_hard=np.empty_like(dp_soft)
    np_masses=np.empty((Q,H));gate_fraction=[]
    for k,h in enumerate(hs):
        a,b,info=dp.decode(moments[:,k],x,k)
        dp_soft[:,k],dp_hard[:,k]=a,b
        gate_fraction.append(float(np.mean(info["gate_eligible"])))
        nm=empirical(reference,x,float(h))
        np_masses[:,k]=nm[:,0]
        a,b,_=np_dec.decode(nm,x,k)
        np_soft[:,k],np_hard[:,k]=a,b
    candidates_finished=perf_counter()
    var_soft,var_hard,auxiliary_diagnostics=auxiliary_variances(
        dp,moments,x,config=cfg,unit_id=unit_id)
    auxiliary_finished=perf_counter()
    union=np.concatenate([dp_soft,dp_hard],axis=1)
    private_mass,cond=field.terminal_masses(union)
    terminal_finished=perf_counter()
    np_union=np.concatenate([np_soft,np_hard],axis=1)
    np_mass=empirical_density_grid(reference,np_union.reshape(Q*2*H,D),hs).reshape(Q,2*H,H)
    scores={
        "DP-MD-S":evaluate(moments[:,:,0],private_mass[:,:H],var_soft,hs,n),
        "DP-MD-H":evaluate(moments[:,:,0],private_mass[:,H:],var_hard,hs,n),
        "NP-MD-S":evaluate(np_masses,np_mass[:,:H],np.zeros_like(var_soft),hs,n),
        "NP-MD-H":evaluate(np_masses,np_mass[:,H:],np.zeros_like(var_hard),hs,n)}
    outputs={"RAW":x.copy(),
             "DP-MD-S":select_estimate(dp_soft,scores["DP-MD-S"]),
             "DP-MD-H":select_estimate(dp_hard,scores["DP-MD-H"]),
             "NP-MD-S":select_estimate(np_soft,scores["NP-MD-S"]),
             "NP-MD-H":select_estimate(np_hard,scores["NP-MD-H"])}
    diag=field.release_metadata()
    diag.update(n=n,D=D,Q=Q,d=int(intrinsic_dimension),delta=delta,epsilon=epsilon,
                base_sd=s,pointwise_fallback=cfg.pointwise_fallback,
                gate_eligibility_fraction=gate_fraction,conditional=cond,
                auxiliary=auxiliary_diagnostics,
                inner_estimator=dict(mode=cfg.inner_estimator,B=cfg.inner_B,
                    beta=(4./D if cfg.inner_estimator=="frobenius_cv" else None),
                    nonpositive_policy="raw_mc_mean",**dp.cv_diagnostics),
                initial_release_sha256=initial_release_sha256,
                auxiliary_privacy_impact="none, independent postprocessing of calibrated release",
                stage_timings=dict(
                    initial_field_seconds=initial_finished-started,
                    initial_candidate_decode_seconds=candidates_finished-initial_finished,
                    marginal_auxiliary_seconds=auxiliary_finished-candidates_finished,
                    terminal_conditional_seconds=terminal_finished-auxiliary_finished,
                    final_scores_seconds=perf_counter()-terminal_finished,
                    total_seconds=perf_counter()-started),
                selected_h={k:v["h"] for k,v in scores.items()},
                ideal_mechanism_dp_only=True,
                research_result_contains_nonprivate_data=True)
    return PairedResult(outputs,{k:v["h"] for k,v in scores.items()},scores,diag)

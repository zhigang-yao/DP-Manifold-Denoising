"""Exact structural factorization of the finite-spectral shared multiquery field.

Adapted from the user's d3_multiquery_joint_h_D50_capability.zip.
For fixed public queries with affine-span rank r, O(D-r) symmetry reduces
the full Q*H*p Gaussian moment covariance to three representation blocks:
scalar, vector, and traceless tensor. ALL query and bandwidth correlations
are retained. New arbitrary-position scalar mass readings are one Schur
conditional continuation of the SAME field.

Exact for the chosen quadrature-defined Gaussian covariance with isotropic
numerical jitter, up to floating-point; not machine-level DP certification.
No auxiliary Monte Carlo is sampled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import hyp0f1

from modules.md.highd_kernel import HighDFiniteKernel
from modules.md.moments import empirical
from modules.md.shared_memory import (plan_structured_memory,require_memory,validate_budget,
                            require_linalg_runtime,initial_draw_workspace,geometry_workspace)


@dataclass(frozen=True)
class JetMarginal:
    """O(D)-invariant local jet covariance, including isotropic numerical jitter."""
    scalar: np.ndarray
    vector_variance: float
    traceless_variance: float
    D: int

    @property
    def q_variance(self):
        return float(self.scalar[0, 0])

    def sample(self, rng, count):
        D=self.D
        ii,jj=np.triu_indices(D,1)
        count=int(count)
        p=1+D+D*(D+1)//2
        # One fixed-size normal row per replicate gives chunk-invariant draws.
        z=rng.standard_normal((count,2+2*D+len(ii)))
        two=z[:,:2]@cholesky(self.scalar,lower=True,check_finite=False).T
        vec=z[:,2:2+D]*np.sqrt(self.vector_variance)
        diag=z[:,2+D:2+2*D].copy()
        diag-=np.mean(diag,axis=-1,keepdims=True)
        diag*=np.sqrt(self.traceless_variance)
        off=z[:,2+2*D:]*np.sqrt(self.traceless_variance)
        # Diagonal Hermite-2 coordinates are A_ii/sqrt(2);
        # off-diagonal Hermite-2 coordinates are A_ij.
        out=np.empty((count,p),float)
        out[:,0]=two[:,0]
        out[:,1:1+D]=vec
        index=1+D
        which={(int(a),int(b)):t for t,(a,b) in enumerate(zip(ii,jj))}
        for j in range(D):
            for i in range(j+1):
                out[:,index]=(two[:,1]/np.sqrt(D)+diag[:,i]
                             if i==j else off[:,which[i,j]])
                index+=1
        return out


class SharedStructuredField:
    """One finite equal-weight Gaussian spectral field for all Q and H.

    Initial public sites must be fixed before the private release. The
    terminal conditional read is a single adaptive scalar batch, with
    positions selected from earlier noisy moments and independent randomness.
    """

    def __init__(self,queries,bandwidths,*,order=128,jitter=1e-9,
                 memory_limit_gib=8.0,span_tol=1e-10,
                 bandwidth_correlation="shared"):
        started=perf_counter()
        x=np.asarray(queries,float)
        hs=np.asarray(bandwidths,float)
        if (x.ndim!=2 or not len(x) or not np.isfinite(x).all()
                or x.shape[1]<2 or hs.ndim!=1 or not len(hs)
                or not np.isfinite(hs).all() or np.any(hs<=0)
                or np.any(np.diff(hs)<=0)):
            raise ValueError("public queries [Q,D] and positive increasing bandwidths required")
        if bandwidth_correlation not in ("shared","independent"):
            raise ValueError("bandwidth_correlation must be shared or independent")
        geometry_check=require_memory(geometry_workspace(x.shape[1],len(x)),
            memory_limit_gib=memory_limit_gib,stage="structured query geometry")
        self.bandwidth_correlation=bandwidth_correlation
        self.queries=x.copy()
        self.hs=hs
        self.Q,self.D=x.shape
        self.H=len(hs)
        self.p=(self.D+1)*(self.D+2)//2
        self.jitter=float(jitter)
        self.memory_limit_bytes=validate_budget(memory_limit_gib)
        if not self.jitter>0:raise ValueError("positive jitter required")
        # A deterministic singular-value threshold is numerical, not a
        # low-rank approximation to a non-negligible public query direction.
        centered=x-np.mean(x,axis=0)
        _,singular,vt=np.linalg.svd(centered,full_matrices=True)
        threshold=max(float(span_tol),float(singular[0])*span_tol) if len(singular) else span_tol
        rank=int(np.sum(singular>threshold))
        residual=float(np.max(np.abs(centered@vt[rank:].T))) if rank<self.D else 0.
        if residual>max(1e-12,100*span_tol):
            raise ValueError("public affine-span numerical rank not certified")
        self.O=vt.T.copy() # columns are new axes in the original frame
        self.z=x@self.O
        self.r=rank
        self.q=self.D-rank
        self.has_perp=self.q>0
        self.ia,self.ib=np.triu_indices(self.r)
        self.off_i,self.off_j=np.triu_indices(self.q,1)
        self.ncore=1+self.r+len(self.ia)+int(self.has_perp)
        self.nvector=1+self.r
        self.sites=[(i,j) for i in range(self.Q) for j in range(self.H)]
        self.S=len(self.sites)*self.ncore
        self.V=len(self.sites)*self.nvector if self.has_perp else 0
        self.T=len(self.sites) if self.q>=2 else 0
        self.dense_dimension=self.Q*self.H*self.p
        # Three dense representation blocks, before counting work buffers.
        factor_bytes=8*(self.S**2+self.V**2+self.T**2)
        self.plan=dict(D=self.D,Q=self.Q,H=self.H,span_rank=self.r,perp_dim=self.q,
            scalar_block=self.S,vector_block=self.V,traceless_block=self.T,
            largest_factor_dimension=max(self.S,self.V,self.T),
            dense_dimension=self.dense_dimension,
            dense_factor_bytes=8*self.dense_dimension**2,
            reduced_factor_bytes=factor_bytes,
            reduction_factor=(self.dense_dimension**2/(self.S**2+self.V**2+self.T**2)))
        self._memory_order=order
        memory_plan=plan_structured_memory(self.D,self.Q,self.H,span_rank=self.r,order=order)
        self.plan["memory_plan"]=memory_plan.as_dict()
        self.plan["geometry_memory_check"]=geometry_check
        self.plan["initial_memory_check"]=require_memory(
            memory_plan.initial_peak_bytes,memory_limit_gib=memory_limit_gib,
            stage="structured initial factors")
        self.plan["linalg_runtime"]=require_linalg_runtime(max(self.S,self.V,self.T))
        self.kernel=HighDFiniteKernel(self.D,hs,order=order,scales=hs,
             weights=np.ones(len(hs))/len(hs))
        if self.bandwidth_correlation=="shared":
            self._radial_weights=[
                np.column_stack([self.kernel._radial_weights(float(h),float(k),4)[m]
                                 for h in hs for k in hs])
                for m in range(5)]
            self._freq=self.kernel._frequency
            self.single_kernels=None
        else:
            # One independent single-h RKHS process per bandwidth. This is
            # a different calibrated release, not shared kernel truncation.
            self._radial_weights=None
            self._freq=None
            self.single_kernels=[
                HighDFiniteKernel(self.D,[float(h)],order=order,
                                  scales=[float(h)],weights=[1.])
                for h in hs]
        self.marginals=self._make_marginals()
        self._initial_factors()
        self.plan["setup_seconds"]=perf_counter()-started

    def coeffs(self,distances,maximum=4,chunk=256):
        """Radial derivative C_0..C_max as in HighDFiniteKernel.radials."""
        if maximum<0 or maximum>4:raise ValueError("radial derivative order must be 0..4")
        distance=np.asarray(distances,float)
        if self.bandwidth_correlation=="independent":
            ans=np.zeros(distance.shape+(self.H,self.H,maximum+1),dtype=float)
            # The independent h marginal is evaluated with precisely the
            # canonical single-atom finite quadrature, not a new closed form.
            displacement=np.zeros(distance.shape+(self.D,),dtype=float)
            displacement[...,0]=distance
            for k,(h,kernel) in enumerate(zip(self.hs,self.single_kernels)):
                radial=kernel.radials(displacement,float(h),float(h),maximum)
                for m,values in enumerate(radial):
                    ans[...,k,k,m]=values
            return ans
        flat=distance.ravel()
        ans=np.empty((len(flat),self.H*self.H,maximum+1))
        for start in range(0,len(flat),chunk):
            end=min(len(flat),start+chunk)
            freq=flat[start:end,None]*self._freq[None,:]
            z=-.25*freq*freq
            for k in range(maximum+1):
                phi=hyp0f1(self.D/2+k,z)
                ans[start:end,:,k]=phi@self._radial_weights[k]
        return ans.reshape(distance.shape+(self.H,self.H,maximum+1))

    def _make_marginals(self):
        co=self.coeffs(np.array([0.]),2)[0]
        D=self.D
        out=[]
        for j,h in enumerate(self.hs):
            f,c1,c2=co[j,j]
            s=np.array([[f,(D*f+h*h*D*c1)/np.sqrt(2*D)],
                [(D*f+h*h*D*c1)/np.sqrt(2*D),
                 (D*D*f+2*D*D*h*h*c1+h**4*D*(D+2)*c2)/(2*D)]])
            s+=np.eye(2)*self.jitter
            vec=-h*h*c1+self.jitter
            ten=h**4*c2+self.jitter
            if not np.linalg.eigvalsh(s).min()>0 or not min(vec,ten)>0:
                raise ArithmeticError("invalid structured marginal covariance")
            out.append(JetMarginal(s,float(vec),float(ten),D))
        return out

    def _scalar_pair(self,c,dr,h,k):
        f,c1,c2,c3,c4=c;r=len(dr);q=self.q
        B=np.zeros((self.ncore,self.ncore))
        B[0,0]=f
        if r:
            iv=slice(1,1+r);ia=slice(1+r,1+r+len(self.ia))
            B[iv,0]=h*c1*dr;B[0,iv]=-k*c1*dr
            B[iv,iv]=-h*k*(c1*np.eye(r)+c2*np.outer(dr,dr))
            a,b=self.ia,self.ib
            diag=(a==b).astype(float);prod=dr[a]*dr[b]
            weights=np.where(a==b,1/np.sqrt(2),1.)
            B[ia,0]=weights*(diag*f+h*h*(c1*diag+c2*prod))
            B[0,ia]=weights*(diag*f+k*k*(c1*diag+c2*prod))
            iaa=a[None,:];ibb=b[None,:];alpha=np.arange(r)[:,None]
            third=c2*(dr[:,None]*diag[None,:]+dr[iaa]*(alpha==ibb)
                     +dr[ibb]*(alpha==iaa))+c3*dr[:,None]*prod[None,:]
            B[iv,ia]=h*weights[None,:]*(diag[None,:]*c1*dr[:,None]+k*k*third)
            B[ia,iv]=-k*weights[:,None]*(diag[:,None]*c1*dr[None,:]+h*h*third.T)
            ai=a[:,None];bi=b[:,None];aj=a[None,:];bj=b[None,:]
            dAB=diag[:,None];dCD=diag[None,:]
            rrAB=prod[:,None];rrCD=prod[None,:]
            ddAC=ai==aj;ddAD=ai==bj;ddBC=bi==aj;ddBD=bi==bj
            t2=c2*(dAB*dCD+ddAC*ddBD+ddAD*ddBC)
            t3=c3*(dAB*rrCD+ddAC*dr[bi]*dr[bj]+ddAD*dr[bi]*dr[aj]
                   +ddBC*dr[ai]*dr[bj]+ddBD*dr[ai]*dr[aj]+dCD*rrAB)
            t4=c4*rrAB*rrCD
            both=(dAB*dCD*f+dAB*k*k*(c1*dCD+c2*rrCD)
                  +dCD*h*h*(c1*dAB+c2*rrAB)+h*h*k*k*(t2+t3+t4))
            B[ia,ia]=both*weights[:,None]*weights[None,:]
        if q:
            index=self.ncore-1;sq=np.sqrt(q/2)
            B[0,index]=sq*(f+k*k*c1)
            B[index,0]=sq*(f+h*h*c1)
            if r:
                B[1:1+r,index]=h*sq*dr*(c1+k*k*c2)
                B[index,1:1+r]=-k*sq*dr*(c1+h*h*c2)
                a,b=self.ia,self.ib;diag=(a==b).astype(float);prod=dr[a]*dr[b]
                weights=np.where(a==b,1/np.sqrt(2),1.)
                B[1+r:1+r+len(a),index]=weights*sq*(diag*(f+k*k*c1)
                   +h*h*(diag*c1+c2*prod+k*k*(c2*diag+c3*prod)))
                B[index,1+r:1+r+len(a)]=weights*sq*(diag*(f+h*h*c1)
                   +k*k*(diag*c1+c2*prod+h*h*(c2*diag+c3*prod)))
            B[index,index]=.5*(q*f+q*(h*h+k*k)*c1+h*h*k*k*(q+2)*c2)
        return B

    def _vector_pair(self,c,dr,h,k):
        _,c1,c2,c3,_=c
        V=np.zeros((self.nvector,self.nvector))
        V[0,0]=-h*k*c1
        if self.r:
            V[0,1:]=h*k*k*c2*dr
            V[1:,0]=-h*h*k*c2*dr
            V[1:,1:]=h*h*k*k*(c2*np.eye(self.r)+c3*np.outer(dr,dr))
        return V

    def _initial_factors(self):
        Q,H=self.Q,self.H
        diffs=self.z[:,None,:]-self.z[None,:,:]
        co=self.coeffs(np.linalg.norm(diffs,axis=-1),4)
        KS=np.empty((self.S,self.S),order="F")
        KV=np.empty((self.V,self.V),order="F") if self.V else None
        KT=np.empty((self.T,self.T),order="F") if self.T else None
        for i,(a,j) in enumerate(self.sites):
            ia=slice(i*self.ncore,(i+1)*self.ncore)
            va=slice(i*self.nvector,(i+1)*self.nvector)
            for t in range(i+1):
                b,k=self.sites[t]
                ib=slice(t*self.ncore,(t+1)*self.ncore)
                vb=slice(t*self.nvector,(t+1)*self.nvector)
                c=co[a,b,j,k]
                delta=diffs[a,b,:self.r]
                block=self._scalar_pair(c,delta,self.hs[j],self.hs[k])
                KS[ia,ib]=block;KS[ib,ia]=block.T
                if KV is not None:
                    v=self._vector_pair(c,delta,self.hs[j],self.hs[k])
                    KV[va,vb]=v;KV[vb,va]=v.T
                    if KT is not None:
                        KT[i,t]=KT[t,i]=self.hs[j]**2*self.hs[k]**2*c[2]
        for mat in (KS,KV,KT):
            if mat is not None:mat[np.diag_indices(len(mat))]+=self.jitter
        self.Ls=cholesky(KS,lower=True,overwrite_a=True,check_finite=False)
        if not np.shares_memory(self.Ls,KS):raise RuntimeError("initial scalar factor was unexpectedly copied")
        del KS
        self.Lv=cholesky(KV,lower=True,overwrite_a=True,check_finite=False) if KV is not None else None
        if KV is not None and not np.shares_memory(self.Lv,KV):raise RuntimeError("initial vector factor was unexpectedly copied")
        del KV
        self.Lt=cholesky(KT,lower=True,overwrite_a=True,check_finite=False) if KT is not None else None
        if KT is not None and not np.shares_memory(self.Lt,KT):raise RuntimeError("initial tensor factor was unexpectedly copied")

    def _white(self,rng):
        s=rng.standard_normal(self.S)
        if not self.q:return s,None,None,None
        v=rng.standard_normal((self.V,self.q))
        d=rng.standard_normal((self.T,self.q)) if self.T else None
        if d is not None:d-=np.mean(d,axis=1,keepdims=True)
        o=rng.standard_normal((self.T,len(self.off_i))) if self.T else None
        return s,v,d,o

    def _packed_initial(self,white):
        zs,zv,zd,zo=white
        Q,H,D,r,q=self.Q,self.H,self.D,self.r,self.q
        scalar=(self.Ls@zs).reshape(Q,H,self.ncore)
        v=np.zeros((Q,H,D))
        A=np.zeros((Q,H,D,D))
        v[:,:,:r]=scalar[:,:,1:1+r]
        aa,bb=self.ia,self.ib
        if len(aa):
            nums=scalar[:,:,1+r:1+r+len(aa)]*np.where(aa==bb,np.sqrt(2),1.)
            A[:,:,aa,bb]=nums;A[:,:,bb,aa]=nums
        if q:
            part=(self.Lv@zv).reshape(Q,H,self.nvector,q)
            v[:,:,r:]=part[:,:,0,:]
            for i in range(r):
                A[:,:,i,r:]=part[:,:,i+1,:]
                A[:,:,r:,i]=part[:,:,i+1,:]
            trace=scalar[:,:,-1]/np.sqrt(q)
            if self.T:
                diag=(self.Lt@zd).reshape(Q,H,q)
                off=(self.Lt@zo).reshape(Q,H,len(self.off_i))
            else:
                diag=np.zeros((Q,H,q))
                off=np.empty((Q,H,0))
            for i in range(q):A[:,:,r+i,r+i]=np.sqrt(2)*(trace+diag[:,:,i])
            A[:,:,r+self.off_i,r+self.off_j]=off
            A[:,:,r+self.off_j,r+self.off_i]=off
        # Return the natural moment coordinates in the ORIGINAL ambient frame.
        v=v@self.O.T
        A=np.einsum("ia,qhab,jb->qhij",self.O,A,self.O,optimize=True)
        ans=np.empty((Q,H,self.p))
        ans[:,:,0]=scalar[:,:,0]
        ans[:,:,1:1+D]=v
        idx=1+D
        for j in range(D):
            for i in range(j+1):
                ans[:,:,idx]=A[:,:,i,j]/(np.sqrt(2) if i==j else 1.)
                idx+=1
        return ans

    def initial(self,reference,sd,rng):
        if hasattr(self,"initial_noise"):
            raise RuntimeError("structured joint field history is already initialized")
        reference=np.asarray(reference,float)
        if reference.shape[1]!=self.D or not np.isfinite(reference).all():
            raise ValueError("invalid private reference")
        self.plan["initial_draw_memory_check"]=require_memory(
            initial_draw_workspace(self.D,self.Q,self.H,len(reference)),
            memory_limit_gib=self.memory_limit_bytes/2**30,stage="structured initial draws")
        self.reference=reference
        self.sd=float(sd)
        self.rng=rng
        self.white=self._white(rng)
        self.initial_noise=self._packed_initial(self.white)
        outputs=[empirical(reference,self.queries,float(h))
                 +self.sd*self.initial_noise[:,j]
                 for j,h in enumerate(self.hs)]
        self.initial_values=np.stack(outputs,axis=1)
        return self.initial_values

    def _fill_terminal_scalar_cross(self,output,coeff,un,location,bindex):
        """Batch the original BS formulas when the public span fills the space."""
        aa,bb=self.ia,self.ib
        da=(aa==bb).astype(float)
        weights=np.where(da>0,1/np.sqrt(2),1.)
        for i in range(self.Q):
            # The componentwise subtraction and products follow the scalar path.
            v=(self.z[i]-un[location])[:,:self.r]
            prod=v[:,aa]*v[:,bb]
            for j,h in enumerate(self.hs):
                f,c1,c2=coeff[i,location,j,bindex].T
                a=i*self.H+j
                block=output[a*self.ncore:(a+1)*self.ncore]
                block[0]=f
                block[1:1+self.r]=((h*c1)[:,None]*v).T
                block[1+self.r:1+self.r+len(aa)]=(weights[None,:]*(
                    da[None,:]*f[:,None]+h*h*(
                        da[None,:]*c1[:,None]+c2[:,None]*prod))).T

    def _terminal_scalar_covariance(self,un,location,bindex):
        """Evaluate each symmetric location distance once, retaining all h ties."""
        left,right=np.triu_indices(len(un))
        pair_ids=np.empty((len(un),len(un)),dtype=np.intp)
        pair_ids[left,right]=np.arange(len(left))
        pair_ids[right,left]=pair_ids[left,right]
        distances=np.linalg.norm(un[left]-un[right],axis=-1)
        radial=self.coeffs(distances,0)[...,0]
        U=len(location)
        BB=np.empty((U,U))
        # Bounded indexing workspace; radial blocks retain the original h order.
        for start in range(0,U,256):
            end=min(U,start+256)
            pairs=pair_ids[location[start:end,None],location[None,:]]
            BB[start:end]=radial[pairs,bindex[start:end,None],bindex[None,:]]
        return BB

    def conditional_masses(self,candidates,*,debug=False):
        """One terminal scalar batch, including all H density scales, in shared field."""
        if getattr(self,"_terminal_done",False):raise RuntimeError("terminal history sealed")
        if not hasattr(self,"initial_noise"):raise RuntimeError("field not initialized")
        pos=np.asarray(candidates,float)
        if pos.ndim!=3 or pos.shape[0]!=self.Q or pos.shape[1]<1 or pos.shape[2]!=self.D or not np.isfinite(pos).all():
            raise ValueError("candidates must be [Q,C,D]")
        Q,C,D,H,r,q=self.Q,pos.shape[1],self.D,self.H,self.r,self.q
        rotated=pos.reshape(-1,D)@self.O
        # Distinct coordinates and functionals retain all cross-query ties.
        originals={x.tobytes():i for i,x in enumerate(self.queries)}
        unique=[];unique_map={}
        ids=[]
        for x in pos.reshape(-1,D):
            key=x.tobytes()
            if key not in unique_map:
                unique_map[key]=len(unique);unique.append(x)
            ids.append(unique_map[key])
        ids=np.asarray(ids).reshape(Q,C)
        un=np.asarray(unique,float).reshape(-1,D)@self.O
        missing=[];lookup={}
        grid=np.full((Q,C,H),-1,int)
        cached=np.full((Q,C,H),-1,int)
        for i in range(Q):
            for j in range(C):
                key=pos[i,j].tobytes()
                if key in originals:
                    cached[i,j,:]=originals[key]*H+np.arange(H)
                else:
                    for b in range(H):
                        label=(int(ids[i,j]),b)
                        if label not in lookup:
                            lookup[label]=len(missing)
                            missing.append(label)
                        grid[i,j,b]=lookup[label]
        U=len(missing)
        values=np.empty((Q,C,H))
        if U:
            memory_plan=plan_structured_memory(D,Q,H,span_rank=r,candidate_count=C,
                terminal_scalars=U,unique_locations=len(un),order=self._memory_order)
            self.plan["terminal_memory_plan"]=memory_plan.as_dict()
            self.plan["terminal_memory_check"]=require_memory(
                memory_plan.terminal_extra_bytes,
                memory_limit_gib=self.memory_limit_bytes/2**30,
                stage="structured conditional")
            bindex=np.array([b for loc,b in missing],int)
            location=np.array([loc for loc,b in missing],int)
            BS=np.empty((self.S,U),order="F")
            BV=np.empty((self.V,U),order="F") if q else None
            BT=np.empty((self.T,U),order="F") if self.T else None
            perp=np.empty((U,q)) if q else None
            coeff=self.coeffs(np.linalg.norm(self.z[:,None,:]-un[None,:,:],axis=-1),2)
            if not q:
                self._fill_terminal_scalar_cross(BS,coeff,un,location,bindex)
            else:
                for u,(li,b) in enumerate(missing):
                    diff=self.z-un[li]
                    dr=diff[:,:r]
                    ww=diff[:,r:]
                    if q:
                        perp[u]=ww[0]
                        if np.max(np.abs(ww-ww[0]))>1e-8:
                            raise ArithmeticError("original query span has significant transverse differences")
                    for a,(i,j) in enumerate(self.sites):
                        f,c1,c2=coeff[i,li,j,b]
                        h=self.hs[j]
                        v=dr[i]
                        vblock=np.empty(self.ncore)
                        vblock[0]=f
                        vblock[1:1+r]=h*c1*v
                        if len(self.ia):
                            aa,bb=self.ia,self.ib
                            da=(aa==bb).astype(float)
                            prod=v[aa]*v[bb]
                            vblock[1+r:1+r+len(aa)]=np.where(da>0,1/np.sqrt(2),1.)*(
                                da*f+h*h*(da*c1+c2*prod))
                        if q:vblock[-1]=np.sqrt(q/2)*(f+h*h*c1)+h*h*c2*np.dot(ww[i],ww[i])/np.sqrt(2*q)
                        BS[a*self.ncore:(a+1)*self.ncore,u]=vblock
                        if q:
                            BV[a*self.nvector:(a+1)*self.nvector,u]=np.r_[h*c1,h*h*c2*v]
                            if self.T:BT[a,u]=h*h*c2
            XS=solve_triangular(self.Ls,BS,lower=True,overwrite_b=True,check_finite=False)
            if not np.shares_memory(XS,BS):raise RuntimeError("terminal S solve was unexpectedly copied")
            removed=XS.T@XS
            mean=XS.T@self.white[0]
            del XS,BS
            if q:
                XV=solve_triangular(self.Lv,BV,lower=True,overwrite_b=True,check_finite=False)
                if not np.shares_memory(XV,BV):raise RuntimeError("terminal V solve was unexpectedly copied")
                dot=perp@perp.T
                removed+=(XV.T@XV)*dot
                mean+=np.sum((XV.T@self.white[1])*perp,axis=1)
                del XV,BV
                if self.T:
                    XT=solve_triangular(self.Lt,BT,lower=True,overwrite_b=True,check_finite=False)
                    if not np.shares_memory(XT,BT):raise RuntimeError("terminal T solve was unexpectedly copied")
                    norm=np.einsum("id,id->i",perp,perp)
                    removed+=(XT.T@XT)*.5*(dot*dot-np.outer(norm,norm)/q)
                    zd,zo=self.white[2],self.white[3]
                    quad=(zd@((perp**2-norm[:,None]/q).T))/np.sqrt(2)
                    quad+=zo@((perp[:,self.off_i]*perp[:,self.off_j]).T)
                    mean+=np.sum(XT.T*quad.T,axis=1)
                    del XT,BT,quad,norm
                del dot,perp
            del coeff
            BB=self._terminal_scalar_covariance(un,location,bindex)
            BB[np.diag_indices(U)]+=self.jitter
            schur=.5*(BB-removed+(BB-removed).T)
            eigmin=float(np.linalg.eigvalsh(schur)[0])
            repair=max(0.,-eigmin+2*self.jitter) if eigmin<=0 else 0.
            if repair>1e-7*max(1.,float(np.max(np.diag(BB)))):
                raise ArithmeticError("structured conditional covariance needs excessive repair")
            root=cholesky(schur+repair*np.eye(U),lower=True,check_finite=False)
            if debug:
                self._debug_conditional=dict(
                    mean=mean.copy(),covariance=schur.copy(),
                    locations=np.array([unique[li] for li,b in missing]),
                    bandwidths=np.array([self.hs[b] for li,b in missing]),
                    conditional_extra_jitter=repair)
            new=mean+root@self.rng.standard_normal(U)
        else:
            repair=0.
            new=np.empty(0)
        for i in range(Q):
            for j in range(C):
                for k in range(H):
                    if cached[i,j,k]>=0:
                        refidx=cached[i,j,k]
                        values[i,j,k]=self.initial_noise[refidx//H,refidx%H,0]
                    else:
                        values[i,j,k]=new[grid[i,j,k]]
        self._terminal_done=True
        return values,dict(mode=("structured_shared_joint"
                                 if self.bandwidth_correlation=="shared"
                                 else "structured_independent_per_bandwidth"),
            initial_span_rank=self.r,
            terminal_new_scalars=U,conditional_extra_jitter=repair,
            retained_cross_query_covariances=True,
            retained_cross_bandwidth_covariances=(self.bandwidth_correlation=="shared"),
            bandwidth_correlation=self.bandwidth_correlation,
            numerical_ideal_mechanism_not_machine_certified=True)

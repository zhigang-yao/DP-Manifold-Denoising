"""Public synthetic geometry and independent private-reference replicates.

Query targets and tangent vectors are used ONLY here and in the evaluator.
They are never passed to the proposed estimator. Geometry choices are
versioned; a manuscript claim requires matching this public design hash.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from geometry.noise import sample_uniform_ball_noise

GEOMETRIES=("trefoil","torus","swiss_roll","sphere3","circle","double_graph","linked_tori","bowl4","paper_swiss_roll","swiss_roll_candidate")
DIMENSIONS={"trefoil":(1,3),"torus":(2,3),"swiss_roll":(2,3),"sphere3":(3,4),"circle":(1,2),"double_graph":(3,5),"linked_tori":(2,3),"bowl4":(3,4),"paper_swiss_roll":(2,3),"swiss_roll_candidate":(2,3)}
DOUBLE_GRAPH_ALPHA = .3  # paper value (2026-11-09); the 2040 freeze had .12
DOUBLE_GRAPH_VERSION = "double_graph_alpha0.3_volume_interior08_fullambient_v2"
# On the unit 3-ball the two graph gradients have norms <=2*alpha and
# <=alpha. Their Gram determinant is maximized at u=(0,+/-1,0).
DOUBLE_GRAPH_VOLUME_BOUND = np.sqrt((1+4*DOUBLE_GRAPH_ALPHA**2)*(1+DOUBLE_GRAPH_ALPHA**2))

@dataclass(frozen=True)
class SyntheticConfig:
    geometry: str = "trefoil"
    n: int = 2000
    Q: int = 100
    D: int | None = None
    sigma: float = 0.04
    a: float = 0.08
    replicate: int = 0
    seed: int = 20261105

def _unit(y):
    return y/np.linalg.norm(y,axis=-1,keepdims=True)

def _rng(seed, geometry, replicate, role):
    g=GEOMETRIES.index(geometry)
    return np.random.default_rng(np.random.SeedSequence([seed,g,replicate,role]))

def _graph_volume_weight(u):
    """Exact Hausdorff surface-volume Jacobian for the 3D double-curvature graph."""
    x,y,z=np.asarray(u,float).T
    alpha=DOUBLE_GRAPH_ALPHA
    a=np.stack([2*alpha*x,-2*alpha*y,np.zeros(len(u))],axis=1)
    b=np.stack([np.zeros(len(u)),alpha*z,alpha*y],axis=1)
    aa=np.sum(a*a,axis=1)
    bb=np.sum(b*b,axis=1)
    ab=np.sum(a*b,axis=1)
    return np.sqrt((1+aa)*(1+bb)-ab**2)

def _sample_double_graph(count,rng,*,interior=False):
    accepted=[]
    remaining=count
    while remaining:
        k=max(64,int(remaining*1.25)+50)
        normal=_unit(rng.standard_normal((k,3)))
        u=normal*rng.random(k)[:,None]**(1./3.)
        keep=rng.random(k)<_graph_volume_weight(u)/DOUBLE_GRAPH_VOLUME_BOUND
        if interior:
            keep&=np.sum(u*u,axis=1)<.8**2
        selected=u[keep][:remaining]
        accepted.append(selected)
        remaining-=len(selected)
    u=np.concatenate(accepted,axis=0)
    x,y,z=u.T
    return np.column_stack([u,DOUBLE_GRAPH_ALPHA*(x*x-y*y),DOUBLE_GRAPH_ALPHA*y*z]),u


def _native(geometry,count,rng,*,interior=False):
    if geometry=="swiss_roll_candidate":
        from geometry.candidate_swiss import sample_native
        return sample_native(count,rng,interior=interior)
    if geometry=="bowl4":
        from geometry.paper_geometries import sample_bowl4
        return sample_bowl4(count,rng,interior=interior)
    if geometry=="linked_tori":
        from geometry.linked_tori import sample
        return sample(count,rng,balanced=interior)
    if geometry=="double_graph":
        points,u=_sample_double_graph(count,rng,interior=interior)
        x,y,z=u.T
        alpha=DOUBLE_GRAPH_ALPHA
        T=np.zeros((count,3,5))
        T[:,0,:]=np.column_stack([np.ones(count),np.zeros(count),np.zeros(count),2*alpha*x,np.zeros(count)])
        T[:,1,:]=np.column_stack([np.zeros(count),np.ones(count),np.zeros(count),-2*alpha*y,alpha*z])
        T[:,2,:]=np.column_stack([np.zeros(count),np.zeros(count),np.ones(count),np.zeros(count),alpha*y])
        basis=np.linalg.qr(np.swapaxes(T,1,2))[0]
        return points,np.swapaxes(basis,1,2)
    if geometry=="circle":
        t=rng.uniform(0,2*np.pi,count)
        p=np.stack([np.cos(t),np.sin(t)],axis=1)
        tangent=np.stack([-np.sin(t),np.cos(t)],axis=1)[:,None,:]
        return p,tangent
    if geometry=="sphere3":
        z=_unit(rng.standard_normal((count,4)))
        return z,None
    if geometry=="trefoil":
        # Density uniform in arc length; rejection bound exceeds max speed.
        ts=[]
        while sum(len(a) for a in ts)<count:
            t=rng.uniform(0,2*np.pi,max(512,count))
            v=np.stack([np.cos(t)+4*np.cos(2*t),-np.sin(t)+4*np.sin(2*t),-3*np.cos(3*t)],axis=-1)/3
            keep=rng.uniform(size=len(t))<np.linalg.norm(v,axis=1)/2.5
            ts.append(t[keep])
        t=np.concatenate(ts)[:count]
        p=np.stack([np.sin(t)+2*np.sin(2*t),np.cos(t)-2*np.cos(2*t),-np.sin(3*t)],axis=1)/3
        v=np.stack([np.cos(t)+4*np.cos(2*t),-np.sin(t)+4*np.sin(2*t),-3*np.cos(3*t)],axis=1)/3
        return p,_unit(v)[:,None,:]
    if geometry=="torus":
        # Surface area measure: minor angle phi has density proportional to R+r cos(phi).
        R,r=2/3,1/3
        phi=[]
        while sum(len(a) for a in phi)<count:
            u=rng.uniform(0,2*np.pi,max(512,count))
            phi.append(u[rng.random(len(u))<(R+r*np.cos(u))/(R+r)])
        v=np.concatenate(phi)[:count]
        t=rng.uniform(0,2*np.pi,count)
        rad=R+r*np.cos(v)
        p=np.stack([rad*np.cos(t),rad*np.sin(t),r*np.sin(v)],axis=1)
        t1=np.stack([-np.sin(t),np.cos(t),np.zeros(count)],axis=1)
        t2=np.stack([-np.sin(v)*np.cos(t),-np.sin(v)*np.sin(t),np.cos(v)],axis=1)
        return p,np.stack([t1,t2],axis=1)
    if geometry=="swiss_roll":
        # One-turn smooth sheet. Interior queries exclude fixed boundary strips.
        # This is an explicit simulation geometry; keep distinct from archived S-curve.
        theta=[]
        limit=2*np.pi
        lo,hi=(0.3,limit-0.3) if interior else (0.,limit)
        def speed(t):
            r=.25+.14*t
            return np.sqrt(r*r+.14**2)
        max_speed=speed(hi)
        while sum(len(a) for a in theta)<count:
            u=rng.uniform(lo,hi,max(512,count))
            theta.append(u[rng.random(len(u))<speed(u)/max_speed])
        t=np.concatenate(theta)[:count]
        y=rng.uniform(-.38,.38,count) if interior else rng.uniform(-.5,.5,count)
        r=.25+.14*t
        p=np.stack([r*np.cos(t),r*np.sin(t),y],axis=1)
        t1=_unit(np.stack([.14*np.cos(t)-r*np.sin(t),.14*np.sin(t)+r*np.cos(t),np.zeros(count)],axis=1))
        t2=np.tile([0.,0.,1.],(count,1))
        return p,np.stack([t1,t2],axis=1)
    raise ValueError("unknown geometry")

def generate(c: SyntheticConfig):
    if c.geometry not in GEOMETRIES or c.n<2 or c.Q<1 or c.sigma<0 or c.a<0:
        raise ValueError("invalid geometry/data settings")
    if c.geometry=="linked_tori":
        from geometry.linked_tori import REACH
        if not np.isfinite(c.a+c.sigma) or c.a+c.sigma>=REACH:
            raise ValueError(f"query offset plus noise radius must stay below linked-tori reach {REACH:g}")
    d,native_D=DIMENSIONS[c.geometry]
    D=native_D if c.D is None else int(c.D)
    if D<native_D:
        raise ValueError("ambient dimension must be >=native embedding dimension")
    if c.geometry=="paper_swiss_roll":
        from geometry.paper_geometries import generate_paper_swiss
        return generate_paper_swiss(c,D,_rng(c.seed,c.geometry,c.replicate,1),
                                    _rng(c.seed,c.geometry,c.replicate,2))
    emb=(np.eye(D,native_D) if D==native_D or c.geometry in ("double_graph","linked_tori","bowl4","swiss_roll_candidate")
         else np.linalg.qr(_rng(c.seed,c.geometry,0,7).standard_normal((D,native_D)))[0])
    clean,tangents=_native(c.geometry,c.n,_rng(c.seed,c.geometry,c.replicate,1))
    latent,T=_native(c.geometry,c.Q,_rng(c.seed,c.geometry,0,3),interior=True)
    ref=clean@emb.T
    latent_ambient=latent@emb.T
    g=_rng(c.seed,c.geometry,0,4).standard_normal((c.Q,D))
    if c.geometry=="sphere3":
        g_native=g@emb
        tg_native=g_native-(g_native*latent).sum(axis=1)[:,None]*latent
        normals=g-tg_native@emb.T
    else:
        t_ambient=T@emb.T
        normals=g-np.einsum("qkd,qk->qd",t_ambient,
                             np.einsum("qkd,qd->qk",t_ambient,g))
    normals=_unit(normals)
    query=latent_ambient+c.a*normals
    noise=sample_uniform_ball_noise(c.n,D,c.sigma,_rng(c.seed,c.geometry,c.replicate,2))
    extra={}
    if c.geometry=="double_graph":
        extra=dict(geometry_metadata=dict(
            geometry="double_graph",version=DOUBLE_GRAPH_VERSION,
            intrinsic_dimension=3,native_dimension=5,alpha=DOUBLE_GRAPH_ALPHA,
            reference_base_radius=1.,query_base_radius=.8,
            sampling_measure="uniform_surface_volume",
            query_normal_space="full_ambient",normalization="none"))
    if c.geometry=="linked_tori":
        from geometry.linked_tori import component_labels,metadata
        extra=dict(component=component_labels(latent),
                   reference_component=component_labels(clean),geometry_metadata=metadata())
    if c.geometry=="swiss_roll_candidate":
        from geometry.candidate_swiss import metadata
        extra=dict(geometry_metadata=metadata())
    return dict(reference=ref+noise,queries=query,latent=latent_ambient,
                d=d,D=D,clean_reference=ref,
                config=c,noise_model="uniform_ball",known_normal_distance=c.a,**extra)

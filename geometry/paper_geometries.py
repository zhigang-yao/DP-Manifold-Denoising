"""Reuse audited source geometry without inheriting old estimator policies."""
import numpy as np
from . import manifolds
from geometry.queries import make_query_bank
from geometry.noise import sample_uniform_ball_noise


def sample_bowl4(count, rng, *, interior=False):
    """F(u)=(u,.4||u||²), u in B3(1); exact graph-volume rejection.

The new multiquery interior cutoff .8 mirrors double_graph and is a
reviewable proposed query protocol, not a historical fixed-Q1 reproduction.
"""
    pieces=[]
    left=count
    while left:
        size=max(64,2*left)
        direction=rng.standard_normal((size,3))
        direction/=np.linalg.norm(direction,axis=1)[:,None]
        radius=rng.random(size)**(1/3)
        accept=rng.random(size)<np.sqrt(1+.64*radius*radius)/np.sqrt(1.64)
        if interior:
            accept&=radius<.8
        selected=(direction*radius[:,None])[accept][:left]
        pieces.append(selected)
        left-=len(selected)
    u=np.concatenate(pieces)
    points=np.column_stack([u,.4*np.sum(u*u,axis=1)])
    derivative=np.broadcast_to(np.eye(3),(count,3,3)).copy()
    derivative=np.concatenate([derivative,(.8*u)[:,:,None]],axis=2)
    tangent=np.linalg.qr(np.swapaxes(derivative,1,2))[0]
    return points,np.swapaxes(tangent,1,2)


def generate_paper_swiss(c, D, reference_rng, noise_rng):
    """Original volume-centered diameter-two Swiss roll and audited queries.

Uses the existing manuscript manifold/projection implementation. D>3 uses
its deterministic isometry and the complete ambient normal space. Seeds
are explicit in SyntheticConfig rather than pretending to reuse old runs.
"""
    meta=manifolds.geometry_metadata("swiss_roll")
    embedding=manifolds._embedding("swiss_roll",D)
    raw=manifolds._sample_raw("swiss_roll",c.n,reference_rng)
    clean=((raw-np.asarray(meta["raw_center"]))*meta["scale"])@embedding.T
    bank=make_query_bank("swiss_roll",c.Q,c.seed*1000+c.replicate if c.replicate else c.seed,[c.a],ambient_dimension=D)
    noise=sample_uniform_ball_noise(c.n,D,c.sigma,noise_rng)
    return dict(reference=clean+noise,clean_reference=clean,
                queries=bank["points"][c.a],latent=bank["base"],
                boundary_distance=bank["base_boundary_distance"],
                geometry_metadata=meta,d=2,D=D,config=c,
                noise_model="uniform_ball",known_normal_distance=c.a)

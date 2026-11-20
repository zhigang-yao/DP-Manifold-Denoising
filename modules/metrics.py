"""Evaluation only: never pass latent targets into algorithm implementations."""
import numpy as np

def latent_rmse(prediction,latent):
    p=np.asarray(prediction,float)
    y=np.asarray(latent,float)
    if p.shape!=y.shape or not np.isfinite(p).all():
        raise ValueError("invalid prediction shape or values")
    return float(np.sqrt(np.mean(np.sum((p-y)**2,axis=1))))

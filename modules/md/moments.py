"""Gaussian empirical moments and natural-coordinate unpacking."""

import numpy as np


def empirical(reference, x, h, kind="moments"):
    d = x.shape[1]
    out = []
    for z in x:
        u = (reference - z) / h
        weight = np.exp(-0.5 * np.sum(u * u, axis=1))
        q = weight.mean()
        if kind == "density":
            out.append([q])
            continue
        v = (weight[:, None] * u).mean(axis=0)
        if kind == "density_gradient":
            out.append(np.r_[q, v / h])
            continue
        B = (u.T * weight) @ u / len(u)
        out.append(
            np.r_[
                q,
                v,
                [
                    B[i, j] / (np.sqrt(2) if i == j else 1.0)
                    for j in range(d)
                    for i in range(j + 1)
                ],
            ]
        )
    return np.array(out)


def unpack(t, d):
    q = t[..., 0]
    v = t[..., 1 : 1 + d]
    B = np.empty(t.shape[:-1] + (d, d))
    k = 1 + d
    for j in range(d):
        for i in range(j + 1):
            B[..., i, j] = B[..., j, i] = t[..., k] * (np.sqrt(2) if i == j else 1.0)
            k += 1
    return q, v, B


def empirical_density_grid(reference,sites,bandwidths,*,chunk_size=32):
    """Evaluate the original Gaussian mass at all h with reused distances.

    No kernel approximation is introduced. scipy's direct squared-Euclidean
    distance loop avoids an [sites,n,D] temporary, and a reference-sized row
    still uses the original arithmetic mean. Only float64 operation order may
    differ from repeated (reference-z)/h evaluations.
    """
    from scipy.spatial.distance import cdist
    reference=np.asarray(reference,dtype=float);sites=np.asarray(sites,dtype=float)
    hs=np.asarray(bandwidths,dtype=float)
    if (reference.ndim!=2 or len(reference)<1 or sites.ndim!=2
            or sites.shape[1]!=reference.shape[1] or hs.ndim!=1 or len(hs)<1
            or not all(np.isfinite(a).all() for a in (reference,sites,hs))
            or np.any(hs<=0) or not isinstance(chunk_size,int) or chunk_size<1):
        raise ValueError('finite references/sites and positive bandwidths required')
    out=np.empty((len(sites),len(hs)),dtype=float)
    for lo in range(0,len(sites),chunk_size):
        hi=min(len(sites),lo+chunk_size)
        distance2=cdist(sites[lo:hi],reference,metric='sqeuclidean')
        for k,h in enumerate(hs):
            out[lo:hi,k]=np.exp((-.5/(h*h))*distance2).mean(axis=1)
    return out

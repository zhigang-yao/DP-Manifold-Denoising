"""Two equal disjoint torus surfaces in a Hopf-link arrangement.

Approved fixed geometry: R=.8, r=.2, centers (+/-.4,0,0), core
planes xy/xz. This is d=2 on BOTH components, unlike legacy torus_ring.
Coordinates are deliberately not normalized to a unit bounding ball.
"""
import numpy as np

MAJOR_RADIUS = .8
MINOR_RADIUS = .2
CENTER_OFFSET = MAJOR_RADIUS / 2
SURFACE_GAP = MAJOR_RADIUS - 2*MINOR_RADIUS
# The orthogonal core circles are exactly R apart. For R>2r, the two
# surface tubes have gap R-2r; the union reach also accounts for curvature.
REACH = min(MINOR_RADIUS,MAJOR_RADIUS-MINOR_RADIUS,SURFACE_GAP/2)


def metadata():
    return dict(geometry="linked_tori", version="linked_tori_R08_r02_balanced_queries_v3",
                intrinsic_dimension=2, native_dimension=3,
                major_radius=MAJOR_RADIUS, minor_radius=MINOR_RADIUS,
                centers=[[-CENTER_OFFSET, 0., 0.], [CENTER_OFFSET, 0., 0.]],
                core_planes=["xy", "xz"], minimum_surface_gap=SURFACE_GAP,
                reach=REACH, maximum_norm=CENTER_OFFSET+MAJOR_RADIUS+MINOR_RADIUS,
                sampling_measure="uniform_surface_area_equal_component_mixture",
                query_component_allocation="balanced_seeded")


def sample(count, rng, *, balanced=False):
    """IID normalized surface-area measure; return points and tangent frames."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    pieces=[]
    left=count
    while left:
        phi=rng.uniform(0,2*np.pi,max(64,2*left))
        accepted=phi[rng.random(len(phi)) < (MAJOR_RADIUS+MINOR_RADIUS*np.cos(phi))/(MAJOR_RADIUS+MINOR_RADIUS)][:left]
        pieces.append(accepted)
        left-=len(accepted)
    phi=np.concatenate(pieces)
    theta=rng.uniform(0,2*np.pi,count)
    radius=MAJOR_RADIUS+MINOR_RADIUS*np.cos(phi)
    points=np.column_stack([radius*np.cos(theta),radius*np.sin(theta),MINOR_RADIUS*np.sin(phi)])
    t1=np.column_stack([-np.sin(theta),np.cos(theta),np.zeros(count)])
    t2=np.column_stack([-np.sin(phi)*np.cos(theta),-np.sin(phi)*np.sin(theta),np.cos(phi)])
    frame=np.stack([t1,t2],axis=1)
    if balanced:
        extra=int(rng.integers(0,2)) if count%2 else 0
        first=count//2+int(count%2 and extra==0)
        component=np.r_[np.zeros(first,dtype=int),np.ones(count-first,dtype=int)]
        rng.shuffle(component)
    else:
        component=rng.integers(0,2,count)
    second=component==1
    points[second]=points[second][:,[0,2,1]]
    frame[second]=frame[second][:,:,[0,2,1]]
    points[:,0]+=np.where(second,CENTER_OFFSET,-CENTER_OFFSET)
    return points,frame


def _component_projection(x, component):
    p=x[:,:3].copy()
    p[:,0]-=(-CENTER_OFFSET if component==0 else CENTER_OFFSET)
    if component==1:
        p=p[:,[0,2,1]]
    rho=np.hypot(p[:,0],p[:,1])
    direction=np.zeros((len(p),2))
    np.divide(p[:,:2],rho[:,None],out=direction,where=rho[:,None]>0)
    direction[rho==0,0]=1.
    tube=np.column_stack([rho-MAJOR_RADIUS,p[:,2]])
    length=np.linalg.norm(tube,axis=1)
    unit=np.zeros_like(tube)
    np.divide(tube,length[:,None],out=unit,where=length[:,None]>0)
    unit[length==0,0]=1.
    native=np.column_stack([(MAJOR_RADIUS+MINOR_RADIUS*unit[:,0])[:,None]*direction,MINOR_RADIUS*unit[:,1]])
    if component==1:
        native=native[:,[0,2,1]]
    native[:,0]+=(-CENTER_OFFSET if component==0 else CENTER_OFFSET)
    out=np.zeros_like(x)
    out[:,:3]=native
    return out


def component_labels(points):
    """Nearest component, deterministically choosing A for an exact tie."""
    x=np.asarray(points,float)
    if x.ndim!=2 or x.shape[1]<3 or not np.isfinite(x).all():
        raise ValueError("finite points with ambient dimension >=3 required")
    a,b=_component_projection(x,0),_component_projection(x,1)
    return (np.sum((x-b)**2,axis=1)<np.sum((x-a)**2,axis=1)).astype(int)


def project(points):
    """Exact nearest point on the union, including zero-padded ambient axes."""
    x=np.asarray(points,float)
    labels=component_labels(x)
    return np.where(labels[:,None]==0,_component_projection(x,0),_component_projection(x,1))

"""Resource plans for the exact, fully resident structured Gaussian field.

Plans are conservative working-set estimates, not hard allocator guarantees.
They never change a release mechanism or lower its numerical precision.
"""
from dataclasses import dataclass, asdict
from numbers import Integral, Real
from pathlib import Path
import math
import os
import re
import psutil
from threadpoolctl import threadpool_info

GIB=2**30
CAMPAIGN_MAX_BUDGET_GIB=16.
CAMPAIGN_MAX_QUERIES=100
RESERVE_BYTES=512*2**20
PLAN_VERSION='structured_inplace_v1'


def _integer(value,name,minimum=0,maximum=None):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum or (maximum is not None and value>maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum if maximum is not None else "infinity"}]')
    return int(value)


def validate_budget(memory_limit_gib,*,maximum_gib=None):
    if (isinstance(memory_limit_gib,bool) or not isinstance(memory_limit_gib,Real)
            or not math.isfinite(memory_limit_gib) or memory_limit_gib<=0
            or (maximum_gib is not None and memory_limit_gib>maximum_gib)):
        raise ValueError('memory_limit_gib must be finite and positive'
                         +(f' and <={maximum_gib:g} GiB for this campaign' if maximum_gib is not None else ''))
    return int(memory_limit_gib*GIB)


@dataclass(frozen=True)
class StructuredMemoryPlan:
    D:int
    Q:int
    H:int
    span_rank:int
    perp_dim:int
    S:int
    V:int
    T:int
    terminal_scalars:int
    terminal_locations:int
    factor_bytes:int
    initial_extra_bytes:int
    terminal_extra_bytes:int
    reserve_bytes:int
    version:str=PLAN_VERSION

    @property
    def initial_peak_bytes(self):return self.factor_bytes+self.initial_extra_bytes

    @property
    def terminal_peak_bytes(self):return self.factor_bytes+self.terminal_extra_bytes

    def as_dict(self):
        return dict(asdict(self),initial_peak_bytes=self.initial_peak_bytes,
                    terminal_peak_bytes=self.terminal_peak_bytes)


def plan_structured_memory(D,Q,H,*,span_rank=None,candidate_count=None,
                           terminal_scalars=None,unique_locations=None,order=128):
    """Plan initial factors and the complete Hard+Soft union by default.

    Runtime supplies the certified rank and exact uncached scalar/location
    counts. Cached locations still count in radial workspaces when U>0.
    The ten U-square matrices cover simultaneous Schur/eigen/debug work.
    """
    D=_integer(D,'D',2);Q=_integer(Q,'Q',1);H=_integer(H,'H',1)
    order=_integer(order,'order',8)
    r=min(D,Q-1) if span_rank is None else _integer(span_rank,'span_rank',0,min(D,Q-1))
    q=D-r
    C=2*H if candidate_count is None else _integer(candidate_count,'candidate_count',1)
    U=Q*C*H if terminal_scalars is None else _integer(terminal_scalars,'terminal_scalars',0,Q*C*H)
    L=Q*C if unique_locations is None else _integer(unique_locations,'unique_locations',0,Q*C)
    if U>L*H:raise ValueError('terminal scalar count exceeds the location/bandwidth grid')
    core=(r+1)*(r+2)//2+int(q>0)
    S=Q*H*core;V=Q*H*(r+1) if q else 0;T=Q*H if q>=2 else 0
    factors=8*(S*S+V*V+T*T)
    upper=max(16.,math.sqrt(D)+14.) if D>=5 else 16.
    segments=10+(1+max(0,math.ceil((upper-24.)/8.)) if upper>16. else 0)
    nodes=order*segments
    # Coefficients, distance temporaries, local pair blocks, radial weights,
    # special-function scratch, plus allocator/BLAS/baseline reserve.
    initial_extra=8*(5*Q*Q*H*H+2*D*Q*Q+20*core*core
                    +10*H*H*nodes+4*min(256,Q*Q)*nodes)+RESERVE_BYTES
    if U:
        radial=8*((3*Q*L+L*L)*H*H+2*D*max(Q*L,L*L)
                  +4*min(256,max(Q*L,L*L))*nodes)
        tensor_scratch=8*(3*U*(q*(q-1)//2)+3*U*q+2*T*U) if q>=2 else 0
        saved_state=8*(3*D*D+3*Q*Q+6*Q*H*D*D+4*Q*H*((D+1)*(D+2)//2))
        terminal_extra=8*U*(S+V+T)+80*U*U+radial+tensor_scratch+saved_state+RESERVE_BYTES
    else:
        terminal_extra=8*Q*C*H+RESERVE_BYTES
    return StructuredMemoryPlan(D,Q,H,r,q,S,V,T,U,L,factors,initial_extra,
                                terminal_extra,RESERVE_BYTES)


def geometry_workspace(D,Q):
    """Full SVD/orthogonal frame and triangular-index allocation before rank is known."""
    D=_integer(D,'D',2);Q=_integer(Q,'Q',1)
    return 8*(5*D*D+3*Q*Q+5*Q*D)+RESERVE_BYTES


def initial_draw_workspace(D,Q,H,n_reference):
    """Packed noise rotation and the per-query empirical-moment workspace."""
    D=_integer(D,'D',2);Q=_integer(Q,'Q',1);H=_integer(H,'H',1)
    n=_integer(n_reference,'n_reference',1);p=(D+1)*(D+2)//2
    return 8*(4*Q*H*D*D+6*Q*H*p+3*n*D+3*D*D)+RESERVE_BYTES


def decoder_workspace(D,Q,inner_B,auxiliary_chunk_size):
    """Conservative peak for shared Hard/Soft marginal Monte Carlo decoding."""
    D=_integer(D,'D',2);Q=_integer(Q,'Q',1)
    B=_integer(inner_B,'inner_B',1);chunk=_integer(auxiliary_chunk_size,'auxiliary_chunk_size',1)
    count=max(Q,chunk);batch=min(16,count);p=(D+1)*(D+2)//2
    return 8*(8*batch*B*D*D+4*batch*B*p+12*count*D*D+8*count*p)+RESERVE_BYTES


@dataclass(frozen=True)
class MemorySnapshot:
    rss_bytes:int
    available_bytes:int


def cgroup_available_bytes(directory,root):
    """Minimum visible v1/v2 ancestor headroom; None if no finite limit exists."""
    directory,root=Path(directory),Path(root)
    if directory!=root and root not in directory.parents:return None
    limits=[]
    current=directory
    while True:
        for limit_name,usage_name in [('memory.max','memory.current'),
                                      ('memory.limit_in_bytes','memory.usage_in_bytes')]:
            try:
                raw=(current/limit_name).read_text().strip()
                if raw=='max':continue
                limit=int(raw);used=int((current/usage_name).read_text().strip())
                if limit>=0 and used>=0:limits.append(max(0,limit-used))
            except (OSError,ValueError):continue
        if current==root:break
        current=current.parent
    return min(limits) if limits else None


def resolve_cgroup_paths(cgroup_text,mountinfo_text):
    """Map kernel membership paths through visible v1/v2 mount roots."""
    memberships=[]
    for line in cgroup_text.splitlines():
        parts=line.split(':',2)
        if len(parts)==3:
            _,controllers,path=parts
            if controllers=='' or 'memory' in controllers.split(','):
                memberships.append((controllers,Path(path)))
    def unescape(value):
        return re.sub(r"\\([0-7]{3})",lambda m:chr(int(m.group(1),8)),value)
    result=[]
    for line in mountinfo_text.splitlines():
        try:
            before,after=line.split(' - ',1)
            fields=before.split();kind,source,options=after.split()[:3]
            mount_root=Path(unescape(fields[3]));point=Path(unescape(fields[4]))
        except (ValueError,IndexError):continue
        for controllers,membership in memberships:
            matches=(kind=='cgroup2' and controllers=='') or (
                kind=='cgroup' and 'memory' in controllers.split(',')
                and 'memory' in (options+','+source).split(','))
            if not matches:continue
            # At minimum honor every visible ancestor mount limit.
            result.append((point,point))
            try:relative=membership.relative_to(mount_root)
            except ValueError:continue
            result.append((point/relative,point))
    return list(dict.fromkeys(result))


def memory_snapshot():
    available=int(psutil.virtual_memory().available)
    roots=[(Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup')),
           (Path('/sys/fs/cgroup/memory'),Path('/sys/fs/cgroup/memory'))]
    try:
        roots+=resolve_cgroup_paths(Path('/proc/self/cgroup').read_text(),
                                    Path('/proc/self/mountinfo').read_text())
    except OSError:pass
    for directory,root in roots:
        cgroup=cgroup_available_bytes(directory,root)
        if cgroup is not None:available=min(available,cgroup)
    return MemorySnapshot(int(psutil.Process().memory_info().rss),available)


def require_memory(additional_bytes,*,memory_limit_gib,stage,snapshot=None):
    """Refuse before allocation using current RSS and actual visible headroom.

    additional_bytes includes the planner's reserve. A further 256 MiB remains
    available for unrelated activity. Concurrent allocations remain a race;
    use a single worker and external process limits for production campaigns.
    """
    additional_bytes=_integer(additional_bytes,'additional_bytes')
    limit=validate_budget(memory_limit_gib)
    snap=memory_snapshot() if snapshot is None else snapshot
    estimated=snap.rss_bytes+additional_bytes
    if estimated>limit:
        raise MemoryError(f'{stage}: estimated process budget {estimated/GIB:.2f} GiB exceeds {limit/GIB:.2f} GiB')
    if additional_bytes+256*2**20>snap.available_bytes:
        raise MemoryError(f'{stage}: need {additional_bytes/GIB:.2f} GiB plus headroom; available RAM is {snap.available_bytes/GIB:.2f} GiB')
    return dict(stage=stage,current_rss_bytes=snap.rss_bytes,
                additional_bytes=additional_bytes,estimated_process_bytes=estimated,
                available_bytes=snap.available_bytes,budget_bytes=limit)


def linalg_runtime_advice(largest_dimension,*,pools=None):
    """Report the reproduced large-matrix fault; never mutate an active BLAS."""
    pools=threadpool_info() if pools is None else pools
    affected=[dict(p) for p in pools if p.get('internal_api')=='openblas'
              and p.get('architecture')=='SkylakeX'
              and (p.get('version')=='0.3.30' or str(p.get('version','')).startswith('0.3.31.dev'))]
    blocked=bool(affected and largest_dimension>=16384)
    return dict(blocked=blocked,largest_dimension=int(largest_dimension),
        pools=pools,environment_coretype=os.environ.get('OPENBLAS_CORETYPE'),
        advice=('A large-matrix SIGSEGV was reproduced on these OpenBLAS SkylakeX builds. '
                'Before importing NumPy/SciPy, start a fresh process with '
                'OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2. '
                'This is a tested workaround, not a universal BLAS certification.'
                if blocked else 'No known affected large-matrix kernel detected; this is not a benchmark or allocation guarantee.'))


def require_linalg_runtime(largest_dimension):
    report=linalg_runtime_advice(largest_dimension)
    if report['blocked']:raise RuntimeError(report['advice'])
    return report

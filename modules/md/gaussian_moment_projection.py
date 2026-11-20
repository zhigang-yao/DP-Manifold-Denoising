"""Conic projection of *already released* Gaussian raw moments.

The public interface receives only noisy q/v/B and the public noise scale.
The two sets are J=[[q,v.T],[v,B]] >= 0, 0<=q<=1, with the Gaussian arm also
requiring trace(B)<=-2*q*log(q) (closed at q=0). Both use
``dq**2 + ||dv||**2 + .5*||dB||_F**2``. No physical h factors enter this metric.

Clarabel solves in centered natural-coordinate increments in units of sd.
Its cone matrices are fixed for a dimension; only the right-hand side changes.
An accepted output has independent residual checks and a recomputed numerical
primal/dual gap <=5e-5 for objective .5*||increment||**2. These are floating
point numerical certificates, not formal certificates of an exact projection.
Failed outputs retain the original moments, so a caller can decode baseline.
"""
from __future__ import annotations

import time
from typing import Any

import clarabel
import numpy as np
from scipy import sparse

# Fixed public numerical policy, independent of private/truth inputs.
_GAP_TARGET = 5e-5  # Public coordinate target .01 sd: sqrt(2*gap) <= .01.
_FEAS_TARGET = 1e-7  # Each physical violation divided by public sd.
_EPS = np.finfo(float).eps
_SQRT2 = np.sqrt(2.)
_ATTEMPTS = (
    {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-9,
     "max_iter": 180},
    # PSD barriers can stall with Clarabel's default .99 step fraction when
    # inactive box slacks are large in sd units. Tighter stopping tolerances
    # repeat that trajectory. Retry with a fixed conservative step instead;
    # the independent objective/gap/feasibility acceptance is unchanged.
    {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-9,
     "max_iter": 350, "max_step_fraction": .8},
)


def _finite_number(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _svec(matrix: np.ndarray) -> np.ndarray:
    """Clarabel upper-triangle column order; sqrt(2) on off diagonals."""
    size = len(matrix)
    return np.array([matrix[i,j]*(1. if i==j else _SQRT2)
                     for j in range(size) for i in range(j+1)])


def _unsvec(vector: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size,size))
    k = 0
    for j in range(size):
        for i in range(j+1):
            matrix[i,j] = matrix[j,i] = vector[k]/(1. if i==j else _SQRT2)
            k += 1
    return matrix


def _entropy_twice(q: float) -> float:
    # Do not evaluate log(0), and do not introduce a positive mass floor.
    if q == 0. or q == 1.:
        return 0.
    if not 0. < q < 1.:
        return float("nan")
    return -2.*q*np.log(q)


def _joint(q: float, v: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.block([[np.array([[q]]),v[None,:]],[v[:,None],B]])


class GaussianMomentProjector:
    """Reusable small conic problems for one public ambient dimension.

    Instances cache mutable solver workspaces and are not thread safe. Use one
    instance per worker. ``project_pair`` and ``project_batch`` do not mutate
    their inputs. ``sd`` must be finite and positive; true NP moments can be
    checked with the public DP scale used by their paired experiment.
    """

    def __init__(self, D: int):
        if not isinstance(D, (int,np.integer)) or isinstance(D,bool) or D < 1:
            raise ValueError("D must be a positive integer")
        self.D = int(D)
        self._B_indices = [(i,j) for j in range(D) for i in range(j+1)]
        self._width = 1+D+len(self._B_indices)
        self._psd_size = (D+1)*(D+2)//2
        self._P = sparse.eye(self._width,format="csc")
        self._zero = np.zeros(self._width)
        maps = []
        for k in range(self._width):
            unit = np.zeros(self._width)
            unit[k] = 1.
            q,v,B = self._unpack(unit)
            maps.append(_svec(_joint(q,v,B)))
        psd_map = np.column_stack(maps)
        box = np.zeros((2,self._width))
        box[:,0] = [-1.,1.]
        entropy = np.zeros((3,self._width))
        entropy[1,0] = -1.
        for k,(i,j) in enumerate(self._B_indices,start=1+D):
            if i==j:
                entropy[0,k] = -1./_SQRT2
        base = np.vstack([box,-psd_map])
        self._A = {"psd_box": sparse.csc_matrix(base),
                   "gaussian": sparse.csc_matrix(np.vstack([base,entropy]))}
        self._cones = {
            "psd_box": [clarabel.NonnegativeConeT(2),clarabel.PSDTriangleConeT(D+1)],
            "gaussian": [clarabel.NonnegativeConeT(2),clarabel.PSDTriangleConeT(D+1),
                         clarabel.ExponentialConeT()],
        }
        self._solvers: dict[tuple[str,int],Any] = {}

    def _pack(self, q: float, v: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Natural B diagonal is B_ii/sqrt(2); off diagonal is B_ij.
        return np.concatenate(([q],v,[B[i,j]/(_SQRT2 if i==j else 1.)
                                      for i,j in self._B_indices]))

    def _unpack(self, x: np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
        B = np.zeros((self.D,self.D))
        for k,(i,j) in enumerate(self._B_indices,start=1+self.D):
            B[i,j] = B[j,i] = x[k]*(_SQRT2 if i==j else 1.)
        return float(x[0]),x[1:1+self.D].copy(),B

    def _rhs(self, name: str, q: float, v: np.ndarray, B: np.ndarray,
             sd: float) -> np.ndarray:
        b = np.concatenate(([q/sd,(1.-q)/sd],_svec(_joint(q,v,B))/sd))
        if name == "gaussian":
            b = np.concatenate((b,[np.trace(B)/(2.*sd),q/sd,1./sd]))
        return b

    def _call_solver(self, name: str, b: np.ndarray, attempt: int):
        key = (name,attempt)
        if key not in self._solvers:
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            for option,value in _ATTEMPTS[attempt].items():
                setattr(settings,option,value)
            # Full accuracy criteria above determine acceptance. Reduced solver
            # statuses may be inspected but never bypass independent checks.
            self._solvers[key] = clarabel.DefaultSolver(
                self._P,self._zero,self._A[name],b,self._cones[name],settings)
        else:
            self._solvers[key].update(b=b)
        return self._solvers[key].solve()

    def _residuals(self, q: float, v: np.ndarray, B: np.ndarray,
                   sd: float) -> dict[str,Any]:
        box = max(0.,-q,q-1.)
        mineig = float(np.linalg.eigvalsh(_joint(q,v,B))[0])
        entropy = _entropy_twice(q)
        trace = max(0.,float(np.trace(B))-entropy) if np.isfinite(entropy) else None
        return {"box_violation": box,"min_eigenvalue": mineig,
                "trace_violation": trace,"normalized_box_violation": box/sd,
                "normalized_psd_violation": max(0.,-mineig)/sd,
                "normalized_trace_violation": trace/sd if trace is not None else None}

    def _feasible(self, residuals: dict[str,Any], gaussian: bool,
                  tolerance: float = _FEAS_TARGET) -> bool:
        values = [residuals["normalized_box_violation"],
                  residuals["normalized_psd_violation"]]
        if gaussian:
            values.append(residuals["normalized_trace_violation"])
        return bool(all(x is not None and np.isfinite(x) for x in values) and
                    max(values) <= tolerance)

    def _identity_ok(self, q: float, v: np.ndarray, B: np.ndarray,
                     sd: float, gaussian: bool) -> bool:
        if not 0. <= q <= 1.:
            return False
        r = self._residuals(q,v,B,sd)
        # Finite-sample rank-deficient PSD sums can have eigensolver roundoff.
        # Permit only a machine-roundoff bound, also bounded in sd units.
        roundoff = min(64*_EPS*max(1.,np.linalg.norm(_joint(q,v,B),2)),1e-10*sd)
        if r["min_eigenvalue"] < -roundoff:
            return False
        return not gaussian or r["trace_violation"] == 0.

    def _repair_primal(self, q: float, v: np.ndarray, B: np.ndarray,
                       gaussian: bool) -> tuple[float,np.ndarray,np.ndarray,float]:
        """Tiny feasibility blend, *not* a substitute for conic optimization.

        Convexity gives a bound toward the fixed strictly feasible anchor
        (q=.5,v=0,B=.25*I/D). The new objective/gap must still pass acceptance.
        This can repair a roundoff violation without clipping the wrong metric.
        """
        anchor_B = .25*np.eye(self.D)/self.D
        anchor_eig = .25/self.D
        anchor_trace_slack = np.log(2.)-.25
        t = 0.
        if q < 0.:
            t = max(t,-q/(.5-q))
        elif q > 1.:
            t = max(t,(q-1.)/(q-.5))
        eig = float(np.linalg.eigvalsh(_joint(q,v,B))[0])
        if eig < 0.:
            t = max(t,-eig/(anchor_eig-eig))
        if gaussian:
            # First bring q into the domain to evaluate the convex entropy term.
            if t > 0.:
                t = min(1.,t+64*_EPS)
                q = (1.-t)*q+t*.5
                v = (1.-t)*v
                B = (1.-t)*B+t*anchor_B
            violation = float(np.trace(B))-_entropy_twice(q)
            extra = max(0.,violation/(violation+anchor_trace_slack)) if violation>0. else 0.
            if extra > 0.:
                extra = min(1.,extra+64*_EPS)
                q = (1.-extra)*q+extra*.5
                v = (1.-extra)*v
                B = (1.-extra)*B+extra*anchor_B
                t = 1.-(1.-t)*(1.-extra)
        elif t > 0.:
            t = min(1.,t+64*_EPS)
            q = (1.-t)*q+t*.5
            v = (1.-t)*v
            B = (1.-t)*B+t*anchor_B
        return q,v,B,t

    def _dual_bound(self, name: str, b: np.ndarray, z: np.ndarray,
                    x: np.ndarray, q: float, v: np.ndarray, B: np.ndarray,
                    sd: float) -> dict[str,Any]:
        """Repair dual cone roundoff and recompute the centered QP dual bound.

        For Ay+s=b, min .5||y||², a cone-feasible z gives
        lower=-.5||A.T z||²-b.T z. We evaluate the equivalent gap
        .5||y+A.T z||²+s.T z in extended precision. It avoids subtracting two
        large original-coordinate objectives and separately bounds dot roundoff.
        """
        z = np.asarray(z,dtype=float).copy()
        if z.shape != b.shape or not np.all(np.isfinite(z)):
            raise ValueError("nonfinite or malformed solver dual")
        original_z = z.copy()
        z[:2] = np.maximum(z[:2],0.)
        stop = 2+self._psd_size
        dual_matrix = _unsvec(z[2:stop],self.D+1)
        eigenvalues,eigenvectors = np.linalg.eigh(dual_matrix)
        if eigenvalues[0] < 0.:
            dual_matrix = (eigenvectors*np.maximum(eigenvalues,0.))@eigenvectors.T
            # Protect reconstructed PSD from negative eigensolver roundoff.
            margin = 16*_EPS*max(np.linalg.norm(dual_matrix,2),np.finfo(float).tiny)
            dual_matrix += margin*np.eye(self.D+1)
            z[2:stop] = _svec(dual_matrix)
        dual_exp_slack = None
        if name == "gaussian":
            a,c,d = z[stop:]
            if a < 0.:
                # K_exp* iff a<0 and (-a)*exp(c/a)<=e*d, or its closure.
                log_need = np.log(-a)+c/a-1.
                if not np.isfinite(log_need) or log_need > 700.:
                    raise ValueError("unreliable exponential-cone dual")
                need = np.exp(log_need)*(1.+16*_EPS)
                d = max(d,need,np.nextafter(0.,1.))
                z[stop:] = [a,c,d]
                dual_exp_slack = float(1.+np.log(d)-np.log(-a)-c/a)
            else:
                z[stop:] = [0.,max(c,0.),max(d,0.)]
                dual_exp_slack = 0.
        Atz = np.asarray(self._A[name].T@z)
        # Slacks derive from returned physical coordinates after any repair.
        slack = self._rhs(name,q,v,B,sd)
        ld = np.longdouble
        products = slack.astype(ld)*z.astype(ld)
        complementarity = np.sum(products,dtype=ld)
        stationarity = x+Atz
        squared_stationarity = np.sum(stationarity.astype(ld)**2,dtype=ld)
        # The safety term covers our evaluation rounding, not a formal exact
        # arithmetic proof of cone membership or eigenvalue computations.
        error = ld(64*_EPS)*(np.sum(np.abs(products),dtype=ld)+
                               np.sum(x.astype(ld)**2)+np.sum(Atz.astype(ld)**2)+1.)
        raw_gap = .5*squared_stationarity+complementarity
        gap = raw_gap+error
        objective = .5*float(np.dot(x,x))
        reliable = bool(np.isfinite(gap) and raw_gap >= -error)
        return {"normalized_gap": float(max(gap,0.)),
                "dual_lower_bound": float(objective-gap),
                "gap_reliable": reliable,"gap_roundoff_allowance": float(error),
                "normalized_stationarity": float(np.linalg.norm(stationarity)),
                "normalized_complementarity": float(complementarity),
                "dual_exp_log_slack": dual_exp_slack,
                "dual_repair_norm": float(np.linalg.norm(z-original_z)),
                }

    def _diagnostics(self, q: float, v: np.ndarray, B: np.ndarray,
                     sd: float) -> dict[str,Any]:
        return {**self._residuals(q,v,B,sd),"projection_seconds": 0.,
                "solver_status": "not_solved","iterations": 0,
                "normalized_gap": None,"normalized_objective": 0.,
                "normalized_displacement": 0.,"route": "failure",
                "retried": False,"solver_calls": 0,"gap_reliable": False,
                "numerical_certificate": False,"primal_repair_displacement": 0.,
                "primal_repair_fraction": 0.,"gap_target": _GAP_TARGET,
                "feasibility_tolerance_sd": _FEAS_TARGET,
                "coordinate_error_target_sd": .01,
                "certificate_scope": "floating_point_numerical_not_formal_exact_projection"}

    def _project_arm(self, name: str, q0: float, v0: np.ndarray,
                     B0: np.ndarray, sd: float) -> dict[str,Any]:
        started = time.perf_counter()
        diagnostics = self._diagnostics(q0,v0,B0,sd)
        original = {"q": q0,"v": v0.copy(),"B": B0.copy(),"ok": False,
                    "diagnostics": diagnostics}
        gaussian = name == "gaussian"
        if self._identity_ok(q0,v0,B0,sd,gaussian):
            diagnostics.update(solver_status="identity",normalized_gap=0.,
                               dual_lower_bound=0.,gap_reliable=True,
                               numerical_certificate=True,route="identity",
                               projection_seconds=time.perf_counter()-started)
            return {**original,"ok": True}
        b = self._rhs(name,q0,v0,B0,sd)
        attempts = []
        for attempt in range(len(_ATTEMPTS)):
            diagnostics["solver_calls"] += 1
            diagnostics["retried"] = attempt > 0
            diagnostics["solver_settings"] = dict(_ATTEMPTS[attempt])
            try:
                solution = self._call_solver(name,b,attempt)
                diagnostics["solver_status"] = str(solution.status)
                diagnostics["iterations"] = int(solution.iterations)
                diagnostics["solver_primal_objective"] = _finite_number(solution.obj_val)
                diagnostics["solver_dual_objective"] = _finite_number(solution.obj_val_dual)
                diagnostics["solver_primal_residual"] = _finite_number(solution.r_prim)
                diagnostics["solver_dual_residual"] = _finite_number(solution.r_dual)
                y = np.asarray(solution.x,dtype=float)
                if y.shape != (self._width,) or not np.all(np.isfinite(y)):
                    raise ValueError("nonfinite or malformed solver primal")
                dq,dv,dB = self._unpack(y)
                q,v,B = q0+sd*dq,v0+sd*dv,B0+sd*dB
                if not all(np.all(np.isfinite(a)) for a in (q,v,B)):
                    raise ValueError("nonfinite reconstructed moments")
                before = self._pack(q,v,B)
                q,v,B,repair_fraction = self._repair_primal(q,v,B,gaussian)
                increment = self._pack(q-q0,v-v0,B-B0)/sd
                repair = np.linalg.norm(self._pack(q,v,B)-before)/sd
                diagnostics.update(self._residuals(q,v,B,sd))
                diagnostics.update(normalized_displacement=float(np.linalg.norm(increment)),
                                   normalized_objective=.5*float(increment@increment),
                                   primal_repair_displacement=float(repair),
                                   primal_repair_fraction=float(repair_fraction))
                diagnostics.update(self._dual_bound(name,b,np.asarray(solution.z),increment,q,v,B,sd))
                status_ok = diagnostics["solver_status"] in ("Solved","AlmostSolved")
                accepted = (status_ok and diagnostics["gap_reliable"] and
                            diagnostics["normalized_gap"] <= _GAP_TARGET and
                            self._feasible(diagnostics,gaussian))
                attempts.append({"status": diagnostics["solver_status"],
                                 "gap": diagnostics["normalized_gap"],
                                 "solver_settings": dict(_ATTEMPTS[attempt]),
                                 "accepted": bool(accepted)})
                if accepted:
                    diagnostics.update(route="clarabel",numerical_certificate=True,
                                       attempts=attempts,
                                       projection_seconds=time.perf_counter()-started)
                    return {"q": float(q),"v": v,"B": B,"ok": True,
                            "diagnostics": diagnostics}
            except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                diagnostics.update(solver_status="solver_error",error=f"{type(exc).__name__}: {exc}",
                                   gap_reliable=False,numerical_certificate=False)
                attempts.append({"status": "solver_error","error": diagnostics["error"],
                                 "solver_settings": dict(_ATTEMPTS[attempt]),
                                 "accepted": False})
        rejected_keys = ("normalized_gap","dual_lower_bound","gap_reliable",
                         "normalized_objective","normalized_displacement",
                         "box_violation","min_eigenvalue","trace_violation",
                         "normalized_box_violation","normalized_psd_violation",
                         "normalized_trace_violation","primal_repair_displacement",
                         "primal_repair_fraction")
        rejected = {key:diagnostics.get(key) for key in rejected_keys}
        diagnostics.update(self._residuals(q0,v0,B0,sd))
        diagnostics.update(route="failure",numerical_certificate=False,attempts=attempts,
                           returned_original=True,rejected_candidate=rejected,
                           normalized_gap=None,dual_lower_bound=None,gap_reliable=False,
                           normalized_objective=0.,normalized_displacement=0.,
                           primal_repair_displacement=0.,primal_repair_fraction=0.,
                           projection_seconds=time.perf_counter()-started)
        return original

    def project_pair(self, q: float, v: np.ndarray, B: np.ndarray,
                     sd: float) -> dict[str,dict[str,Any]]:
        """Project one noisy raw-moment triple into both fixed constraint sets."""
        if np.ndim(q) != 0 or np.ndim(sd) != 0 or not np.isfinite(sd) or sd <= 0.:
            raise ValueError("q must be scalar and sd must be finite and positive")
        q,sd = float(q),float(sd)
        v,B = np.asarray(v,dtype=float),np.asarray(B,dtype=float)
        if v.shape != (self.D,) or B.shape != (self.D,self.D):
            raise ValueError("v/B shape does not match projector dimension")
        if not all(np.all(np.isfinite(a)) for a in (q,v,B)):
            raise ValueError("q/v/B must contain finite noisy moments")
        symmetry_error = np.max(np.abs(B-B.T))
        if symmetry_error > 64*_EPS*max(1.,np.max(np.abs(B))):
            raise ValueError("B must be symmetric")
        # The release constructs symmetric B. Tolerate only representation
        # roundoff here; failure values always retain the original input.
        B_original = B.copy()
        B = .5*(B+B.T)
        psd = self._project_arm("psd_box",q,v,B,sd)
        if not psd["ok"]:
            psd["B"] = B_original.copy()
        started = time.perf_counter()
        # A PSD projection satisfying the stronger set solves that set too.
        # Approximate reuse also needs its accepted gap and Gaussian residual.
        reuse_r = self._residuals(psd["q"],psd["v"],psd["B"],sd)
        if (psd["ok"] and psd["diagnostics"]["numerical_certificate"] and
                self._feasible(reuse_r,True) and reuse_r["trace_violation"] == 0. and
                0. <= psd["q"] <= 1.):
            diagnostics = {**psd["diagnostics"],**reuse_r,"solver_calls": 0,"retried": False,
                           "iterations": 0,"source_iterations": psd["diagnostics"]["iterations"],
                           "route": "identity" if psd["diagnostics"]["route"]=="identity" else "psd_box_reuse",
                           "reuse_source": "psd_box","source_solver_calls": psd["diagnostics"]["solver_calls"],
                           "projection_seconds": time.perf_counter()-started}
            gaussian = {"q": psd["q"],"v": psd["v"].copy(),"B": psd["B"].copy(),
                        "ok": True,"diagnostics": diagnostics}
        else:
            gaussian = self._project_arm("gaussian",q,v,B,sd)
            if not gaussian["ok"]:
                gaussian["B"] = B_original.copy()
        return {"psd_box": psd,"gaussian": gaussian}

    def project_batch(self, q: np.ndarray, v: np.ndarray, B: np.ndarray,
                      sd: float) -> dict[str,dict[str,Any]]:
        """Sequential batch wrapper; one mutable workspace per ambient D."""
        q,v,B = np.asarray(q,dtype=float),np.asarray(v,dtype=float),np.asarray(B,dtype=float)
        if q.ndim!=1 or v.shape!=(len(q),self.D) or B.shape!=(len(q),self.D,self.D):
            raise ValueError("batch q/v/B shape does not match projector dimension")
        if np.ndim(sd)!=0 or not np.isfinite(sd) or sd<=0.:
            raise ValueError("sd must be finite and positive")
        pairs = [self.project_pair(q[i],v[i],B[i],sd) for i in range(len(q))]
        return {name:{"q": np.array([p[name]["q"] for p in pairs]),
                      "v": np.array([p[name]["v"] for p in pairs]).reshape(len(q),self.D),
                      "B": np.array([p[name]["B"] for p in pairs]).reshape(len(q),self.D,self.D),
                      "ok": np.array([p[name]["ok"] for p in pairs],dtype=bool),
                      "diagnostics": [p[name]["diagnostics"] for p in pairs]}
                for name in ("psd_box","gaussian")}

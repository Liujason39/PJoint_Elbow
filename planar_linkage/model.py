from dataclasses import dataclass
import numpy as np


class GeometryError(ValueError):
    """Invalid geometry or an unreachable actuator length."""


class SingularityError(ValueError):
    """The actuator coordinate cannot locally parameterize the mechanism."""


def _finite(*values):
    if not np.all(np.isfinite(values)):
        raise ValueError("All inputs must be finite")


def _unit(q):
    '''get the np(cos(q), sin(q))'''
    return np.array([np.cos(q), np.sin(q)])


def _normal(q):
    """get the np(-sin(q), cos(q))"""
    return np.array([-np.sin(q), np.cos(q)])


def _intersection(P, Q, radius_p, radius_q, branch, tol):
    '''get the B point coordinate by geometry of intersection of 2 circles'''
    delta = Q - P
    L = np.linalg.norm(delta)
    scale = max(radius_p, radius_q, L)
    eps = tol * scale
    if L <= eps:
        raise GeometryError("Coincident circle centers: no unique intersection")
    if L > radius_p + radius_q + eps or L < abs(radius_p-radius_q)-eps:
        raise GeometryError("Circles do not intersect: unreachable configuration")
    x = (radius_p**2-radius_q**2+L**2)/(2*L)
    h2 = radius_p**2-x*x
    if h2 < -tol*scale**2:
        raise GeometryError("Unreachable configuration")
    e = delta/L
    return P+x*e+branch*np.sqrt(max(0., h2))*np.array([-e[1], e[0]])


@dataclass(frozen=True)
class State:
    '''default as zero point of the p-joint elbow'''
    s: float = 0.21002
    angles: np.ndarray  # theta2, theta3, theta4
    A: np.ndarray
    B: np.ndarray
    branches: tuple[int, int]


@dataclass(frozen=True)
class Mechanism:
    """O2=(0,0), O4=(d,0), O1=(u,v); a=O2A, b=AB, c=O4B.

    O1A is a swiveling actuator of length s (revolute ends), not a
    ground-fixed prismatic guide. All three ground pivots are fixed.
    """
    a: float = 0.048
    b: float = 0.0195
    c: float
    d: float
    u: float
    v: float
    tolerance: float = 1e-12
    singularity_tolerance: float = 1e-10

    def __post_init__(self):
        _finite(self.a, self.b, self.c, self.d, self.u, self.v,
                self.tolerance, self.singularity_tolerance)
        if min(self.a, self.b, self.c, self.d) <= 0:
            raise GeometryError("Link lengths must be positive")
        if not 0 < self.tolerance < 1 or not 0 < self.singularity_tolerance < 1:
            raise ValueError("Tolerances must lie between zero and one")
        if np.hypot(self.u, self.v) == 0:
            raise GeometryError("O1 must differ from O2")

    def position(self, s, branches=(1, 1)):
        """Circle branches: A relative to O2->O1, B relative to A->O4.

        +1 means left of the directed line. These are NOT crossed/open
        labels. Tangencies are allowed for position only.
        """
        _finite(s)
        if s <= 0:
            raise GeometryError("Actuator length must be positive")
        if len(branches) != 2 or any(x not in (-1, 1) for x in branches):
            raise ValueError("branches must contain two values from {-1, +1}")
        A = _intersection(np.zeros(2), np.array([self.u, self.v]),
                          self.a, s, branches[0], self.tolerance)
        B = _intersection(A, np.array([self.d, 0.]), self.b, self.c,
                          branches[1], self.tolerance)
        angles = np.array([np.arctan2(A[1], A[0]),
                           np.arctan2(*(B-A)[::-1]),
                           np.arctan2(B[1], B[0]-self.d)])
        return State(float(s), angles, A, B, tuple(branches))

    def residual(self, angles, s):
        """Position constraints: squared length, x closure, y closure."""
        t2, t3, t4 = np.asarray(angles, dtype=float)
        A = self.a*_unit(t2)
        closure = A+self.b*_unit(t3)-self.c*_unit(t4)-[self.d, 0]
        return np.r_[np.dot(A-[self.u, self.v], A-[self.u, self.v])-s*s,
                     closure]

    def _matrix(self, state):
        t2, t3, t4 = state.angles
        g = self.u*np.sin(t2)-self.v*np.cos(t2) # g = r sin(\triangle)
        # C = dF/dq , F is constraints
        C = np.array([[2*self.a*g, 0, 0],
                      [-self.a*np.sin(t2), -self.b*np.sin(t3), self.c*np.sin(t4)],
                      [self.a*np.cos(t2), self.b*np.cos(t3), -self.c*np.cos(t4)]]) 
        # Dimensionless row scaling avoids a units-dependent singularity test.
        scales = np.array([2*self.a*np.hypot(self.u, self.v),
                           max(self.a,self.b,self.c), max(self.a,self.b,self.c)])
        return C, scales

    def _solve(self, state, rhs):
        C, scales = self._matrix(state)
        scaled = C/scales[:, None]
        sv = np.linalg.svd(scaled, compute_uv=False)
        if sv[-1] <= self.singularity_tolerance*sv[0]:
            raise SingularityError("Singular/near-singular configuration; change coordinates or analyze the constraint directly")
        return np.linalg.solve(scaled, np.asarray(rhs)/scales)

    def jacobian(self, s, branches=(1, 1)):
        """Returns (3,) d(theta2,theta3,theta4)/ds."""
        state = self.position(s, branches)
        return self._solve(state, [2*s, 0, 0])

    def motion(self, s, s_dot=0., s_ddot=0., branches=(1, 1)):
        """Returns state, angular velocity (3,), angular acceleration (3,)."""
        _finite(s_dot, s_ddot)
        state = self.position(s, branches)
        w = self._solve(state, [2*s*s_dot, 0, 0])
        t2, t3, t4 = state.angles
        curvature = 2*self.a*(self.u*np.cos(t2)+self.v*np.sin(t2))*w[0]**2
        radial = (self.a*_unit(t2)*w[0]**2+self.b*_unit(t3)*w[1]**2
                  -self.c*_unit(t4)*w[2]**2)
        alpha = self._solve(state, np.r_[2*(s_dot*s_dot+s*s_ddot)-curvature, radial])
        return state, w, alpha

    def point_jacobian(self, s, point="B", branches=(1, 1)):
        """Returns (2,) d(x,y)/ds for A or B."""
        state = self.position(s, branches)
        J = self.jacobian(s, branches)
        if point == "A":
            return self.a*_normal(state.angles[0])*J[0]
        if point == "B":
            return self.c*_normal(state.angles[2])*J[2]
        raise ValueError("point must be A or B")

    def point_motion(self, s, s_dot=0., s_ddot=0., point="B", branches=(1, 1)):
        """Returns position, velocity, acceleration, each shape (2,)."""
        state, w, alpha = self.motion(s, s_dot, s_ddot, branches)
        if point == "A":
            i, length, p = 0, self.a, state.A
        elif point == "B":
            i, length, p = 2, self.c, state.B
        else:
            raise ValueError("point must be A or B")
        t = state.angles[i]
        return p, length*_normal(t)*w[i], length*(_normal(t)*alpha[i]-_unit(t)*w[i]**2)

    def input_force(self, s, *, torques=(0.,0.,0.), force_A=(0.,0.),
                    force_B=(0.,0.), branches=(1,1)):
        """Required actuator force, positive extending s.

        Loads are external forces ON the mechanism and CCW external link
        torques (theta2,theta3,theta4). Quasi-static, no gravity/inertia/friction
        unless represented by caller-supplied loads. No joint reaction solver.
        """
        tau, fa, fb = map(lambda x: np.asarray(x,dtype=float), (torques, force_A, force_B))
        if tau.shape != (3,) or fa.shape != (2,) or fb.shape != (2,):
            raise ValueError("torques, force_A, force_B require shapes (3,), (2,), (2,)")
        _finite(*tau, *fa, *fb)
        return float(-(self.jacobian(s, branches)@tau
                       +self.point_jacobian(s, "A", branches)@fa
                       +self.point_jacobian(s, "B", branches)@fb))

"""Run from project root: python examples/demo.py (after installation)."""
from planar_linkage import Mechanism
import numpy as np

m = Mechanism(a=1.0, b=2.0, c=1.8, d=2.2, u=-0.8, v=0.3)
s = 1.2
state, omega, alpha = m.motion(s, s_dot=0.02, s_ddot=0.005)
print('A, B [m]:', state.A, state.B)
print('angles [deg]:', np.rad2deg(state.angles))
print('angular velocity [rad/s]:', omega)
print('angular acceleration [rad/s^2]:', alpha)
print('angular Jacobian [rad/m]:', m.jacobian(s))
print('B position Jacobian:', m.point_jacobian(s))
print('required actuator force [N]:', m.input_force(s, force_B=(0., -100.)))

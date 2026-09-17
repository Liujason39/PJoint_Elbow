"""Run from project root: python examples/demo.py """
from planar_linkage import Mechanism
import numpy as np

# m = Mechanism(a=0.048, b=0.0195, c=0.030, d=0.02136, u=-0.05394, v=0.24036)
m = Mechanism() # or use default
s = 0.211
state, omega, alpha = m.motion(s, s_dot=0.02, s_ddot=0.005)
print('A, B [m]:', state.A, state.B)
print('angles [deg]:', np.rad2deg(state.angles))
print('angular velocity [rad/s]:', omega)
print('angular acceleration [rad/s^2]:', alpha)
print('angular Jacobian [rad/m]:', m.jacobian(s))
print('B position Jacobian:', m.point_jacobian(s))
print('required actuator force [N]:', m.input_force(s, force_B=(0., -100.)))


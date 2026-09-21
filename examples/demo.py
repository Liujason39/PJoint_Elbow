"""Run from project root: python examples/demo.py """
from PJointElbow_linkage import Mechanism
import numpy as np

# m = Mechanism(a=0.048, b=0.0195, c=0.030, d=0.02136, u=-0.05394, v=0.24036)
m = Mechanism() # or use default
s = 0.26948 # initial pose = 0.21002
"""the elbow is turn 69.379 in world coordinate"""

# Forward kinematics
state, omega, alpha = m.motion(s, s_dot=0.02, s_ddot=0.005)
print('A, B [m]:', state.A, state.B)
print('angles [deg]:', np.rad2deg(state.angles))
print('angular velocity [rad/s]:', omega)
print('angular acceleration [rad/s^2]:', alpha)
print('angular Jacobian [rad/m]:', m.jacobian(s))
print('B position Jacobian:', m.point_jacobian(s))
print('required actuator force [N]:', m.input_force(s, force_B=(0., -100.)))

# Inverse Kinematics
target_theta4 = np.deg2rad(-59.64780866)

solutions = m.inverse_position(target_theta4)

for i, sol in enumerate(solutions):
    # only choose realitic branch
    if sol.branches != (-1,1):
        continue
    print(f"解 {i + 1}")
    print("s =", sol.s)
    print("A =", sol.A)
    print("B =", sol.B)
    # we should use the branches = (-1, 1)
    print("branches =", sol.branches)

    # 再做一次正運動學，確認回到同一位置
    check = m.position(sol.s, branches=sol.branches)

    np.testing.assert_allclose(
        check.A, sol.A, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        check.B, sol.B, rtol=1e-8, atol=1e-10
    )

    # for inverse angular velocity
    target_omega4 = -0.70199  # rad/s
    s_dot = m.inverse_angular_velocity(sol.s, target_omega4)

    print("\\dot{s} =",s_dot)

    # inverse alpha4
    omega4_target = -0.70199   # rad/s
    alpha4_target = 0.2   # rad/s²
    s_dot, s_ddot = m.inverse_motion(
        sol, omega4_target, alpha4_target
    )
    print("s_dot, s_ddot = ",s_dot, s_ddot)

    # output torques
    tau_out = m.output_torque(
        sol.s,
        actuator_force=100.0,
        branches=sol.branches
    )
    print("output torque of O4->B:", tau_out)


if not solutions:
    print("指定角度無可行逆解")
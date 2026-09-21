import unittest
import numpy as np
from PJointElbow_linkage import Mechanism, GeometryError, SingularityError


def angle_delta(a, b):
    return np.arctan2(np.sin(a-b), np.cos(a-b))

# below are use to check validity of solution 

class Verification(unittest.TestCase):
    def setUp(self):
        self.m = Mechanism(a=1., b=2., c=1.8, d=2.2, u=-.8, v=.3)

    def test_all_branches_and_finite_differences(self):
        m = self.m
        for branches in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            for s in np.linspace(.7,1.6,10):
                with self.subTest(branches=branches,s=s):
                    state = m.position(s, branches)
                    np.testing.assert_allclose(m.residual(state.angles,s),0,atol=2e-14)
                    h=1e-5
                    plus, minus = m.position(s+h,branches),m.position(s-h,branches)
                    fd=angle_delta(plus.angles,minus.angles)/(2*h)
                    np.testing.assert_allclose(m.jacobian(s,branches),fd,rtol=2e-7,atol=2e-8)
                    np.testing.assert_allclose(m.point_jacobian(s,branches=branches),
                                               (plus.B-minus.B)/(2*h),rtol=2e-7,atol=2e-8)
                    # Independently sample s(t), not the analytic velocity formula.
                    dt=1e-4; speed=.17; acceleration=-.08
                    p=m.position(s+speed*dt+.5*acceleration*dt**2,branches)
                    n=m.position(s-speed*dt+.5*acceleration*dt**2,branches)
                    _,w,alpha=m.motion(s,speed,acceleration,branches)
                    afd=(angle_delta(p.angles,state.angles)+angle_delta(n.angles,state.angles))/dt**2
                    np.testing.assert_allclose(alpha,afd,rtol=2e-4,atol=2e-6)
                    _,vb,ab=m.point_motion(s,speed,acceleration,branches=branches)
                    np.testing.assert_allclose(vb,(p.B-n.B)/(2*dt),atol=1e-7)
                    np.testing.assert_allclose(ab,(p.B-2*state.B+n.B)/dt**2,atol=2e-6)

    def test_virtual_power(self):
        m=self.m; s=1.2; speed=.13
        tau=np.array([.7,-.3,1.1]); fa=np.array([2.,-5.]); fb=np.array([-3.,-8.])
        f=m.input_force(s,torques=tau,force_A=fa,force_B=fb)
        dt=1e-6
        p=m.position(s+speed*dt); n=m.position(s-speed*dt)
        power=f*speed+tau@ (angle_delta(p.angles,n.angles)/(2*dt))
        power+=fa@((p.A-n.A)/(2*dt))+fb@((p.B-n.B)/(2*dt))
        self.assertAlmostEqual(power,0,places=7)

    def test_unreachable_and_invalid(self):
        with self.assertRaises(GeometryError): self.m.position(20)
        with self.assertRaises(GeometryError): Mechanism(-1,2,3,4,1,1)
        with self.assertRaises(ValueError): self.m.position(1,(0,1))
        with self.assertRaises(ValueError): self.m.position(float('nan'))

    def test_triangle_singularity(self):
        m=Mechanism(1,2,2,2,-1,0)
        m.position(2)  # Position exists, differential map does not.
        with self.assertRaises(SingularityError): m.jacobian(2)

    def test_fourbar_singularity(self):
        m=Mechanism(1,1,1,3,1,1)
        m.position(1, (-1,1))
        with self.assertRaises(SingularityError): m.jacobian(1, (-1,1))

    def test_length_scaling(self):
        m=self.m; k=1000
        scaled=Mechanism(*(k*x for x in (m.a,m.b,m.c,m.d,m.u,m.v)))
        np.testing.assert_allclose(scaled.jacobian(1.2*k)*k,m.jacobian(1.2),atol=1e-12)


if __name__ == '__main__':
    unittest.main()

"""Compare measured joint_4 angles with the local linkage package.

Run: python3 examples/test/compare_joint4.py
Use --direction -1 if positive joint_3_2 displacement retracts the actuator.
Lengths are converted from mm to m; package angles are converted to degrees.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow direct execution without installing the local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PJointElbow_linkage import GeometryError, Mechanism

S_ZERO_M = 0.26948
BRANCHES = (-1, 1)
TEST_POINTS = (
    (0.00, 0.00),
    (8.15, -19.95),
    (17.85, -40.02),
    (27.95, -60.00),
    (37.70, -79.96),
    (46.85, -100.02),
    (55.40, -119.97),
    (63.80, -140.00),
    (72.45, -159.99),
    (77.00, -170.04),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', type=int, choices=(-1, 1), default=1,
                        help='Actuator length change sign (default: +1, extension).')
    args = parser.parse_args()
    mechanism = Mechanism()
    zero_angle = np.rad2deg(mechanism.position(S_ZERO_M, BRANCHES).angles[2])
    print(f's = {S_ZERO_M:.5f} + ({args.direction:+d}) * displacement_mm / 1000 [m]')
    print(f'branches = {BRANCHES}; joint_4 = degrees(state.angles[2])')
    print(f'World-frame joint_4 at zero displacement: {zero_angle:.6f} deg')
    print('Relative = joint_4 minus its zero-displacement angle, wrapped to [-180, 180).')
    print('Errors = calculated minus measured [deg]; no acceptance tolerance specified.\n')
    print('| Point | Displacement (mm) | s (m) | Measured (deg) | Raw (deg) | Raw error (deg) | Relative (deg) | Relative error (deg) | Status |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |')
    raw_errors, relative_errors = [], []
    for index, (displacement, measured) in enumerate(TEST_POINTS, 1):
        s = S_ZERO_M + args.direction * displacement / 1000.0
        prefix = f'| {index} | {displacement:.2f} | {s:.5f} | {measured:.2f} |'
        try:
            state = mechanism.position(s, branches=BRANCHES)
        except GeometryError as error:
            print(f'{prefix} N/A | N/A | N/A | N/A | Unreachable: {error} |')
            continue
        raw = float(np.rad2deg(state.angles[2]))
        relative = (raw - zero_angle + 180.0) % 360.0 - 180.0
        raw_error = raw - measured
        relative_error = relative - measured
        raw_errors.append(raw_error)
        relative_errors.append(relative_error)
        print(f'{prefix} {raw:.4f} | {raw_error:+.4f} | {relative:.4f} | {relative_error:+.4f} | Reachable |')
    print(f'\nReachable: {len(raw_errors)}/{len(TEST_POINTS)}')
    for label, errors in [('Raw', raw_errors), ('Relative', relative_errors)]:
        if errors:
            values = np.asarray(errors)
            print(f'{label} (reachable points only): RMSE = {np.sqrt(np.mean(values**2)):.4f} deg; '
                  f'max absolute error = {np.max(np.abs(values)):.4f} deg')


if __name__ == '__main__':
    main()

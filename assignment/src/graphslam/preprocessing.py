"""Front-end preprocessing: odometry and lidar (Sec. 9.1).

Before anything reaches the factor graph it has to be turned into the two
quantities the back-end understands: a relative pose with a covariance, and a
list of range-bearing measurements. That conversion is the front-end, and the
two graded functions here are its odometry half.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np

from graphslam.utils import ssa


@dataclass(frozen=True)
class Car:
    """Victoria Park vehicle geometry.

    See ``data/victoria_park/raw/info.txt`` and ``car.bmp`` for the schematic.
    """

    L: float = 2.83  # axle distance
    H: float = 0.76  # centre to wheel encoder
    a: float = 0.95  # laser distance in front of the first axle
    b: float = 0.5  # laser distance to the left of centre


# ---------------------------------------------------------------------------
# Task 1 (a): the kinematic bicycle model as a relative pose
# ---------------------------------------------------------------------------


def relative_pose(vel_encoder: float, steer: float, dt: float) -> gtsam.Pose2:
    """Turn one wheel-encoder reading into a relative pose increment.

    The Victoria Park vehicle gives a forward speed measured at the left rear
    wheel and a steering angle. The kinematic bicycle model converts these into
    a body twist, which is then integrated over ``dt``.

    Two steps, both worth thinking about:

    1. **Encoder frame to body frame.** The encoder sits on the left rear wheel,
       a distance ``H`` off the centre line, so during a turn it travels along a
       different arc than the centre of the rear axle. With ``L`` the axle
       distance, the centre-of-axle speed is

           v_body = v_encoder / (1 - (H/L) tan(steer))

       and the yaw rate is ``omega = (v_body / L) tan(steer)``.

    2. **Twist to pose.** The body twist is ``xi = [v_body, 0, omega]``, held
       constant over the interval. Integrating it exactly gives
       ``Delta_T = Exp(xi * dt)``, which follows a circular arc. Do **not**
       Euler-integrate into ``[v*dt, 0, omega*dt]`` and build a ``Pose2`` from
       that: the difference is second order in ``omega*dt``, small per step and
       clearly visible after a few thousand of them. The Lie group exponential
       is covered in Sec. 6.2.3 of the book; ``gtsam.Pose2.Expmap`` implements it.

    Parameters
    ----------
    vel_encoder : float
        Forward velocity at the encoder (left rear wheel) [m/s].
    steer : float
        Wheel steering angle [rad].
    dt : float
        Duration of the interval [s].

    Returns
    -------
    gtsam.Pose2
        The relative pose increment over the interval.
    """
    # TODO(a): build the body twist and integrate it over dt with the exponential map.
    # BEGIN SOLUTION
    car = Car()

    tangent_steer = np.tan(steer)
    velocity_body = vel_encoder / (1.0 - (car.H / car.L) * tangent_steer)
    yaw_rate = (velocity_body / car.L) * tangent_steer

    twist_body = np.array([velocity_body, 0.0, yaw_rate], dtype=float)

    return gtsam.Pose2.Expmap(twist_body * dt)
    # END SOLUTION


# ---------------------------------------------------------------------------
# Task 1 (b): preintegration
# ---------------------------------------------------------------------------


def preintegrate(
    poses: list[gtsam.Pose2],
    covariances: list[np.ndarray],
) -> tuple[gtsam.Pose2, np.ndarray]:
    """Compound a sequence of relative poses into a single one, with covariance.

    Wheel odometry arrives far faster than lidar scans. Rather than putting a
    pose in the graph for every encoder tick, the increments between two scans
    are compounded into one relative pose and one covariance, and enter the
    graph as a single ``BetweenFactorPose2``. This is preintegration, mentioned
    in Sec. 9.1 -- the same idea that is used for IMU data, in its simplest form.

    The mean is just repeated composition. The covariance is the interesting
    part: composition is nonlinear, so each increment's covariance has to be
    pushed through the Jacobians of the composition,

        Sigma <- H1 Sigma H1^T + H2 Sigma_i H2^T

    where ``H1`` and ``H2`` are the Jacobians of ``compose`` with respect to its
    first and second argument. This is ordinary linear covariance propagation --
    it just happens on a manifold, so the Jacobians are the ones of the group
    operation rather than of a vector sum.

    Parameters
    ----------
    poses : list of gtsam.Pose2
        Relative increments, in time order.
    covariances : list of np.ndarray, each shape=(3, 3)
        Covariance of each increment.

    Returns
    -------
    gtsam.Pose2
        The compounded relative pose.
    np.ndarray, shape=(3, 3)
        Its covariance.

    Notes
    -----
    ``gtsam.Pose2.compose`` takes two optional Jacobian arguments, which must be
    preallocated in Fortran order::

        H1 = np.zeros((3, 3), order="F")
        H2 = np.zeros((3, 3), order="F")
        composed = a.compose(b, H1, H2)

    An empty input list should give the identity pose and a zero covariance.
    """
    # TODO(b): compound the increments and propagate the covariance.
    # BEGIN SOLUTION
    compounded = gtsam.Pose2.Identity()
    compounded_cov = np.zeros((3, 3))

    for increment, increment_cov in zip(poses, covariances):
        H1 = np.zeros((3, 3), order="F")
        H2 = np.zeros((3, 3), order="F")

        compounded = compounded.compose(increment, H1, H2)
        compounded_cov = H1 @ compounded_cov @ H1.T + H2 @ increment_cov @ H2.T

    return compounded, compounded_cov
    # END SOLUTION


# ---------------------------------------------------------------------------
# Lidar preprocessing  (given -- you do not need to change anything below)
# ---------------------------------------------------------------------------


def extract_tree_measurements(scan: np.ndarray, range_limit: float) -> np.ndarray:
    """Convert one raw lidar scan to filtered range-bearing tree measurements."""
    measurements = np.asarray(detect_trees(scan), dtype=float).reshape(-1, 2)
    return measurements[measurements[:, 0] < range_limit]


def detect_trees(scan):
    """
    Convert lidar scan to tree detections in a single 180° laser scan (0.5° resolution)
    Code taken from: https://github.com/ramanans1/EKF-SLAM/blob/master/tree_extraction.py

    Parameters
    ----------
    scan : np.ndarray, shape=(361,)

    Returns
    -------
    z : np.ndarray, shape=(M, 2)
        Detected tree measurements as (range, bearing) pairs.
    """
    M11 = 75
    M10 = 1
    daa = 5 * np.pi / 306
    M2 = 1.5
    M2a = 10 * np.pi / 360
    M3 = 3
    M5 = 1
    daMin2 = 2 * np.pi / 360

    RR = scan

    AA = np.array(range(361)) * np.pi / 360

    (ii1,) = np.where(RR < M11)

    L1 = len(ii1)
    if L1 < 1:
        return []

    R1 = RR[ii1]
    A1 = AA[ii1]

    ii2 = np.flatnonzero((np.abs(np.diff(R1)) > M2) | (np.diff(A1) > M2a))

    L2 = len(ii2) + 1
    ii2u = np.append(ii2, L1 - 1)
    ii2 = np.insert(ii2 + 1, 0, 0)
    # ii2u = int16([ ii2, L1 ])
    # ii2  = int16([1, ii2+1 ])

    # %ii2 , size(R1) ,

    R2 = R1[ii2]
    A2 = A1[ii2]

    A2u = A1[ii2u]
    R2u = R1[ii2u]

    x2 = R2 * np.cos(A2)
    y2 = R2 * np.sin(A2)
    x2u = R2u * np.cos(A2u)
    y2u = R2u * np.sin(A2u)

    flag = np.zeros(L2)

    L3 = 0
    M3c = M3 * M3

    if L2 > 1:
        L2m = L2 - 1
        dx2 = x2[1:L2] - x2u[:L2m]
        dy2 = y2[1:L2] - y2u[:L2m]

        dl2 = dx2 * dx2 + dy2 * dy2
        ii3 = np.flatnonzero(dl2 < M3c)
        L3 = len(ii3)
        if L3 > 0:
            flag[ii3] = 1
            flag[ii3 + 1] = 1

        if L2 > 2:
            L2m = L2 - 2
            dx2 = x2[2:L2] - x2u[0:L2m]
            dy2 = y2[2:L2] - y2u[0:L2m]

            dl2 = dx2 * dx2 + dy2 * dy2
            ii3 = np.flatnonzero(dl2 < M3c)
            L3b = len(ii3)
            if L3b > 0:
                flag[ii3] = 1
                flag[ii3 + 2] = 1
                L3 = L3 + L3b

            if L2 > 3:
                L2m = L2 - 3
                dx2 = x2[3:L2] - x2u[0:L2m]
                dy2 = y2[3:L2] - y2u[0:L2m]

                dl2 = dx2 * dx2 + dy2 * dy2
                ii3 = np.flatnonzero(dl2 < M3c)
                L3b = len(ii3)
                if L3b > 0:
                    flag[ii3] = 1
                    flag[ii3 + 3] = 1
                    L3 = L3 + L3b

    if L2 > 1:
        ii3 = np.array(range(L2 - 1))
        ii3 = np.flatnonzero(
            (A2[ii3 + 1] - A2u[ii3]) < daMin2
        )  # objects close (in angle) from viewpoint.
        L3b = len(ii3)
        if L3b > 0:
            ff = R2[ii3 + 1] > R2u[ii3]  # which object is in the back?
            ii3 = ii3 + ff
            flag[ii3] = 1  # mark them for the deletion
            L3 = L3 + L3b
        iixx = ii3

    if L3 > 0:
        ii3 = np.flatnonzero(flag == 0)
        L3 = len(ii3)
        ii4 = ii2[ii3].astype(np.float64)
        ii4u = ii2u[ii3].astype(np.float64)
        R4 = R2[ii3]
        R4u = R2u[ii3]
        A4 = A2[ii3]
        A4u = A2u[ii3]
        x4 = x2[ii3]
        y4 = y2[ii3]
        x4u = x2u[ii3]
        y4u = y2u[ii3]
    else:
        ii4 = ii2.astype(np.float64)
        ii4u = ii2u.astype(np.float64)
        R4 = R2
        R4u = R2u
        A4 = A2
        A4u = A2u
        x4 = x2
        y4 = y2
        x4u = x2u
        y4u = y2u

    dx2 = x4 - x4u
    dy2 = y4 - y4u
    dl2 = dx2 * dx2 + dy2 * dy2

    ii5 = np.flatnonzero(dl2 < (M5 * M5))
    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R4[ii5]
    R5u = R4u[ii5]
    A5 = A4[ii5]
    A5u = A4u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]

    ii5 = np.flatnonzero((R5 > M10) & (A5 > daa) & (A5u < (np.pi - daa)))

    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]
    dL5 = (A5u + np.pi / 360 - A5) * (R5 + R5u) / 2

    compa = np.abs(R5 - R5u) < (dL5 / 3)

    ii6 = np.flatnonzero(~compa)
    ii6 = ii4[ii6]

    ii5 = np.flatnonzero(compa)
    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]
    dL5 = dL5[ii5]

    auxi = (ii4 + ii4u) / 2
    iia = np.floor(auxi)
    iib = np.ceil(auxi)

    Rs = (R1[iia.astype(int)] + R1[iib.astype(int)]) / 2

    ranges = Rs + dL5 / 2.0
    angles = (A5 + A5u) / 2.0 - np.pi / 2
    angles = ssa(angles)  # wrap to [-pi, pi]
    diameters = dL5

    z = np.vstack((ranges, angles)).T  # keeps the dims

    return z


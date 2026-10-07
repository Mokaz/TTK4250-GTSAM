# GTSAM: the part you need for Task 1

GTSAM's Python package wraps a C++ library, and its own documentation is
written for C++. This page lists every GTSAM call Task 1 needs, with what it
means. Everything here was checked against the GTSAM 4.3 in `environment.yml`.

```python
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
```

Your editor shows the full signatures of everything below (see "Editor support"
in the README). In a terminal, `help(gtsam.Pose2.compose)` does the same.

## Poses: `gtsam.Pose2`

A 2D pose: position `(x, y)` and heading `theta`, an element of SE(2).

| Call | Meaning |
|---|---|
| `gtsam.Pose2(x, y, theta)` | A pose. |
| `gtsam.Pose2()` | The identity: no translation, no rotation. Same as `gtsam.Pose2.Identity()`. |
| `a.compose(b)` or `a * b` | The pose `b`, given relative to `a`, expressed in `a`'s parent frame. Composing body-frame increments in sequence chains them. |
| `a.between(b)` | `a.inverse() * b`: where `b` is, seen from `a`. This is the odometry factor's "measurement" (the `⊖` of (9.5)). |
| `a.inverse()` | The inverse pose. |
| `a.x()`, `a.y()`, `a.theta()` | The components. `a.translation()` is the position as a numpy array. |
| `gtsam.Pose2.Expmap(xi)` | The exponential map (Sec. 6.2.3): a tangent vector `xi = [v_x, v_y, omega]`, as a numpy array, to a pose. |
| `gtsam.Pose2.Logmap(p)` | Its inverse: a pose to its tangent vector. |
| `a.range(point)` | Distance from `a` to a 2D point. |
| `a.bearing(point)` | Direction of the point, seen from `a`, relative to its heading. Returns a `gtsam.Rot2`, not a float. |
| `a.transformFrom(p_body)` | A point given in `a`'s body frame, expressed in the world frame. `a.transformTo(p_world)` goes the other way. |

The tangent space, and therefore every pose covariance and every pose Jacobian
in this code, is ordered `[x, y, theta]` and lives at the pose itself (Sec. 6.2).

## Rotations and points

| Call | Meaning |
|---|---|
| `gtsam.Rot2(angle)` | A 2D rotation by `angle` radians. `gtsam.Rot2.fromAngle(angle)` is the same. |
| `r.theta()` | Back to an angle in radians, wrapped to (−π, π]. |
| `r.rotate(p)` | Rotate a 2D point. |
| `gtsam.Point2(x, y)` | A 2D point. In Python this is simply a numpy array of shape (2,); any such array works wherever GTSAM wants a point. |

## Jacobians: how GTSAM hands them back

Many GTSAM methods can also return their Jacobians. You ask for them by passing
numpy arrays as extra arguments, which GTSAM fills in place. They must already
exist, be `float64`, and have exactly the right shape; anything else raises a
`TypeError`. `order="F"` (column-major) is the conventional choice: older GTSAM
versions required it, and it is never wrong.

```python
H1 = np.zeros((3, 3), order="F")
H2 = np.zeros((3, 3), order="F")
c = a.compose(b, H1, H2)        # H1 = dc/da, H2 = dc/db, both 3x3

H_pose = np.zeros((1, 3), order="F")
H_point = np.zeros((1, 2), order="F")
r = a.range(point, H_pose, H_point)   # one row each: range is a scalar
```

The Jacobians are with respect to the tangent spaces, the same convention as the
covariances, so `H @ Sigma @ H.T` is the right way to propagate a pose
covariance through them.

## Keys

Every variable in the graph has an integer key. `X(k)` is pose number `k` and
`L(j)` is landmark number `j`. A key encodes its letter, so every `L` key is
smaller than every `X` key; Task 1 (e) is about exactly that.

## Building the graph

| Call | Meaning |
|---|---|
| `graph.add(factor)` | Add a factor to a `gtsam.NonlinearFactorGraph`. |
| `gtsam.BetweenFactorPose2(key1, key2, measured, noise_model)` | A relative-pose measurement `measured` (a `Pose2`) between two pose variables: the odometry term of (9.6). |
| `gtsam.BearingRangeFactor2D(pose_key, point_key, bearing, range, noise_model)` | A landmark measurement: the last term of (9.6). The bearing is a `gtsam.Rot2`, the range a float. Note the order: **bearing first**, while this code base stores measurements as `[range, bearing]`. |

## Noise models

| Call | Meaning |
|---|---|
| `gtsam.noiseModel.Gaussian.Covariance(cov)` | Gaussian noise with a full covariance matrix. |
| `gtsam.noiseModel.Diagonal.Sigmas(np.array([...]))` | Independent noise per component, given as standard deviations. Throws away any correlation. |

The given code builds the landmark noise model for you (`bearing_range_noise_model`),
already ordered `[bearing, range]` to match the factor.

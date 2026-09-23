# Installing GTSAM

The back-end uses [GTSAM](https://gtsam.org) through its Python bindings. For
most platforms `uv sync` (or `pip install -e ".[dev]"`) just works, because
GTSAM ships pre-built wheels.

## If there is no wheel for your platform

Check first, since this changes between releases:

```sh
pip download gtsam --no-deps -d /tmp/gtsam-wheel
```

If that fails, the options in order of least effort:

1. **Windows: use WSL.** Install Ubuntu from the Microsoft Store, then follow
   the Linux path inside it. This is by far the least painful route.
2. **Use conda-forge**, which packages GTSAM for some platforms the PyPI wheels
   do not cover.
3. **Build from source.** Follow the
   [official instructions](https://github.com/borglab/gtsam/blob/develop/INSTALL.md)
   and the [Python wrapper README](https://github.com/borglab/gtsam/blob/develop/python/README.md).
   Build in Release mode -- a Debug build carries extra checks and is
   substantially slower, which matters for the runtime discussion in Task 3.

## Covariance recovery across GTSAM versions

The API for recovering a joint marginal covariance has changed between GTSAM
releases. `graphslam.factor_graph.query_joint_covariance` probes once at
startup and picks the fastest method your build supports:

| method | what it does | needs |
|---|---|---|
| `bayes_tree` | asks the Bayes tree directly | a recent GTSAM |
| `marginals` | rebuilds a `Marginals` over the whole graph | any GTSAM 4.x |
| `elimination` | linearize, marginalize, invert the Hessian | any GTSAM 4.x |

It prints which one it chose on the first step. All three compute the same
quantity; they differ in cost, sometimes by a lot. You can pin one with
`backend.covariance_method` in the config, and comparing them on Victoria Park
is a reasonable thing to put in the report.

If none of them work on your build, that is a bug in the handout and not in
your code -- please report it on the forum.

## GTSAM installation

The factor graph backend uses GTSAM through its Python bindings.

### On Linux or macOS

```bash
uv sync
```

PyPI publishes GTSAM 4.3 wheels for Linux (x86-64, ARM64) and macOS
(universal2), so nothing is compiled.

### On Windows

There is no Windows wheel on PyPI — `uv sync` fails with "no source
distribution or wheel for the current platform". Use conda-forge, which builds
GTSAM 4.3.0 for win-64:

```bash
conda env create -f assignment/environment.yml
conda activate ttk4250
pip install -e . --no-deps
```

That environment file lives in `assignment/` but serves both projects; the
dependency lists are the same. Note that `master_code` and `graphslam` both
define `run_sim` and `run_real` entry points, so only one of them can be
installed editable in a given environment at a time. Re-run
`pip install -e . --no-deps` from whichever project directory you are working
in, or make a second environment.

### A note on the version pin

This project previously required `gtsam-develop`, the development pre-release.
The reason was covariance recovery: `slam.py` queries the joint marginal over
the current pose and the nearby landmarks with

```python
isam2.jointMarginalCovariance(cov_query).fullMatrix()
```

and `ISAM2::jointMarginalCovariance` was only exposed through the Python wrapper
in development builds. During early development GTSAM was built from source for
the same reason, because the C++ covariance-recovery functions were not wrapped
at all.

That work has since landed upstream. **GTSAM 4.3 exposes the incremental
joint-covariance query on `ISAM2` in the stable release**, along with
`jointMarginalInformation`, `marginalInformation`, and Bayes-tree caching of the
shortcut conditionals that make the query fast. The dependency is now
`gtsam>=4.3,<5` and neither a pre-release nor a source build is needed.

### Building from source

Still possible, and what you want if you need to modify the C++ library or
expose something the wrapper does not. Use a Release build: Debug builds carry
extra checks and are substantially slower, which matters for any runtime
measurement.

- [Installation](https://github.com/borglab/gtsam/blob/develop/INSTALL.md)
- [Python wrapper](https://github.com/borglab/gtsam/blob/develop/python/README.md)
- [Development guide](https://github.com/borglab/gtsam/blob/develop/DEVELOP.md)

### Still not available on a stock build

`jointMarginalSupportCliqueCount`, the diagnostic that counts how many Bayes
tree cliques a covariance query had to touch, is *not* part of 4.3. The call in
`slam.py` remains commented out, and `num_support_cliques` is still logged as
zero unless you are running a patched build.

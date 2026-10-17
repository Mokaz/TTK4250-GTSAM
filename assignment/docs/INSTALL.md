# Installing

```sh
conda info                      # conda version must be 25 or newer
conda env create -f environment.yml
conda activate ttk4250_ga2
pip install -e . --no-deps
```

Identical on Windows, Linux and macOS. Install
[Miniconda](https://www.anaconda.com/download) first — the **minimal** installer,
not the full Anaconda Distribution. GTSAM comes from conda-forge, because PyPI
has no Windows wheel for it; `environment.yml` sets the channel for you.

## If you already have Anaconda

Run `conda info` before anything else. You want `conda version` at 25 or newer:
an older conda, which an existing Anaconda install often supplies, cannot solve
this environment.

## Accept or remove the Anaconda channel Terms of Service

On a fresh Miniconda, the very first `conda env create` stops before doing
anything:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the
following channels. Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
    - https://repo.anaconda.com/pkgs/msys2
```

Miniconda is configured to use Anaconda's own channels, and recent conda refuses
to contact them until you have accepted their terms. Two ways past it.

**Either accept them**, exactly as the error says:

```sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

**Or remove those channels entirely**, which is the better option here. This
assignment uses nothing from them — every package comes from conda-forge, and
`environment.yml` says so:

```sh
conda config --add channels conda-forge
conda config --remove channels defaults
conda config --set channel_priority strict
```

If the middle command complains that `defaults` is not in the `channels` key, it
was already absent; carry on. After this the Terms of Service prompt does not
appear, because conda never consults Anaconda's channels.

## Check the install

```sh
python -c "import gtsam; print(hasattr(gtsam.ISAM2(), 'jointMarginalCovariance'))"
pytest
```

`True`, then the test suite runs. Before you have done Task 1, only the tests of
the given code pass; the rest fail with `NotImplementedError`.

`pip install -e . --no-deps` only registers the package and its commands
(`run_sim`, `run_real`, `plot_run`). `--no-deps` matters: without it pip would
try to fetch GTSAM from PyPI and undo what conda just did.

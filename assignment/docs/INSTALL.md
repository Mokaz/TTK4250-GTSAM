# Installing

```sh
conda info                      # conda version must be 25 or newer
conda env create -f environment.yml
conda activate ttk4250_ga2
pip install -e . --no-deps
```

Identical on Windows, Linux and macOS. Install
[Miniconda](https://www.anaconda.com/download) first — the **minimal** installer,
not the full Anaconda Distribution.

Why the minimal one: a full Anaconda base carries `conda-build`, Navigator and
several hundred packages that all pin `conda`, which can hold it years behind.
A conda that old cannot solve this environment. Worse, if Anaconda is already on
the machine it hooks your shell and will quietly serve its own old conda even
after you install something newer. The section below is entirely about that.

The environment itself is pulled from conda-forge regardless of which installer
you use — `environment.yml` sets the channel explicitly.

## First: accept or remove the Anaconda channel Terms of Service

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

Verify the install with:

```sh
python -c "import gtsam; print(hasattr(gtsam.ISAM2(), 'jointMarginalCovariance'))"
pytest
```

`True`, then a green test suite (75 tests as this is written).

`pip install -e . --no-deps` only registers the package and its entry points
(`run_sim`, `run_real`, `plot_run`). `--no-deps` matters: without it pip would
try to fetch GTSAM from PyPI and undo what conda just did.

## Why conda and not pip or uv

**PyPI has no Windows wheel for GTSAM.** The 4.3.0 release publishes wheels for
Linux (x86-64, ARM64) and macOS (universal2) only, and GTSAM's own documentation
says so directly. On Windows, `pip install gtsam` and `uv sync` both fail with
"no source distribution or wheel for the current platform".

conda-forge builds GTSAM 4.3.0 for **win-64, linux-64, linux-aarch64, osx-64 and
osx-arm64**, so it is the one channel that covers everyone. No WSL, no Docker,
no admin rights, no building from source.

If you are on Linux or macOS and prefer `uv`, it will work — `pyproject.toml`
still carries the dependency list. But the supported path, and the one the
assignment is tested against, is the environment file.

## Why GTSAM 4.3 specifically

The back-end recovers the joint covariance of the local map every step with

```python
isam2.jointMarginalCovariance(gtsam.KeyVector(keys))
```

and `ISAM2.jointMarginalCovariance` first reached a stable release in 4.3. On
4.2 it does not exist and you get an `AttributeError` on the first step that
sees a landmark.

Under the hood this is a Steiner-tree query over the Bayes tree: GTSAM visits
only the cliques on the paths joining the queried variables and compresses long
non-branching stretches with shortcut conditionals. The cost tracks how close
those variables are in the tree rather than the size of the map, which is what
makes a joint covariance per time step affordable — see Sec. 9.4 and 9.5 of the
book.

## Two ways to recover the covariance

`backend.covariance_method` selects between them. Both compute the same matrix.

| Value | How | Cost |
| --- | --- | --- |
| `bayes_tree` | `isam2.jointMarginalCovariance`, incremental, on the Bayes tree | roughly flat in map size |
| `marginals` | `gtsam.Marginals` rebuilt from the whole graph every call | grows with the whole graph |

`bayes_tree` is the default and the one to use. `marginals` is there so you can
time the two against each other — compare the logged
`duration_covariance_extraction` on Victoria Park and you have the argument of
Sec. 9.3.2–9.5 as a single plot.

## If you already have Anaconda: read this

Every problem below comes from the same root cause — an existing Anaconda
install supplying an **old conda**, which cannot solve this environment. On a
machine with only Miniconda and no prior Anaconda, none of this happens.

Check first, before anything else:

```
conda info
```

You want `conda version` at 25 or newer, `base environment` pointing at your
Miniconda directory, and `virtual packages` listing a real `__win` such as
`__win=10.0.26100`. If it says `__win=0=0`, that conda is too old and the rest
of this section applies.

### Symptom 1: the solve fails on `__win`

```
package gtsam-4.3.0-... requires libboost >=1.90.0,<1.91.0a0, but none of the
providers can be installed
   libboost ... requires __win >=10, which conflicts with any installable
   versions previously reported
```

Nothing is missing. `__win` is conda's virtual package for the Windows version;
old conda reports it as `0`, so `>=10` can never be satisfied. conda-forge now
pins Windows versions this way, as it does `__glibc` on Linux.

`CONDA_OVERRIDE_WIN=10` does **not** help — the override arrived in the same
conda releases that fixed the detection.

Neither does updating Anaconda's conda, by command line or by Navigator.
Anaconda pins conda in its base environment, so `conda update -n base conda`
reports "All requested packages already installed" and changes nothing.

**Fix:** use the new conda by its full path. Not `conda`, and not
`condabin\conda.bat` — that wrapper honours the `CONDA_EXE` variable Anaconda
sets, so it hands straight back to the old conda. Use the executable:

```
C:\Users\<you>\miniconda3\Scripts\conda.exe env create -f environment.yml
```

### Symptom 2: your new conda prompt runs Anaconda's conda

`conda info` shows `base environment: ...\anaconda3` even though you opened the
Miniconda (or Miniforge) prompt.

Anaconda's `conda init cmd.exe` writes an AutoRun entry into the registry that
fires in *every* command window, including the one your new install provides.

**Fix:** call the new conda's `Scripts\conda.exe` by its full path. To fix it
permanently instead, run `conda init --reverse --all` from an Anaconda prompt —
or uninstall Anaconda, which is cleaner — or delete the key:

```
reg delete "HKCU\Software\Microsoft\Command Processor" /v AutoRun /f
```

This affects `cmd.exe` only; PowerShell keeps its Anaconda hook, and
`conda init cmd.exe` puts it back.

### Symptom 3: the environment builds, then will not activate

```
EnvironmentNameNotFound: Could not find conda environment: ttk4250_ga2
```

The environment was created under your new install's `envs`, but the conda
doing the activating is Anaconda's and does not look there.

**Fix:** activate by full path.

```
conda activate C:\Users\<you>\miniconda3\envs\ttk4250_ga2
```

Do *not* work around this by calling `envs\ttk4250_ga2\python.exe` directly.
Conda relies on activation to put `Library\bin` on `PATH`, which is where
GTSAM's DLLs live, and you will get an opaque DLL load error instead.

## Other problems

**The solve takes forever, or picks strange builds.** The `defaults` channel is
winning. The environment file sets `nodefaults` for this reason; if you built
the environment some other way, use `conda config --set channel_priority strict`.

**`run_sim: command not found`.** The environment is not active, or you skipped
`pip install -e . --no-deps`.

**An `AttributeError` on a GTSAM name.** Check the version with
`conda list gtsam`. Note also that 4.3 removed everything deprecated in 4.2, so
snippets from older tutorials may not apply.

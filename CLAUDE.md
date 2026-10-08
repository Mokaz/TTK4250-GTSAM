# TTK4250 Group Assignment 2 — factor graph SLAM

Replacing the old EKF-SLAM Group Assignment 2 with a factor-graph SLAM assignment
built on a GTSAM/iSAM2 master's-thesis code base. Martin is the course staff
member building it; students are 4th/5th-year NTNU cybernetics.

## Layout

```text
src/master_code/      original thesis code, essentially untouched
assignment/           the teaching fork, package `graphslam` — THE REFERENCE SOLUTION
handout/              generated student skeleton (never edit by hand)
latex/group-assignment-2-gtsam/
                      ga2_*.tex is the new assignment text; graded2.tex and
                      task0*.tex are the ORIGINAL EKF-SLAM files — do not touch
latex/env/            LaTeX preamble
```

After any change to a `ga2_*.tex` file or to the LaTeX snippets, rebuild the
PDF in place so `latex/group-assignment-2-gtsam/build/ga2_graded2.pdf` is
current (the folder is gitignored; Martin reads that file):

```sh
cd latex/group-assignment-2-gtsam
latexmk -lualatex -shell-escape -interaction=nonstopmode -outdir=build ga2_graded2.tex
```

It needs LuaLaTeX (fontspec) and `-shell-escape` (minted). Check the log for
`!` errors, and look at the rendered page when a change adds long inline code:
`\pythoninline` does not wrap, so a long call runs past the margin.
If the build fails with "Serious error that appeared not to generate a log
file" or a locked `.data.minted` file, Martin's editor is building into the same
`build/` at the same time: wait for its `latexmk` to finish and rerun.

`assignment/` is the source of truth. `handout/` is regenerated from it with
`python tools/make_handout.py --out ../handout`; the LaTeX code snippets come
from the same pass with `--latex`.

`assignment/typings/` holds generated GTSAM type stubs (GTSAM is a compiled
module with `py.typed` and no `.pyi`, so editors otherwise see nothing in it).
Regenerate with `tools/make_gtsam_stubs.py` whenever the GTSAM pin changes; it
is copied into the handout. `make_handout.py` empties `handout/` instead of
deleting it, because Windows cannot delete a folder an editor or shell has open.

`assignment/docs/GTSAM.md` is the student-facing reference for every GTSAM call
Task 1 needs; every fact in it was checked against GTSAM 4.3. If a task starts
needing a new GTSAM call, add it there. Note: in 4.3, Jacobian out-arguments
need float64 and the exact shape (else `TypeError`); C order works as well as
`order="F"`, so docs say F is conventional, not required.

`handout/` is gitignored on purpose: it is fully generated, it is ~12 MB
because the generator copies `data/` into it, and a committed copy can drift
from `assignment/`. Zip it at release time instead of tracking it.

## Environment

conda, not uv — GTSAM 4.3 has no Windows wheel on PyPI. Environment name
`ttk4250_ga2`, built from `assignment/environment.yml`, with the package
registered via `pip install -e . --no-deps`.

Shell activation does not reliably survive into an agent's shell on Windows.
Prefer running things explicitly in the environment:

```sh
conda run --no-capture-output -n ttk4250_ga2 pytest
conda run --no-capture-output -n ttk4250_ga2 run_real --steps 2000 --no-show-plots
```

If you do activate first (`conda activate ttk4250_ga2`), plain `pytest`,
`run_sim`, `run_real` and `plot_run` all work.

`docs/INSTALL.md` carries the student-facing setup, including three Windows
traps that cost real time (old Anaconda conda cannot solve the env; Anaconda's
`conda init cmd.exe` AutoRun hook hijacks a fresh Miniconda prompt; a fresh
Miniconda stops on `CondaToSNonInteractiveError`).

## The editable-install trap — read this before debugging anything

`assignment/` and `handout/` both install as the package `graphslam`, and
`pip install -e .` registers the **environment**, not the directory you are
standing in. After `cd handout && pip install -e . --no-deps`, every later
`pytest` / `run_sim` / `run_real` runs the handout's code from any directory,
including `assignment/`. The symptom is a traceback whose paths say
`handout\src\graphslam` and a `NotImplementedError` from a task that is
implemented.

Always check first:

```sh
python -c "import graphslam, pathlib; print(pathlib.Path(graphslam.__file__).parent)"
```

## The ten graded functions

Each is wrapped in `# TODO(<task>):` + `# BEGIN SOLUTION` / `# END SOLUTION`.
`make_handout.py` parses exactly those markers, so do not reformat them.

| | Function | File | Book anchor |
|---|---|---|---|
| a | `relative_pose` | `preprocessing.py` | (9.5), Sec. 6.2.3 |
| b | `preintegrate` | `preprocessing.py` | Sec. 9.1 |
| c1 | `predict_pose` | `factor_graph.py` | (9.5), (9.13) |
| c2 | `add_odometry_factor` | `factor_graph.py` | (9.4), (9.6) |
| c3 | `add_landmark_factor` | `factor_graph.py` | (9.6) |
| d | `predict_measurement` | `factor_graph.py` | (9.7), (9.8) |
| e | `assemble_joint_covariance` | `factor_graph.py` | Sec. 9.4.1, (9.27)–(9.28) |
| f | `innovation_covariance` | `factor_graph.py` | **(9.26)** |
| g1 | `inverse_measurement` | `factor_graph.py` | (9.2) |
| g2 | `TentativeLandmark.is_confirmed` | `landmark_manager.py` | track initiation |

Book references are to `sf2026c.pdf` (the 2026 edition with Lie theory).

## Three conventions that silently produce a plausible wrong system

**Ordering.** Measurements are `[range, bearing]` everywhere in this code base.
GTSAM's `BearingRangeFactor2D` wants bearing first.

**Frames.** The covariance GTSAM reports for a `Pose2`, and the Jacobians from
`Pose2.range` / `Pose2.bearing`, live in the tangent space at the current
estimate. Used together they are consistent; mixing either with a hand-derived
`d/d[x, y, theta]` is not. Same applies to NEES — see
`graphslam.utils.pose2_tangent_error`.

**Block order.** A joint covariance must have its blocks in the same order as
the Jacobians it meets in (f). Task (e) assembles P from GTSAM's
`JointMarginal.at(key_i, key_j)`, which names the variables and so cannot be
misordered (decided 8 Oct). Do not lay P out from `.fullMatrix()`: in 4.3 it
follows the request order, older GTSAM documented it as key-sorted, and the
thesis code assumed the latter. On 1 Oct that mismatch silently scrambled P
(14 m from GNSS instead of 1.5 m on Victoria Park, every test green). The back-end
calls GTSAM directly (`isam2.jointMarginalCovariance`, or `gtsam.Marginals` for
`covariance_method: marginals` and the batch solver); there is no wrapper and no
`elimination` route any more. `test_the_back_end_returns_P_with_the_pose_block_first`
guards the whole path for every route.

## Testing

89 tests, ~8 s. The suite is also the grading instrument, so a test's failure
message is student-facing: write them that way.

Run the handout through the suite after any change to the solution blocks — the
expected result is that only the tests touching given code pass, currently 30
of 89. The assignment is pass/fail (decided 2 Oct 2026), so this is a sanity
check that no graded work hides behind a given-code test, not a scoring floor.

`tests/test_plotting.py` exists because every other test runs with
`save_plots=False`; without it a rename in the plotting stack escapes the suite
and only explodes after a finished multi-minute run. Keep that covered.

## Measured

Victoria Park, 2000 steps, after the block-order fix: 141 landmarks, 1.5 m RMS
from GNSS (GNSS ANIS 1.07 with the 1 m sigma), landmark ANIS 0.27.

Absolute timings vary a lot with machine load: the same pre-fix code took 22 s
in one session and 42 s in another. Compare back to back only. On 1 Oct, back to
back: fixed 35 s of step time (optimization 11.2 / covariance 11.0 /
association 7.3 / local-map extraction 4.7), pre-fix 40 s. Per-step median
17.5 ms, 95th percentile 33 ms.

Full run (6 Oct, quiet machine; the data set has 7247 scan steps, so
`--steps 7300` is simply "all of it"). Back to back with a 20.6 s 2000-step run:

| Tuning | Run time | Landmarks | RMS vs GNSS | GNSS ANIS (1 m) | Landmark ANIS |
|---|---|---|---|---|---|
| `real_default.yaml` | 146 s | 288 | 1.79 m | 1.36 | 0.26 |
| odometry 0.1 m / 0.3° | 291 s | 1282 | 218 m | ~2e4 | 0.49 |

Steps 5000–7000 cost about 3x the earlier ones (revisits, bigger cliques):
per-step median 13.7 ms, 95th percentile 61 ms, max 88 ms. The lowered-noise
run is the one Task 3 asks for; it diverges completely and takes twice as long.

## Known gaps and open decisions

- **Tuning intent for `real_default.yaml` (decided 2 Oct).** The shipped values
  are a working baseline, deliberately not consistent: GNSS-consistent (ANIS
  1.07, 1.5 m RMS) but locally conservative (landmark ANIS 0.27). Lowering
  odometry noise pushes landmark ANIS towards 1 but breaks the run (370–470
  landmarks, 35–70 m from GNSS), because independent-increment odometry cannot
  be right both per step and over a loop. Task 3 now says so and asks for one
  tuning set that lowers the odometry noise. Do not "fix" the tuning.
- **Pass/fail (decided 2 Oct).** The number of tests passed is not a score;
  it feeds an overall evaluation of the submission together with the report.
  `ga2_graded2.tex` and `ga2_task01_implement.tex` say so. The text talks about
  neither grades nor points anywhere; keep it that way.
- `jointMarginalSupportCliqueCount` is not in GTSAM 4.3, so
  `num_support_cliques` is always zero. The plots that used it are skipped
  unless a patched build fills it in.
- No `--steps` limit is needed for Task 3 (see Measured). Task 3 has a "Plan
  your runs" paragraph: explore with `--steps 2000`, report only full runs
  (the runtime plot's expensive regime starts after scan 5000). Its numbers
  (2.5 min, ~140 landmarks at 2000, 20 s) come from the Measured section;
  update both together.
- `Car.a` / `Car.b` (lidar offset) parsed but unused. Backlogged (2 Oct) as a
  possible optional exercise; the system works without it.
- Reference solution not yet solved once from the student side and timed.
- Victoria Park start heading is `victoria_park.initial_heading_deg` (36°,
  inherited from the EKF-SLAM assignment). Best-fit rotation onto GNSS over
  2000 steps is 35.99°, so leave it. It is unobservable: changing it rotates
  the whole solution and only affects the GNSS comparison.
- JCBB's skip-branch bound (`data_association.py`, `n + (M - j - 2) >=`) is
  the strict one from Neira & Tardós: it prunes branches that could only tie
  on pairings. The book (Sec. 7.3.2) says "at least as many". The book's
  version gave identical results on 2000 Victoria Park steps and 48% more
  association time. Kept, with a comment saying all of this (6 Oct).

- **Ground-truth associator removed (7 Oct).** The fork had added
  `association.method: gt` (the thesis only had a stub); it was broken (637
  landmarks instead of 78, a fixed 2 m gate against 1.35 m of bearing noise at
  range). Replaced by `graphslam/evaluation.py`: after every simulated run, map
  landmarks are matched one-to-one (Hungarian, 3 m) to the true landmarks the
  run observed, and found / missed / duplicates / spurious are printed, shown in
  the final-snapshot title and by `plot_run`. Task 2's front-end/back-end
  question now uses it. Baseline: 76 of 78, 0 duplicates.
- **JCBB blow-up (found 7 Oct).** Duplicate landmarks give each measurement
  2-3.6 compatible candidates and JCBB visits roughly their product; runs go
  from 10 s to 5 min, or hours. `sim_default.yaml`'s M 1, N 1 is the
  amplifier. **Changed 7 Oct: `sim_default.yaml` now ships M 2, N 3, gate
  0.5.** Baseline: 76 of 78 mapped, 0 duplicates, 0.39 m RMSE, ANIS 0.65, ANEES
  0.68, 10 s. True R: 14 s instead of 285 s (13 duplicates instead of 83). Still
  slow: true R plus halved odometry (53 s), everything at a fifth (5 min, fails),
  `range_local` 40 (hangs) -- those need a JCBB work limit if anything.
  The thesis code on GTSAM 4.3 shows the same blow-up with R halved (214
  landmarks, 345 s), but not with the true R (80, 9 s), because its scrambled
  S is wide enough that JCBB never rejects a true pairing. With a correct S,
  rejections happen at the 1 - alpha rate, and M 1, N 1 turns each into a
  permanent duplicate. M 1, N 1 is only safe while S is conservative.
  Ctrl+C now saves the run so far, marked `aborted` in metadata.json;
  `--no-save-on-abort` on run_sim/run_real turns that off (7 Oct).
- Simulated data's true noise: range sd 0.05 m, bearing sd 0.99 deg. The
  shipped `sigma_range` 0.2 is why Task 2's landmark ANIS is 0.65.

## Working style for this repo

Explanations and hints for the graded functions live in the task text
(`ga2_task01_implement.tex`), not in their docstrings (decided 7 Oct, after
three stale docstring claims turned up in one day). A graded function's
docstring is a one-line summary naming its task, plus Parameters and Returns
(types, shapes, units, the [range, bearing] ordering). New hints go into the
LaTeX; GTSAM API facts go into `docs/GTSAM.md`.

Do not change code when asked only to review it. Do not touch the original
EKF-SLAM LaTeX files. When a design decision affects the students' experience
(scoring, what a task tests, what the shipped tuning demonstrates), surface it
rather than picking silently.

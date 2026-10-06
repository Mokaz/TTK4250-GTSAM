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

`assignment/` is the source of truth. `handout/` is regenerated from it with
`python tools/make_handout.py --out ../handout`; the LaTeX code snippets come
from the same pass with `--latex`.

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
| e | `reorder_joint_covariance` | `factor_graph.py` | Sec. 9.4.1, (9.27)–(9.28) |
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

**Block order.** GTSAM 4.3's `bayes_tree` and `marginals` joint-covariance
queries return blocks in the order the keys were requested; the `elimination`
route returns them in key order (every `L` key sorts before every `X` key).
`query_joint_covariance` therefore always queries with sorted keys, which is
what task (e) assumes. Until 1 Oct 2026 it did not, and the default path
silently scrambled P: 14 m from GNSS instead of 1.5 m on Victoria Park, with
every test green. `test_local_joint_covariance_matches_gtsam_block_by_block`
guards it now; keep any new covariance route covered by it.

## Testing

82 tests, ~8 s. The suite is also the grading instrument, so a test's failure
message is student-facing: write them that way.

Run the handout through the suite after any change to the solution blocks — the
expected result is that only the tests touching given code pass, currently 27
of 82. The assignment is pass/fail (decided 2 Oct 2026), so this is a sanity
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
- Full 7300-step Victoria Park run not yet timed; the `--steps` figure in the
  assignment text is provisional.
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

## Working style for this repo

Do not change code when asked only to review it. Do not touch the original
EKF-SLAM LaTeX files. When a design decision affects the students' experience
(scoring, what a task tests, what the shipped tuning demonstrates), surface it
rather than picking silently.

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
A collision can also leave a corrupt PDF while your own `latexmk` exits 0, so
check `tasklist | findstr latexmk` before building, and verify afterwards that
`pdftotext build/ga2_graded2.pdf -` prints no "Syntax Error" lines.

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
| g | `inverse_measurement` | `factor_graph.py` | (9.2) |
| h | `TentativeLandmark.is_confirmed` | `landmark_manager.py` | track initiation |

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

95 tests, ~10 s. The suite is also the grading instrument, so a test's failure
message is student-facing: write them that way.

Every test of a graded function carries one pytest marker named after its part
(`@pytest.mark.c3`), so students run `pytest -m c3` (decided 9 Oct; the task
text and README say so). A part's tests must need only that part: on the empty
handout, `pytest -m X` fails with `NotImplementedError: Task 1 (X)` and nothing
else. Build test inputs from GTSAM directly (`_exact_measurement`), not from
another graded function. Given-code tests in the graded files are marked
`given`; `tests/test_markers.py` enforces all of this, and `--strict-markers`
catches typos.

Run the handout through the suite after any change to the solution blocks — the
expected result is that only the tests touching given code pass, currently 35
of 95. The assignment is pass/fail (decided 2 Oct 2026), so this is a sanity
check that no graded work hides behind a given-code test, not a scoring floor.

`tests/test_plotting.py` exists because every other test runs with
`save_plots=False`; without it a rename in the plotting stack escapes the suite
and only explodes after a finished multi-minute run. Keep that covered.

## Measured

Victoria Park, 2000 steps, after the block-order fix: 141 landmarks, 1.5 m RMS
from GNSS (GNSS ANIS 2.14, expected 2, 95% interval [1.89, 2.12] with the 1 m
sigma; 1.07 per DOF), landmark ANIS 0.27 per DOF (interval [0.93, 1.07]).

**Metric scales (decided 9 Oct).** The book's scale, (4.67): pose ANEES is the
mean NEES (expected 3), GNSS ANIS the mean NIS (expected 2), each with the
interval chi2inv(0.025/0.975, N d)/N. The landmark NIS has a varying dimension
(2 per associated landmark), so it is averaged per DOF (expected 1, interval
chi2inv(., D)/D) -- (4.67) with mixed dimensions. Every per-step bound and
interval is 95% (`evaluation.CONFIDENCE`); the NEES and GNSS figures used 90%
before. Older numbers in this file (ANEES 0.68, GNSS ANIS 1.07) were quoted per
DOF. The run summary (`graphslam.evaluation.RunSummary`, printed after every
run and by `plot_run`, stored under `"summary"` in metadata.json) prints each
average with its expected value and interval. Figures and summary share one
implementation in `evaluation.py`, and
`test_a_run_stores_its_summary_and_it_matches_the_figures` keeps them together.
Sim baseline (shipped 9 Oct, deterministic): 72 of 78 mapped, 6 missed, 0
duplicates, 3 spurious, RMSE x 0.72 m, y 0.80 m, yaw 1.18 deg, ANEES 8.37
[2.85, 3.15], landmark ANIS 0.33 [0.98, 1.02]. Task 2 quotes 72 of 78, 3
spurious and ANEES 8.37 as the Task 1 check; update it if the baseline moves.
Task 2 now warns that (4.67) assumes independent terms, which a smoother's pose
errors are not, so the ANEES interval is too narrow; the same holds for the
time-clustered GNSS NIS.

Absolute timings vary a lot with machine load: the same pre-fix code took 22 s
in one session and 42 s in another. Compare back to back only. On 1 Oct, back to
back: fixed 35 s of step time (optimization 11.2 / covariance 11.0 /
association 7.3 / local-map extraction 4.7), pre-fix 40 s. Per-step median
17.5 ms, 95th percentile 33 ms.

Full run (6 Oct, quiet machine; the data set has 7247 scan steps, so
`--steps 7300` is simply "all of it"). Back to back with a 20.6 s 2000-step run:

| Tuning | Run time | Landmarks | RMS vs GNSS | GNSS ANIS (1 m, exp. 2) | Landmark ANIS (per DOF) |
|---|---|---|---|---|---|
| `real_default.yaml` | 146 s | 288 | 1.79 m | 2.73 | 0.26 |
| odometry 0.1 m / 0.3° | 291 s | 1282 | 218 m | ~4e4 | 0.48 |

Re-run 9 Oct after the `max_range` / `range_local` split (55 m local map):
identical landmarks, RMS and ANIS for 2000 steps and both full runs, so the
margin changes no association on this data. Run times that day were inflated
by load (39 s / 203 s / 673 s); the table keeps the quiet-machine times.

Steps 5000–7000 cost about 3x the earlier ones (revisits, bigger cliques):
per-step median 13.7 ms, 95th percentile 61 ms, max 88 ms. The lowered-noise
run is the one Task 3 asks for; it diverges completely and takes twice as long.

## Known gaps and open decisions

- **Tuning intent for `real_default.yaml` (decided 2 Oct).** The shipped values
  are a working baseline, deliberately not consistent: roughly consistent with
  GNSS (ANIS 2.14 against an expected 2, just above its narrow 95% interval
  [1.89, 2.12]; 1.5 m RMS) but locally conservative (landmark ANIS 0.27 per
  DOF). Lowering odometry noise pushes landmark ANIS towards 1 but breaks the
  run (every set with odometry translation 0.2 m or less broke in the 9 Oct
  search), because independent-increment odometry cannot be right both per step
  and over a loop. Task 3 asks for one tuning set that lowers the odometry
  noise. **Corrected 9 Oct:** the landmark NIS *can* reach 1 the other way,
  trusting the lidar more and the odometry less: odometry 0.5 m with R 0.1 m /
  0.1 deg gives ANIS 1.05 and 1.49 m at 2000 steps, and 0.98 / 3.94 m / 389
  landmarks over the full run (baseline 0.26 / 1.79 m / 288). A tight R with
  the baseline's 0.3 m odometry breaks (55 m). So the low NIS is mostly a too
  large R, and Q must stay large because wheel errors are correlated. Task 3's
  text still fits; the solving guide has the table.
- **Task 4(b) uses `association.inject_wrong_at_step` (decided 14 Oct).**
  Loosening the gates produces no mistake on either data set (checked up to
  1 - 1e-12), so the text now has students force one: at that scan step,
  `data_association.force_wrong_association` re-pairs the first associated
  measurement with the unused local landmark whose predicted bearing is
  furthest from it (bearing difference wrapped), and `run_slam` prints a line
  saying so, or that there was nothing to re-pair. Students no longer edit
  `slam.py`. Measured 14 Oct through the option: sim with the true noise,
  step 600 -> 13.9 m, 61 spurious; with Huber 0.41 m (no cost on a clean run).
  Shipped sim baseline, step 600 -> 6.82 m, 73 spurious. Victoria Park, step
  1500 -> 4.59 m; Huber k 5 fixes it (1.50 m); Huber k 1.345 / 3 break a clean
  run (25 / 22 m). (An earlier wrapper without the bearing wrap picked a
  different landmark and barely moved the baseline: 1.08 -> 1.11 m.) Do not "fix" the shipped
  tuning: finding this is the task.
- **Pass/fail (decided 2 Oct).** The number of tests passed is not a score;
  it feeds an overall evaluation of the submission together with the report.
  `ga2_graded2.tex` and `ga2_task01_implement.tex` say so. The text talks about
  neither grades nor points anywhere; keep it that way.
- `jointMarginalSupportCliqueCount` is not in GTSAM 4.3, so
  `num_support_cliques` is always zero. The plots that used it are skipped
  unless a patched build fills it in.
- No `--steps` limit is needed for Task 3 (see Measured). Task 3 has a "Plan
  your runs" paragraph: explore with `--steps 2000`, report only full runs
  (the revisits after scan 5000 are where tunings differ). The runtime plot
  was dropped from Task 3 on 9 Oct to keep the assignment no longer than the
  EKF-SLAM one. Its numbers
  (2.5 min, ~140 landmarks at 2000, 20 s) come from the Measured section;
  update both together.
- `Car.a` / `Car.b` (lidar offset) parsed but unused. Backlogged (2 Oct) as a
  possible optional exercise; the system works without it.
- **Task 1 solved from the student side (9 Oct).** Martin solved the handout
  in `ga2_student` (env `ga2_student`), not timed exactly but "some hours".
  The friction he hit became hints in the task text: the `Pose2.Identity()`
  start for (b), the sub-Jacobian out-arguments for (d), `Point2(range, 0)` for
  (g), `supporting_observations` and both off-by-ones for (h). Decided: Task 1
  stays at its current size. His `run_sim` matched the reference step for step
  (same association counts at all 999 steps, 76 of 78, 0.79 m).
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
- **Simulated data's true noise (measured 9 Oct against ground truth):**
  odometry 0.05 m, 0.05 m, 1.0 deg per step; measurements range 0.05 m,
  bearing 0.99 deg; all white and uncorrelated.
- **Tight initial-pose prior on the sim (decided 9 Oct).** The simulated run
  starts exactly at the true pose; the old prior (0.1 m, 0.1 deg) inflated
  every pose covariance and hid the too-small odometry noise, so the baseline's
  ANEES read 2.05 ("somewhat conservative"). `sim_default.yaml` now ships
  0.001 m / 0.001 deg, not marked `TODO tune` (Martin: no more tuning values).
  Victoria Park keeps its loose prior: its start is genuinely unknown in the
  GNSS frame.
- **Resolved: the true Q and R give ANEES 1.91 on the data set, and that is
  chance (9 Oct).** Monte Carlo: 72 fresh noise draws of the true size around
  the same true trajectory, run through the full pipeline with the true noise:
  ANEES mean 3.71 +- 0.34, median 2.89, 5-95% [1.37, 8.42]; 22% of draws fall
  below 1.91. Landmark ANIS 1.000 within +-0.02 on every draw. Dead reckoning
  alone with the true Q: mean 3.02 over 300 draws, 5-95% [1.07, 6.51]. One
  smoothed trajectory's errors are so correlated that its ANEES barely
  averages; the (4.67) interval is far too narrow for it. Task 2 now says one
  run's ANEES can land anywhere from about 1.5 to 8 with the correct noise.
  Scripts were scratch (redraw odometry as true increment composed with
  Expmap(noise), measurements as h(true pose, nearest true landmark) + noise).
- **Best sim tuning found (search over 112 settings, 9 Oct):** true Q and R
  plus M 1, N 1: 78 of 78 mapped, 0 duplicates, 0.36 m / 0.42 deg RMSE, ANIS
  0.99, ANEES 2.42. M 1, N 1 is safe here because S is right in both Q and R.
  Every sensible set lands at 0.35-0.45 m; lower numbers come with duplicates
  or a bad ANEES (luck on one data set). Heading noise matters most; strict
  birth (M 3, N 4) loses 13 landmarks even with the true noise. Gates: alpha
  is a confidence, so raising it towards 1 *loosens* the gate. Loosening (up to
  1 - 1e-12) changes nothing on either data set; tightening (0.999 / 0.99)
  changes nothing on the sim but breaks Victoria Park (248 landmarks, 26 m).
- **The local map is not a tuning value (decided 9 Oct).** `range_local` and
  `bearing_local_deg` lost their `TODO tune`: they come from the sensor (sim
  80 m, +-90 deg; Victoria Park 50 m, about +-87 deg) plus a margin. Smaller
  hides visible landmarks from association and makes duplicates (sim
  `range_local` 60 and 40 hang). On Victoria Park, tree extraction used
  `range_local` as its cut-off, leaving no margin; it now has its own
  `sensor.max_range` (50 m) and `range_local` is 55 m.
  `test_victoria_park_extracts_trees_out_to_max_range_not_range_local` guards
  the split.
- **Sim baseline made deliberately worse (decided 9 Oct).** Martin found the
  old baseline (0.39 m) too close to the best (0.36 m) for students to see an
  improvement. `sim_default.yaml` now has heading noise 0.5 deg (true 1.0) and
  bearing 2.0 deg (true 1.0) on top of the odometry translation 0.01 m (true
  0.05) and range 0.2 m (true 0.05): every noise value wrong in a known
  direction. Fixing the heading alone halves the error (0.53 m), heading and
  bearing gives 0.39 m, the true noise 0.41 m with ANIS 1.00, the best set
  0.36 m. A working baseline is still how a student checks Task 1, so Task 2
  quotes the deterministic output. `test_with_the_true_noise_the_map_has_no_
  duplicate_or_spurious_landmarks` no longer uses the shipped tuning.
- **Dead tuning knobs removed (9 Oct).** `sigma_velocity` and `sigma_steer_deg`
  were marked `TODO tune` in both configs but never used (only `sigma_odom_*`
  feeds the odometry covariance). Removed from `NoiseConfig` and the configs.
  Old run directories whose `config.yaml` still has them no longer load in
  `plot_run`.

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

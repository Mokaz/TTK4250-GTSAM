# Example report (course staff)

A full example of a good GA2 report. All numbers and figures come from real runs
of the reference solution (`assignment/`). The report text is 1292 words by
`texcount`, counting tables and headings, with captions excluded. That is inside
the 1300-word, 7-page limit.

| Set | Config | Run |
|---|---|---|
| S0 | `assignment/configs/sim_default.yaml` | `run_sim` |
| S1, S2, S3 | `configs/sim_true.yaml`, `sim_inflated.yaml`, `sim_strict_birth.yaml` | `run_sim --config ...` |
| V0 | `assignment/configs/real_default.yaml` | `run_real` (full run) |
| V1, V2, V3 | `configs/real_tight_R_loose_gates.yaml`, `real_tight_R.yaml`, `real_low_odom.yaml` | `run_real --config ...` |
| (text only) | `configs/real_low_odom_loose_gates.yaml` | `run_real --config ...` |
| Task 4 | `configs/task4.yaml`, `task4_huber.yaml` | `run_sim --config ...` |

To reproduce, run each set with `--output-dir runs/<sim|real>/<config name>`,
using `baseline` for S0 and V0. Then run `python make_figures.py figures` from
the folder that holds `runs/`. Every figure is built from the handout's own
plotting functions, passing `ax=`, with one exception: the Task 3 map-size figure
overlays the four runs' landmark counts from `steps.npz`, which no stock plot does.

Build:

```sh
latexmk -lualatex -interaction=nonstopmode -outdir=build ga2_example_report.tex
```

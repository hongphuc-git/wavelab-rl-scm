# WaveLab RL SCM

SAC inverse characterization of a free isotropic pipe, fixed outer radius 8 mm.
Targets: h (0.5–5.5 mm), sqrt(E/rho) (1.5–4 km/s), nu (0.15–0.42).

Open notebooks/WaveLab_RL_Colab.ipynb in Google Colab and enable GPU.
Data and weights remain in Google Drive; this repository contains neither.
The initial experiment memorizes 16 fixed training examples: it is a diagnostic,
not evidence of generalization. Physical parameters are initialized randomly.

EfficientNet V2-L encoders adapt user-provided WaveDL epoch50/100 weights.
The backbone is frozen; the original regression head is not used.
Reference: https://github.com/ductho-le/WaveDL (independent adaptation).

SCM runs in Python/SciPy; MATLAB is not required. Safe STOP saves checkpoint and
replay. Abrupt runtime termination loses progress after the latest snapshot.

## Physical-reward pilot (2026-11-14)

[Open in Colab](https://colab.research.google.com/github/hongphuc-git/wavelab-rl-scm/blob/master/notebooks/WaveLab_RL_Colab.ipynb).
Choose `EXPERIMENT`: `physics_blur` (default), `physics_nearest`, `physics_le2024`, or `parameter_progress`.
The three physics presets use `--reward-mode absolute --param-weight 0`; only J contributes to reward apart from the constant -0.01.
`--misfit-mode` explicitly selects blur-NCC, symmetric nearest-branch distance, or the rasterized Le2024 adaptation.
These are forward-model data misfits, not PDE-residual losses; the SCM solver is not differentiated.
Le2024 has not been independently verified as a reproduction of the original paper, and its effectiveness remains an experiment.

All presets use the same 256/32 train/validation split, seed2026, epoch50 frozen encoder, 4 workers, batch8, horizon8, action0.15, and 4000 steps.
Each uses a new `physics_compare_v1` folder; old results remain untouched. Resume rejects changes to reward, parameter weight, misfit or sigmas.
Run notebook cells in order, not Run all (the last cell requests STOP).
Evaluation exports CSV/JSON and scatter, with final, own-best-J, and common blur-best predictions. Truth is withheld from the evaluation environment.
Compare common-selection metrics across branches; do not compare numerical J values from different definitions.
Code and smoke checks do not constitute completed GPU training or improved accuracy.
See [Muse AI handoff](MUSE_AI_HANDOFF.md) for the execution request.

Checks: `python -m unittest discover -s tests -v`

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

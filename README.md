# bayesian-wq-calibration
This repository contains code and data for the manuscript entitled "Calibration and uncertainty quantification of disinfectant decay parameters in water quality models." A preprint of the manuscript is available here: ...

These scripts should be run in the following order to replicate results:
1. `eki-calibration.jl`
2. `gp-emulator.jl`
3. `mcmc-sampling.jl`

Additionally, the script `ccwi-2025.jl` corresponds to the code used for the conference paper "Bayesian inference for quantifying parameter uncertainty in disinfectant decay models" presented at the 21st Computing and Control in the Water Industry (CCWI) Conference.

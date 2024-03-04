[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.23.5-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.23.5/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)
[![Notebook](https://img.shields.io/badge/notebook-7.0.6+-green?logo=jupyter&logoColor=white)](https://pypi.org/project/notebook/7.0.6)
[![torch](https://img.shields.io/badge/torch-2.0.0-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.0.1-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![diffusers](https://img.shields.io/badge/diffusers-0.22.0-green)](https://github.com/huggingface/diffusers/)
[![transformers](https://img.shields.io/badge/transformers-1.35.0-green)](https://github.com/huggingface/transformers/)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

<!-- omit in toc -->
# Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion

###  [Project page](https://HilaManor.github.io/AudioEditing) | [Arxiv](https://arxiv.org/abs/2402.10009) | [Text-Based Space](https://huggingface.co/spaces/hilamanor/audioEditing)

This repository contains the official code release for ***Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion***.

<!-- omit in toc -->
## Table of Contents

- [Requirements](#requirements)
- [Usage Example](#usage-example)
  - [Text-Based Editing](#text-based-editing)
  - [Unsupervised Editing](#unsupervised-editing)
  - [SDEdit](#sdedit)
- [Citation](#citation)

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Usage Example

Supported models are AudioLDM, TANGO, and AudioLDM2. For unsupervised editing, Stable Diffusion is also supported.

### Text-Based Editing

```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_run.py --cfg_tar <target_cfg_strength> --cfg_src <source_cfg_strength> --init_aud <input_audio_path> --target_prompt <description of the wanted edited signal> --tstart <edit from timestep> --model_id <model_name> --results_path <path to dump results>
```

You can supply a source prompt that describes the original audio by using `--source_prompt`.  
Use `python main_run.py --help` for all options.

use `--mode ddim` to run DDIM inversion and editing (note that `--tstart` must be equal to `num_diffusion_steps` (by default set to `200`)).

### Unsupervised Editing

First extract the PCs for your wanted timesteps:

```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_pc_extract_inv.py  --init_aud <input_audio_path> --model_id <model_name> --results_path <path to dump results> --drift_start <start extraction timestep> --drift_end  <end extraction timestep> --n_evs <amount of evs to extract>
```

You can supply a source prompt that describes the original audio by using `--source_prompt`.

Then apply the PCs:

```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_pc_apply_drift.py --extraction_path <path to extracted .pt file> --drift_start <timestep to start apply> --drift_end <timestep to end apply> --amount <edit strength> --evs <ev nums to apply> 

```

By using `--use_specific_ts_pc <timestep num>` you choose a different $t$ from $t'$.  
Add `--combine_evs` to apply all the given PCs together.  
Changing `--evals_pt` to empty will try to get the eigenvalues from the extracted path, and will not work unless the applied timesteps were run in extraction.  

Use `python main_pc_extract_inv.py --help` and `python main_pc_apply_drift.py --help` for all options.

To recreate the random vectors baseline, use `--rand_v`.  Image samples can be recreated using `images_pc_extract_inv.py` and `images_pc_apply_drift.py`.

### SDEdit

SDEdit can be run similarly with:

```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_run_sdedit.py --cfg_tar <target_cfg_strength> --init_aud <input_audio_path> --target_prompt <description of the wanted edited signal> --tstart <edit from timestep> --model_id <model_name> --results_path <path to dump results>
```

Use `python main_run_sdedit.py --help` for all options.

Image samples can be recreated using `images_run_sdedit.py`.

# Citation

If you use this code for your research, please cite our paper:

```
@article{manor2024zeroshot,
    title={Zero-Shot Unsupervised and Text-Based Audio Editing Using {DDPM} Inversion}, 
    author={Manor, Hila and Michaeli, Tomer},
    journal={arXiv preprint arXiv:2402.10009},
    year={2024},
}
```

# Acknowledgements

Parts of this code are heavily based on [DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) and on [Gaussian Denoising Posterior](https://github.com/HilaManor/GaussianDenoisingPosterior).

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

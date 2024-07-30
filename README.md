[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.23.5-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.23.5/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)
[![Notebook](https://img.shields.io/badge/notebook-7.0.6+-green?logo=jupyter&logoColor=white)](https://pypi.org/project/notebook/7.0.6)
[![torch](https://img.shields.io/badge/torch-2.0.0-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.0.1-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![diffusers](https://img.shields.io/badge/diffusers-0.22.0-green)](https://github.com/huggingface/diffusers/)
[![transformers](https://img.shields.io/badge/transformers-1.35.0-green)](https://github.com/huggingface/transformers/)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]
[![CC BY 4.0][cc-by-shield]][cc-by]

<!-- omit in toc -->
# Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion [ICML 2024]

### [Project page](https://HilaManor.github.io/AudioEditing) | [Arxiv](https://arxiv.org/abs/2402.10009) | [Text-Based Space](https://huggingface.co/spaces/hilamanor/audioEditing)

This repository contains the official code release for ***Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion***.

<!-- omit in toc -->
## Table of Contents

- [Requirements](#requirements)
- [Usage Example](#usage-example)
  - [Text-Based Editing](#text-based-editing)
  - [Unsupervised Editing](#unsupervised-editing)
  - [SDEdit](#sdedit)
- [Evaluation](#evaluation)
- [MedleyMDPrompts](#medleymdprompts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

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

- You can supply a source prompt that describes the original audio by using `--source_prompt`.  
- `tstart` is set to `100` by default, which is the configuration used in the user study. The quantitative results in the paper include values ranging from `40` to `100`.

Use `python main_run.py --help` for all options.

use `--mode ddim` to run DDIM inversion and editing (note that for plain DDIM Inversion `--tstart` must be equal to `num_diffusion_steps` (by default set to `200`)).

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

- `tstart` is set to `100` by default, which is the configuration used in the user study. The quantitative results in the paper include values ranging from `40` to `100`.

Use `python main_run_sdedit.py --help` for all options.

Image samples can be recreated using `images_run_sdedit.py`.

## Evaluation

We provide our code used to run LPAPS, CLAP and FAD based evaluations. The code is adapted from multiple repos:

- FAD is from [microsoft/fadtk](https://github.com/microsoft/fadtk).
- LPAPS is adapted from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).
- CLAP is adapted from [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft).

We provide the full code (that works on our directory structure) as an example of use.

## MedleyMDPrompts

The `MedleyMDPrompts` dataset contains manually labeled prompts for the MusicDelta subset of the MedleyDB dataset [Bittner et al. 2014](https://www.researchgate.net/profile/Justin-Salamon/publication/265508421_MedleyDB_A_Multitrack_Dataset_for_Annotation-Intensive_MIR_Research/links/54106cc70cf2f2b29a4109ff/MedleyDB-A-Multitrack-Dataset-for-Annotation-Intensive-MIR-Research.pdf). The MusicDelta subset is comprised of 34 musical excerpts in varying styles and in lengths ranging from 20 seconds to 5 minutes.  
This prompts dataset includes 3-4 source prompts for each signal, and 3-12 editing target prompts for each of the source prompts, totalling 107 source prompts and 696 target prompts.  
In the `captions_targets.csv`, the column `can_be_used_without_source` refers to whether this target prompt was designed to complement a source prompt or not, and therefore should provide enough information to edit a signal on their own. This is just a guideline, you might find that for your application all target prompts are enough on their own.  
The `source_caption_index` column indexes the (ordered) index (starting from 1) of the source prompt for the same signal this target prompt relates to. This data can be used together with `can_be_used_without_source`.

## Citation

If you use this code or the MedleyMDPrompts dataset for your research, please cite our paper:

```
@inproceedings{manor2024zeroshot,
  title = 	 {Zero-Shot Unsupervised and Text-Based Audio Editing Using {DDPM} Inversion},
  author =       {Manor, Hila and Michaeli, Tomer},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {34603--34629},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/manor24a.html},
}
```

## Acknowledgements

Parts of this code are heavily based on [DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) and on [Gaussian Denoising Posterior](https://github.com/HilaManor/GaussianDenoisingPosterior).

AudioLDM2 is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa]. Therefore, using the weights of AudioLDM2 (the default) and code originating in the `code/audioldm` folder is under the same license (eg., `utils.py:load_audio` uses code from `code/audioldm`).  
The rest of the code (inversion, PCs computation) is licensed under an MIT license.

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

The evaluation code adapts code from differently licensed repos:

- FAD is from [microsoft/fadtk](https://github.com/microsoft/fadtk), under MIT License.
- LPAPS is adapted from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), under BSD-2-Clause License.
- CLAP's weights are under CC0-1.0 License, from [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- CLAP's processing code is adapted from [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft), under MIT License.

Our *MedleyMDPrompts* dataset is licensed under CC-BY-4.0 License.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

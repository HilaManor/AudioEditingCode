import pandas as pd
import numpy as np
import torch
import os
import re
import torchaudio
from typing import NamedTuple, Tuple, List, Optional, Union
from tqdm import tqdm

from evals.meta_clap_consistency import CLAPTextConsistencyMetric
from evals.lpaps import LPAPS


class CombinedRes(NamedTuple):
    ours: pd.DataFrame
    sdedit: pd.DataFrame
    orig: pd.DataFrame
    musicgen: pd.DataFrame
    musicgen_large: pd.DataFrame
    ddim: pd.DataFrame


class ScorePair(NamedTuple):
    lpaps: dict
    clap: dict


class Scores(NamedTuple):
    ours: ScorePair
    sdedit: ScorePair
    orig: ScorePair
    musicgen: ScorePair
    ddim: ScorePair


def compute_lpaps_with_windows(aud1: torch.Tensor, aud1_sr: int, aud2: torch.Tensor, aud2_sr: int, model: LPAPS,
                               windows_size1: Optional[int] = None, windows_size2: Optional[int] = None,
                               overlap: float = 0.1, method: str = 'mean', device: str = 'cuda:0') -> float:
    """Calculate the LPAPS score for the given audio files, windowed. If windows_size1 or windows_size2 is None, it will default to 10 seconds.

    :param torch.Tensor aud1: The first audio file to compute LPAPS for
    :param int aud1_sr: The sample rate of the first audio file
    :param torch.Tensor aud2: The second audio file to compute LPAPS for
    :param int aud2_sr: The sample rate of the second audio file
    :param LPAPS model: The LPAPS model to use
    :param Optional[int] windows_size1: Window size in seconds for the first audio file. Defaults to 10 seconds (None)
    :param Optional[int] windows_size2: Window size in seconds for the second audio file. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined LPAPS score
    """

    if windows_size1 is None:
        windows_size1 = int(aud1_sr * 10)
    if windows_size2 is None:
        windows_size2 = int(aud2_sr * 10)

    scores = []
    for i, j in zip(range(0, aud1.shape[-1], int(windows_size1 * (1 - overlap))),
                    range(0, aud2.shape[-1], int(windows_size2 * (1 - overlap)))):
        window1 = aud1[:, i:i + windows_size1]
        window2 = aud2[:, j:j + windows_size2]
        scores.append(model(window1.unsqueeze(0).to(device), window2.unsqueeze(0).to(device),
                      torch.tensor([aud1_sr], device=device),
                      torch.tensor([aud2_sr], device=device)).item())

    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)


def compute_clap_with_windows(aud: torch.Tensor, aud_sr: int, prompt: str, model: CLAPTextConsistencyMetric,
                              windows_size: Optional[int] = None, overlap: float = 0.1,
                              method: str = 'mean', device: torch.device = 'cuda:0') -> float:
    """Calculate the CLAP score for the given audio file and prompt, windowed. If windows_size is None, it will default to 10 seconds. 

    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int aud_sr: The sample rate of the audio file
    :param str prompt: The prompt to compute CLAP relative to
    :param CLAPTextConsistencyMetric model: The CLAP model to use
    :param Optional[int] windows_size: Window size in seconds. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined CLAP score
    """
    if windows_size is None:
        windows_size = int(aud_sr * 10)
    scores = []
    for i in range(0, aud.shape[-1], int(windows_size * (1 - overlap))):
        window = aud[:, i:i + windows_size]
        model.update(window.unsqueeze(0).to(device), [prompt], torch.tensor([aud_sr], device=device))
        scores.append(model.compute())
        model.reset()
    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)


def calc_scores(clap_ckpt_name: str,
                device,
                ours_dirs: Union[str, List[str]],
                sdedit_dirs: Union[str, List[str]],
                musicgen_dir: str,
                ddim_dirs: Union[str, List[str]],
                inputs_orig: str,
                prev_pt: Optional[str] = None,
                win_length: Optional[int] = None,
                overlap: float = 0.1,
                method: str = 'mean',
                clap_checkpoint_path: str = 'clap/pretrained') -> Scores:
    """Calculate the CLAP and LPAPS scores for the given directories
    Note: This function is built for our used directory structure, it's here as an example.

    :param str clap_ckpt_name: the name of the CLAP checkpoint to use
    :param _type_ device: torch device to use
    :param Union[str, List[str]] ours_dirs: our results directories
    :param Union[str, List[str]] sdedit_dirs: sdedit results directories
    :param str musicgen_dir: musicgen results directory
    :param Union[str, List[str]] ddim_dirs: ddim results directories
    :param str inputs_orig: input directory
    :param Optional[str] prev_pt: path to checkpoint of calculated scores, defaults to None
    :param Optional[int] win_length: window length in seconds to use, defaults to None
    :param float overlap: fraction of overlap between windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param str clap_checkpoint_path: the path where CLAP's weights are stored, defaults to 'clap/pretrained'
    :return Scores: 
    """
    clap_model = CLAPTextConsistencyMetric(model_path=os.path.join(clap_checkpoint_path, clap_ckpt_name),
                                           model_arch='HTSAT-base' if 'fusion' not in clap_ckpt_name else 'HTSAT-tiny',
                                           enable_fusion='fusion' in clap_ckpt_name
                                           ).to(device)
    clap_model.eval()

    lpaps_model = LPAPS(net='clap', device=device,
                        net_kwargs={'model_arch': 'HTSAT-base' if 'fusion' not in clap_ckpt_name
                                    else 'HTSAT-tiny',
                                    'chkpt': clap_ckpt_name,
                                    'enable_fusion': 'fusion' in clap_ckpt_name},
                        checkpoint_path=clap_checkpoint_path)

    clap_score_orig = {}
    all_clap_ours = {}
    all_clap_sdedit = {}
    all_clap_musicgen = {}
    all_clap_ddim = {}

    all_lpaps_orig = {}
    all_lpaps_ours = {}
    all_lpaps_sdedit = {}
    all_lpaps_musicgen = {}
    all_lpaps_ddim = {}

    if prev_pt is not None and os.path.exists(prev_pt):
        prev_pt: Scores = torch.load(prev_pt)['scores']
        all_clap_ours = prev_pt.ours.clap
        all_lpaps_ours = prev_pt.ours.lpaps
        all_clap_sdedit = prev_pt.sdedit.clap
        all_lpaps_sdedit = prev_pt.sdedit.lpaps
        clap_score_orig = prev_pt.orig.clap
        all_lpaps_orig = prev_pt.orig.lpaps
        all_clap_musicgen = prev_pt.musicgen.clap
        all_lpaps_musicgen = prev_pt.musicgen.lpaps
        all_clap_ddim = prev_pt.ddim.clap
        all_lpaps_ddim = prev_pt.ddim.lpaps

    if ours_dirs is str:
        ours_dirs = [ours_dirs]
        sdedit_dirs = [sdedit_dirs]
        ddim_dirs = [ddim_dirs]

    audio_inputs = os.listdir(ours_dirs[0])

    with torch.no_grad():
        for audio_input in tqdm(audio_inputs):
            audio_dir_musicgen = os.path.join(musicgen_dir, audio_input)

            # Initialize the dictionaries for the audio input
            for score_dict in [all_clap_ours, clap_score_orig, all_clap_sdedit, all_clap_musicgen, all_clap_ddim,
                               all_lpaps_ours, all_lpaps_orig, all_lpaps_sdedit, all_lpaps_musicgen, all_lpaps_ddim]:
                if audio_input not in score_dict:
                    score_dict[audio_input] = {}

            orig_aud_path = os.path.join(inputs_orig, audio_input + ".wav")
            orig_aud, orig_aud_sr = torchaudio.load(orig_aud_path)

            # LPAPS relative to itself
            if all_lpaps_orig[audio_input] is not float:
                all_lpaps_orig[audio_input] = calc_lpaps_win(lpaps_model, orig_aud, orig_aud, orig_aud_sr, orig_aud_sr,
                                                             win_length, method, overlap, device)

            # Get MusicGen files
            try:
                musicgen_target_prompts = {x[7:-4]: x for x in os.listdir(audio_dir_musicgen)}
            except FileNotFoundError:
                # We didn't run this in musicgen somehow
                print(f'Missing MUSICGEN {audio_input}')
                continue

            for ours_dir, sdedit_dir, ddim_dir in zip(ours_dirs, sdedit_dirs, ddim_dirs):
                audio_dir_ours = os.path.join(ours_dir, audio_input)
                audio_dir_sdedit = os.path.join(sdedit_dir, audio_input)
                audio_dir_ddim = os.path.join(ddim_dir, audio_input)

                # Get SDedit files
                try:
                    sdedit_target_prompts = {x[4:-7].replace('_', ' '): x for x in os.listdir(audio_dir_sdedit)}
                except FileNotFoundError:
                    print(f'Missing SDEDIT {audio_input}')
                    # We didn't get to that input in sdedit yet
                    continue
                # Get DDIM files
                try:
                    {x[4:-7].replace('_', ' '): x for x in os.listdir(audio_dir_ddim)}
                except FileNotFoundError:
                    # We didn't run this in ddim somehow
                    print(f'Missing DDIM {audio_input}')
                    continue

                for src_prompt_dir in os.listdir(audio_dir_ours):
                    src_prompt = src_prompt_dir[4:].replace('_', ' ')
                    # Initialize the dictionaries for the source prompt
                    for score_dict in [all_lpaps_ours, all_lpaps_ddim, all_clap_ours, all_clap_ddim]:
                        if score_dict[audio_input].get(src_prompt) is None:
                            score_dict[audio_input][src_prompt] = {}

                    src_prompt_ddim_dir = os.path.join(audio_dir_ddim, src_prompt_dir)
                    src_prompt_dir = os.path.join(audio_dir_ours, src_prompt_dir)

                    for target_prompt in os.listdir(src_prompt_dir):
                        inner_dir = os.path.join(src_prompt_dir, target_prompt)
                        inner_ddim_dir = os.path.join(src_prompt_ddim_dir, target_prompt)
                        target_prompt = target_prompt[4:-7].replace('_', ' ')

                        if src_prompt == target_prompt:
                            print(f'Skipping {audio_input}, {src_prompt}, {target_prompt}')
                            continue

                        # Calc orig CLAP score for CLAP accuracy
                        if clap_score_orig[audio_input].get(target_prompt) is None:
                            clap_score_orig[audio_input][target_prompt] = calc_clap_win(
                                clap_model, orig_aud, orig_aud_sr, target_prompt, win_length, method, overlap, device)

                        # Initialize the dictionaries for the target prompt
                        for score_dict in [all_clap_ours, all_clap_ddim, all_lpaps_ours, all_lpaps_ddim]:
                            if score_dict[audio_input][src_prompt].get(target_prompt) is None:
                                score_dict[audio_input][src_prompt][target_prompt] = {}

                        # CALC for OURS
                        ours_skips = set([int(re.findall(r'_skip_(\d+)_', x)[0]) for x in os.listdir(inner_dir)
                                          if x.endswith('.wav') and not x.startswith('orig')])
                        for skip in ours_skips:
                            for score_dict in [all_clap_ours, all_lpaps_ours]:
                                if score_dict[audio_input][src_prompt][target_prompt].get(skip) is None:
                                    score_dict[audio_input][src_prompt][target_prompt][skip] = {}

                            ours_tarcfgs = set([int(re.findall(r'_cfg_d_(\d+).0_', x)[0]) for x in os.listdir(inner_dir)
                                                if x.endswith('.wav') and not x.startswith('orig')
                                                and f'_skip_{skip}_' in x])
                            for tarcfg in ours_tarcfgs:
                                for score_dict in [all_clap_ours, all_lpaps_ours]:
                                    if score_dict[audio_input][src_prompt][target_prompt][skip].get(tarcfg) is None:
                                        score_dict[audio_input][src_prompt][target_prompt][skip][tarcfg] = {}

                                ours_auds = {float(re.findall(r'cfg_e_(\d+\.\d+)_', x)[0]): os.path.join(inner_dir, x)
                                             for x in os.listdir(inner_dir)
                                             if x.endswith('.wav') and not x.startswith('orig') and
                                             f'_skip_{skip}_' in x and f'_cfg_d_{tarcfg}.0_' in x}
                                for srccfg in ours_auds:
                                    if all_clap_ours[audio_input][src_prompt][target_prompt][skip][tarcfg].get(srccfg) is None or \
                                            all_lpaps_ours[audio_input][src_prompt][target_prompt][skip][tarcfg].get(srccfg) is None:
                                        ours_aud, ours_aud_sr = torchaudio.load(ours_auds[srccfg])

                                        all_clap_ours[audio_input][src_prompt][target_prompt][skip][tarcfg][srccfg] = calc_clap_win(
                                            clap_model, ours_aud, ours_aud_sr, target_prompt, win_length, method, overlap, device)

                                        all_lpaps_ours[audio_input][src_prompt][target_prompt][skip][tarcfg][srccfg] = calc_lpaps_win(
                                            lpaps_model, ours_aud, orig_aud, ours_aud_sr, orig_aud_sr, win_length, method, overlap, device)

                        # CALC for DDIM
                        try:
                            ddim_skips = [int(re.findall(r'_skip_(\d+)_', x)[0]) for x in os.listdir(inner_ddim_dir) if x.endswith('.wav') and not x.startswith('orig') and 'skip' in x]
                            if len([x for x in os.listdir(inner_ddim_dir) if x.endswith('.wav') and not x.startswith('orig') if '200timesteps' in x]):
                                ddim_skips += [0]
                            ddim_skips = set(ddim_skips)

                            for skip in ddim_skips:
                                for score_dict in [all_clap_ddim, all_lpaps_ddim]:
                                    if score_dict[audio_input][src_prompt][target_prompt].get(skip) is None:
                                        score_dict[audio_input][src_prompt][target_prompt][skip] = {}

                                ddim_tarcfgs = set([int(re.findall(r'_cfg_d_(\d+).0_', x)[0])
                                                    for x in os.listdir(inner_ddim_dir)
                                                    if x.endswith('.wav') and not x.startswith('orig') and
                                                    (('skip' in x and f'_skip_{skip}_' in x) or
                                                     ('200timesteps' in x and skip == 0))])

                                for tarcfg in ddim_tarcfgs:
                                    for score_dict in [all_clap_ddim, all_lpaps_ddim]:
                                        if score_dict[audio_input][src_prompt][target_prompt][skip].get(tarcfg) is None:
                                            score_dict[audio_input][src_prompt][target_prompt][skip][tarcfg] = {}

                                    ddim_auds = {float(re.findall(r'cfg_e_(\d+\.\d+)_', x)[0]): os.path.join(inner_ddim_dir, x)
                                                 for x in os.listdir(inner_ddim_dir)
                                                 if x.endswith('.wav') and not x.startswith('orig')
                                                 and (('skip' in x and f'_skip_{skip}_' in x) or ('200timesteps' in x and skip == 0))
                                                 and f'_cfg_d_{tarcfg}.0_' in x}
                                    for srccfg in ddim_auds:
                                        if all_lpaps_ddim[audio_input][src_prompt][target_prompt][skip][tarcfg].get(srccfg) is None or \
                                                all_clap_ddim[audio_input][src_prompt][target_prompt][skip][tarcfg].get(srccfg) is None:
                                            ddim_aud, ddim_aud_sr = torchaudio.load(ddim_auds[srccfg])

                                            all_clap_ddim[audio_input][src_prompt][target_prompt][skip][tarcfg][srccfg] = calc_clap_win(
                                                clap_model, ddim_aud, ddim_aud_sr, target_prompt, win_length, method, overlap, device)
                                            all_lpaps_ddim[audio_input][src_prompt][target_prompt][skip][tarcfg][srccfg] = calc_lpaps_win(
                                                lpaps_model, ddim_aud, orig_aud, ddim_aud_sr, orig_aud_sr, win_length, method, overlap, device)
                        except (KeyError, FileNotFoundError, IndexError):
                            print(f'Missing DDIM {audio_input}, {src_prompt}, {target_prompt}, {skip}, {tarcfg}, {srccfg}')

                        # CALC for MUSICGEN
                        try:
                            musicgen_tmp = musicgen_target_prompts[target_prompt]
                        except KeyError:
                            # We didn't get to that prompt in musicgen yet
                            print(f'Missing MUSICGEN {audio_input}, {src_prompt}, {target_prompt}')
                            continue

                        for score_dict in [all_clap_musicgen, all_lpaps_musicgen]:
                            if score_dict[audio_input].get(target_prompt) is None:
                                score_dict[audio_input][target_prompt] = []

                        # TODO WHY DOES THIS HAPPEN
                        if not len([x for x in all_clap_musicgen[audio_input][target_prompt]]) or \
                                not len([x for x in all_lpaps_musicgen[audio_input][target_prompt]]):
                            musicgen_aud, musicgen_aud_sr = torchaudio.load(os.path.join(audio_dir_musicgen,
                                                                                         musicgen_tmp))

                            all_clap_musicgen[audio_input][target_prompt].append(
                                {'score': calc_clap_win(clap_model, musicgen_aud, musicgen_aud_sr, target_prompt,
                                                        win_length, method, overlap, device)})
                            all_lpaps_musicgen[audio_input][target_prompt].append(
                                {'score': calc_lpaps_win(lpaps_model, musicgen_aud, orig_aud, musicgen_aud_sr,
                                                         orig_aud_sr, win_length, method, overlap, device)})

                        # CALC for SDEDIT
                        try:
                            sdedit_tmp = sdedit_target_prompts[target_prompt]
                        except KeyError:
                            # We didn't get to that prompt in sdedit yet
                            print(f'Missing SDEDIT {audio_input}, {src_prompt}, {target_prompt}')
                            continue

                        for score_dict in [all_clap_sdedit, all_lpaps_sdedit]:
                            if score_dict[audio_input].get(target_prompt) is None:
                                score_dict[audio_input][target_prompt] = {}

                        sdedit_skips = set([int(re.findall(r'_skip(\d+)', x)[0])
                                            for x in os.listdir(os.path.join(audio_dir_sdedit, sdedit_tmp))
                                            if x.endswith('.wav') and not x.startswith('orig')])
                        for skip in sdedit_skips:
                            for score_dict in [all_clap_sdedit, all_lpaps_sdedit]:
                                if score_dict[audio_input][target_prompt].get(skip) is None:
                                    score_dict[audio_input][target_prompt][skip] = {}

                            sdedits_auds = {int(re.findall(r'cfg(\d+).0', x)[0]): os.path.join(audio_dir_sdedit,
                                                                                               sdedit_tmp, x)
                                            for x in os.listdir(os.path.join(audio_dir_sdedit, sdedit_tmp))
                                            if x.endswith('.wav') and not x.startswith('orig') and f'_skip{skip}_' in x}
                            if not len(sdedits_auds):  # If we didn't print the skip in the filename it's 12
                                sdedits_auds = {12: os.path.join(audio_dir_sdedit, sdedit_tmp,
                                                                 [x for x in os.listdir(os.path.join(audio_dir_sdedit,
                                                                                                     sdedit_tmp))
                                                                  if x.endswith('.wav') and not x.startswith('orig')
                                                                  and f'_skip{skip}' in x][0])}

                            for tarcfg in sdedits_auds:
                                if all_clap_sdedit[audio_input][target_prompt][skip].get(tarcfg) is None or \
                                        all_lpaps_sdedit[audio_input][target_prompt][skip].get(tarcfg) is None:
                                    sdedits_aud, sdedits_aud_sr = torchaudio.load(sdedits_auds[tarcfg])

                                    all_clap_sdedit[audio_input][target_prompt][skip][tarcfg] = calc_clap_win(
                                        clap_model, sdedits_aud, sdedits_aud_sr, target_prompt,
                                        win_length, method, overlap, device)
                                    all_lpaps_sdedit[audio_input][target_prompt][skip][tarcfg] = calc_lpaps_win(
                                        lpaps_model, sdedits_aud, orig_aud, sdedits_aud_sr, orig_aud_sr,
                                        win_length, method, overlap, device)

    return Scores(ours=ScorePair(clap=all_clap_ours, lpaps=all_lpaps_ours),
                  sdedit=ScorePair(clap=all_clap_sdedit, lpaps=all_lpaps_sdedit),
                  orig=ScorePair(clap=clap_score_orig, lpaps=all_lpaps_orig),
                  musicgen=ScorePair(clap=all_clap_musicgen, lpaps=all_lpaps_musicgen),
                  ddim=ScorePair(clap=all_clap_ddim, lpaps=all_lpaps_ddim))


def calc_clap_win(clap_model: CLAPTextConsistencyMetric, aud: torch.Tensor, sr: int, target_prompt: str,
                  win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the CLAP score between an audio file and a prompt, with optional windowing

    :param CLAPTextConsistencyMetric clap_model: An initialized CLAP model to use
    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int sr: The sample rate of the audio file
    :param str target_prompt: The prompt to compute CLAP relative to
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The CLAP score
    """
    if win_length is None:
        clap_model.update(aud.unsqueeze(0).to(device), [target_prompt], torch.tensor([sr], device=device))
        tmp = clap_model.compute()
        clap_model.reset()
        return tmp
    else:
        return compute_clap_with_windows(
            aud, sr, target_prompt, clap_model, device=device,
            windows_size=win_length * sr, overlap=overlap, method=method)


def calc_lpaps_win(lpaps_model: LPAPS, aud1: torch.Tensor, aud2: torch.Tensor, sr1: int, sr2: int,
                   win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the LPAPS score between two audio files, with optional windowing

    :param LPAPS lpaps_model: An initialized LPAPS model to use
    :param torch.Tensor aud1: First audio file
    :param torch.Tensor aud2: Second audio file
    :param int sr1: Sample rate of the first audio file
    :param int sr2: Sample rate of the second audio file
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The LPAPS score
    """
    if win_length is None:
        return lpaps_model(aud1.unsqueeze(0).to(device),
                           aud2.unsqueeze(0).to(device),
                           torch.tensor([sr1], device=device),
                           torch.tensor([sr2], device=device)).item()
    else:
        return compute_lpaps_with_windows(aud1, sr1, aud2, sr2, lpaps_model,
                                          windows_size1=win_length * sr1,
                                          windows_size2=win_length * sr2,
                                          overlap=overlap, method=method, device=device)


def combine_scores(scores: Scores) -> Tuple[CombinedRes, CombinedRes]:
    """Combine the scores into easier-to-handel DataFrames

    :param Scores scores: The scores to combine
    :return Tuple[CombinedRes, CombinedRes]: The combined scores
    """
    all_clap_ours = scores.ours.clap
    all_lpaps_ours = scores.ours.lpaps
    all_clap_sdedit = scores.sdedit.clap
    all_lpaps_sdedit = scores.sdedit.lpaps
    clap_score_orig = scores.orig.clap
    all_lpaps_orig = scores.orig.lpaps
    all_clap_musicgen = scores.musicgen.clap
    all_lpaps_musicgen = scores.musicgen.lpaps
    all_clap_ddim = scores.ddim.clap
    all_lpaps_ddim = scores.ddim.lpaps

    all_lpaps_orig_df = pd.DataFrame.from_dict(all_lpaps_orig, orient='index').reset_index()
    all_lpaps_orig_df = all_lpaps_orig_df.melt(id_vars='index', var_name='target_prompt', value_name='lpaps_score')
    all_lpaps_orig_df = all_lpaps_orig_df[~all_lpaps_orig_df['lpaps_score'].isna()]
    all_lpaps_orig_df.rename(columns={'index': 'input_name'}, inplace=True)
    all_lpaps_orig_df.sort_values(by=['input_name', 'target_prompt'], inplace=True)

    df_all_lpaps_musicgen = pd.DataFrame([(input_name, prompt, item['score'])
                                          for input_name, prompts_dict in all_lpaps_musicgen.items()
                                          for prompt, items in prompts_dict.items()
                                          for item in items],
                                         columns=['input_name', 'target_prompt', 'lpaps_score'])

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, source_prompts in all_lpaps_ours.items():
        for source_prompt, target_prompts in source_prompts.items():
            for target_prompt, skips in target_prompts.items():
                for skip, tarcfgs in skips.items():
                    for tarcfg, srccfgs in tarcfgs.items():
                        for srccfg, lpaps_score in srccfgs.items():
                            flattened_data.append({
                                'input_name': input_name,
                                'source_prompt': source_prompt,
                                'target_prompt': target_prompt,
                                'skip': skip,
                                'tarcfg': tarcfg,
                                'srccfg': srccfg,
                                'lpaps_score': lpaps_score
                            })
    # Create the dataframe
    df_all_lpaps_ours = pd.DataFrame(flattened_data)

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, source_prompts in all_lpaps_ddim.items():
        for source_prompt, target_prompts in source_prompts.items():
            for target_prompt, skips in target_prompts.items():
                for skip, tarcfgs in skips.items():
                    for tarcfg, srccfgs in tarcfgs.items():
                        for srccfg, lpaps_score in srccfgs.items():
                            flattened_data.append({
                                'input_name': input_name,
                                'source_prompt': source_prompt,
                                'target_prompt': target_prompt,
                                'lpaps_score': lpaps_score,
                                'skip': skip,
                                'tarcfg': tarcfg,
                                'srccfg': srccfg
                            })
    # Create the dataframe
    df_all_lpaps_ddim = pd.DataFrame(flattened_data)

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, target_prompts in all_lpaps_sdedit.items():
        for target_prompt, skips in target_prompts.items():
            for skip, tarcfgs in skips.items():
                for tarcfg, lpaps_score in tarcfgs.items():
                    flattened_data.append({
                        'input_name': input_name,
                        'target_prompt': target_prompt,
                        'skip': skip,
                        'tarcfg': tarcfg,
                        'lpaps_score': lpaps_score
                    })

    # Create the dataframe
    df_all_lpaps_sdedit = pd.DataFrame(flattened_data)

    clap_score_orig_df = pd.DataFrame.from_dict(clap_score_orig, orient='index').reset_index()
    clap_score_orig_df = clap_score_orig_df.melt(id_vars='index', var_name='target_prompt', value_name='clap_score')
    clap_score_orig_df = clap_score_orig_df[~clap_score_orig_df['clap_score'].isna()]
    clap_score_orig_df.rename(columns={'index': 'input_name'}, inplace=True)
    clap_score_orig_df.sort_values(by=['input_name', 'target_prompt'], inplace=True)

    df_all_clap_musicgen = pd.DataFrame([(input_name, prompt, item['score'])
                                         for input_name, prompts_dict in all_clap_musicgen.items()
                                         for prompt, items in prompts_dict.items()
                                         for item in items],
                                        columns=['input_name', 'target_prompt', 'clap_score'])

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, source_prompts in all_clap_ours.items():
        for source_prompt, target_prompts in source_prompts.items():
            for target_prompt, skips in target_prompts.items():
                for skip, tarcfgs in skips.items():
                    for tarcfg, srccfgs in tarcfgs.items():
                        for srcfg, clap_score in srccfgs.items():
                            flattened_data.append({
                                'input_name': input_name,
                                'source_prompt': source_prompt,
                                'target_prompt': target_prompt,
                                'skip': skip,
                                'tarcfg': tarcfg,
                                'srccfg': srcfg,
                                'clap_score': clap_score
                            })
    # Create the dataframe
    df_all_clap_ours = pd.DataFrame(flattened_data)

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, source_prompts in all_clap_ddim.items():
        for source_prompt, target_prompts in source_prompts.items():
            for target_prompt, skips in target_prompts.items():
                for skip, tarcfgs in skips.items():
                    for tarcfg, srccfgs in tarcfgs.items():
                        for srccfg, clap_score in srccfgs.items():
                            flattened_data.append({
                                'input_name': input_name,
                                'source_prompt': source_prompt,
                                'target_prompt': target_prompt,
                                'clap_score': clap_score,
                                'skip': skip,
                                'tarcfg': tarcfg,
                                'srccfg': srccfg
                            })
    # Create the dataframe
    df_all_clap_ddim = pd.DataFrame(flattened_data)

    # Flatten the nested dictionary
    flattened_data = []
    for input_name, target_prompts in all_clap_sdedit.items():
        for target_prompt, skips in target_prompts.items():
            for skip, tarcfgs in skips.items():
                for tarcfg, clap_score in tarcfgs.items():
                    flattened_data.append({
                        'input_name': input_name,
                        'target_prompt': target_prompt,
                        'skip': skip,
                        'tarcfg': tarcfg,
                        'clap_score': clap_score
                    })

    # Create the dataframe
    df_all_clap_sdedit = pd.DataFrame(flattened_data)

    return CombinedRes(ours=df_all_clap_ours,
                       sdedit=df_all_clap_sdedit,
                       orig=clap_score_orig_df,
                       musicgen=df_all_clap_musicgen,
                       musicgen_large=None,
                       ddim=df_all_clap_ddim), CombinedRes(ours=df_all_lpaps_ours,
                                                           sdedit=df_all_lpaps_sdedit,
                                                           orig=all_lpaps_orig_df,
                                                           musicgen=df_all_lpaps_musicgen,
                                                           musicgen_large=None,
                                                           ddim=df_all_lpaps_ddim)


import os
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
import random
from typing import Optional, List, Tuple, Dict
try:
    from pc_drift import PromptEmbeddings
except ModuleNotFoundError:
    from .pc_drift import PromptEmbeddings
from models import PipelineWrapper


def load_image(image_path: str, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
               device: Optional[torch.device] = None) -> torch.Tensor:
    if type(image_path) is str:
        from PIL import Image
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, : 3]
    else:
        image = image_path

    import torchvision.transforms as T
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top: h-bottom, left:w-right]
    h, w, c = image.shape
    
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = T.functional.to_tensor(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    # image = torch.from_numpy(image).float() / 127.5 - 1
    # image = image.permute(2, 0, 1).unsqueeze(0).to(device) 

    return image


def load_audio(audio_path: str, fn_STFT, left: int = 0, right: int = 0, device: Optional[torch.device] = None
               ) -> torch.tensor:
    if type(audio_path) is str:
        import audioldm
        import audioldm.audio

        duration = audioldm.utils.get_duration(audio_path)

        mel, _, _ = audioldm.audio.wav_to_fbank(audio_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
        mel = mel.unsqueeze(0)
    else:
        mel = audio_path

    c, h, w = mel.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    mel = mel[:, :, left:w-right]
    mel = mel.unsqueeze(0).to(device)

    return mel


def set_reproducability(seed: int, extreme: bool = True):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Extreme options
        if extreme:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Even more extreme options
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_height_of_spectrogram(length: int, ldm_stable: PipelineWrapper) -> int:
    vocoder_upsample_factor = np.prod(ldm_stable.model.vocoder.config.upsample_rates) / \
        ldm_stable.model.vocoder.config.sampling_rate

    if length is None:
        length = ldm_stable.model.unet.config.sample_size * ldm_stable.model.vae_scale_factor * \
            vocoder_upsample_factor

    height = int(length / vocoder_upsample_factor)

    # original_waveform_length = int(length * ldm_stable.model.vocoder.config.sampling_rate)
    if height % ldm_stable.model.vae_scale_factor != 0:
        height = int(np.ceil(height / ldm_stable.model.vae_scale_factor)) * ldm_stable.model.vae_scale_factor
        print(
            f"Audio length in seconds {length} is increased to {height * vocoder_upsample_factor} "
            f"so that it can be handled by the model. It will be cut to {length} after the "
            f"denoising process."
        )

    return height


def plot_corrs(args, corrs: List[np.array], in_corrs: List[List[torch.Tensor]],
               in_norms: List[List[torch.Tensor]], save_path: str,
               image_name_png: str, logging_dict: Dict[str, any], n_ev: int = 1) -> None:
    # Plot timesteps correlations
    # save_full_path_corrstxt = os.path.join(save_path, image_name_png + "_corrs.txt")
    save_full_path_corrspng = os.path.join(save_path, image_name_png + "_corrs.png")
    corrs_xs = np.arange(args.drift_start-1, args.drift_start-1 - len(corrs), -1)
    # with open(save_full_path_corrstxt, 'w') as f:
    #     f.write('\n'.join([str(x) for x in corrs]))
    for ev_num in range(n_ev):
        ev_corrs = [x[ev_num].detach().cpu().item() for x in corrs]
        plt.plot(corrs_xs, ev_corrs, label='ev ' + str(ev_num + 1))

        corrs_data = [[x, y] for (x, y) in zip(corrs_xs, ev_corrs)]
        corrs_table = wandb.Table(data=corrs_data, columns=["timestep", "correlation"])
        logging_dict[f"pc_correlations_{ev_num + 1}"] = wandb.plot.line(
            corrs_table, "timestep", "correlation", title=f"PCs Correlations #{ev_num + 1}")
    plt.legend()
    plt.savefig(save_full_path_corrspng)
    plt.close()

    incors_timesteps = np.arange(args.drift_start, args.drift_start - len(in_corrs), -1)
    # in_corrs = [[x.detach().cpu().item() for x in incorr] for incorr in in_corrs]
    # in_norms = [[x.detach().cpu().item() for x in in_norm] for in_norm in in_norms]
    if len(in_corrs) > 101:
        in_corrs1 = in_corrs[:len(in_corrs)//2]
        in_corrs2 = in_corrs[len(in_corrs)//2:]
        save_full_path_incorrs1png = os.path.join(save_path, image_name_png + "_incorrs1.png")
        save_full_path_incorrs2png = os.path.join(save_path, image_name_png + "_incorrs2.png")

        plt.figure(figsize=(10, 2*len(in_corrs1)))
        for i, incorr in enumerate(in_corrs1):
            plt.subplot(len(in_corrs1), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i]}")
            plt.legend()
        plt.savefig(save_full_path_incorrs1png)
        plt.close()
        plt.figure(figsize=(10, 2*len(in_corrs2)))
        for i, incorr in enumerate(in_corrs2):
            plt.subplot(len(in_corrs2), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i+len(in_corrs1)]}")
            plt.legend()
        plt.savefig(save_full_path_incorrs2png)
        plt.close()
    else:
        save_full_path_incorrspng = os.path.join(save_path, image_name_png + "_incorrs.png")
        plt.figure(figsize=(10, 2*len(in_corrs)))
        for i, incorr in enumerate(in_corrs):
            plt.subplot(len(in_corrs), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i]}")
            plt.legend()
        plt.savefig(save_full_path_incorrspng)
        plt.close()

    for ev_num in range(n_ev):
        ev_in_corrs = [[x[ev_num].detach().cpu().item() for x in incorr] for incorr in in_corrs]

        logging_dict[f'convergence_{ev_num + 1}'] = wandb.plot.line_series(
            xs=np.arange(args.iters - 1), ys=ev_in_corrs,
            keys=np.arange(args.drift_start, args.drift_start - len(in_corrs), -1),
            title=f"Subspace iterations correlations PC#{ev_num + 1}", xname="iter")
    # logging_dict['norms'] = wandb.plot.line_series(xs=np.arange(args.iters - 1), ys=in_norms,
    #                                                keys=np.arange(args.drift_start,
    #                                                args.drift_start - len(in_norms), -1),
    #                                                title="Subspace iterations norms", xname="iter")


def get_text_embeddings(target_prompt: List[str], target_neg_prompt: List[str], ldm_stable: PipelineWrapper
                        ) -> Tuple[torch.Tensor, PromptEmbeddings, PromptEmbeddings]:
    text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = \
        ldm_stable.encode_text(target_prompt)
    uncond_embedding_hidden_states, uncond_embedding_class_lables, uncond_boolean_prompt_mask = \
        ldm_stable.encode_text(target_neg_prompt)

    text_emb = PromptEmbeddings(embedding_hidden_states=text_embeddings_hidden_states,
                                boolean_prompt_mask=text_embeddings_boolean_prompt_mask,
                                embedding_class_lables=text_embeddings_class_labels)
    uncond_emb = PromptEmbeddings(embedding_hidden_states=uncond_embedding_hidden_states,
                                  boolean_prompt_mask=uncond_boolean_prompt_mask,
                                  embedding_class_lables=uncond_embedding_class_lables)

    return text_embeddings_class_labels, text_emb, uncond_emb

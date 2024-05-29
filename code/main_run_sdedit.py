import argparse
from models import load_model
import os
from torch import inference_mode
import torch
import matplotlib.pyplot as plt
import torchaudio
import wandb
from tqdm import tqdm
from utils import set_reproducability, get_text_embeddings, load_audio
from pc_drift import forward_directional

# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps'],
                        default="cvssp/audioldm-s-full-v2", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_tar", type=float, default=12, help='Classifier-free guidance strength for reverse process')
    parser.add_argument("--num_diffusion_steps", type=int, default=200,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument('--results_path', default='sdedit', help='path to dump results')
    parser.add_argument("--tstart", type=int, default=0,
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true')

    args = parser.parse_args()
    args.eta = 1.
    set_reproducability(args.seed, extreme=False)

    skip = args.num_diffusion_steps - args.tstart
    image_name_png = f's{args.seed}_skip{skip}_cfg{args.cfg_tar}'
    args.image_name_png = image_name_png

    wandb.login()
    wandb_run = wandb.init(project="AudInv", config={},
                           name=args.wandb_name if args.wandb_name is not None else image_name_png,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True),
                           job_type='sdedit')
    wandb.config.update(args)

    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    ldm_stable = load_model(args.model_id, device, args.num_diffusion_steps)
    with torch.no_grad():
        x0 = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device)
    torch.cuda.empty_cache()

    with inference_mode(), torch.no_grad():
        w0 = ldm_stable.vae_encode(x0)

        text_embeddings_class_labels, text_emb, uncond_emb = get_text_embeddings(
            args.target_prompt, args.target_neg_prompt, ldm_stable)

    timesteps = ldm_stable.model.scheduler.timesteps
    latents = []
    for _ in range(len(timesteps) + 1):
        shape = (1, ldm_stable.model.unet.config.in_channels, w0.shape[2],
                 ldm_stable.model.vocoder.config.model_in_dim // ldm_stable.model.vae_scale_factor)
        lat = torch.randn(shape, device=device, dtype=w0.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        lat = lat * ldm_stable.model.scheduler.init_noise_sigma
        latents.append(lat)

    timesteps = timesteps[skip:]
    latents = latents[skip + 1:]

    noise = torch.randn_like(w0, device=device)
    xt = ldm_stable.model.scheduler.add_noise(w0, noise, timesteps[:1].unsqueeze(0))

    del noise, w0

    for it, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        xt, _ = forward_directional(
            ldm_stable, xt, t, latents[it], uncond_emb, text_emb, args.cfg_tar,
            eta=args.eta)

    del latents, uncond_emb, text_emb
    torch.cuda.empty_cache()

    with inference_mode():
        x0_dec = ldm_stable.vae_decode(xt)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)
        orig_audio = ldm_stable.decode_to_mel(x0)

    # same output
    save_path = os.path.join(args.results_path,
                             args.model_id.split('/')[1], os.path.basename(args.init_aud).split('.')[0],
                             'pmt_' + "__".join([x.replace(" ", "_") for x in args.target_prompt]) +
                             "__neg__" + "__".join([x.replace(" ", "_") for x in args.target_neg_prompt]))
    os.makedirs(save_path, exist_ok=True)

    save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
    save_full_path_wave = os.path.join(save_path, image_name_png + ".wav")
    save_full_path_origwave = os.path.join(save_path, "orig.wav")

    plt.imsave(save_full_path_spec, x0_dec[0, 0].T.cpu().detach().numpy())
    torchaudio.save(save_full_path_wave, audio, sample_rate=16000)
    torchaudio.save(save_full_path_origwave, orig_audio, sample_rate=16000)

    logging_dict = {'orig': wandb.Audio(orig_audio.squeeze(), caption='orig', sample_rate=16000),
                    'orig_spec': wandb.Image(x0[0, 0].T.cpu().detach().numpy(), caption='orig'),
                    'gen': wandb.Audio(audio.squeeze(), caption=image_name_png, sample_rate=16000),
                    'gen_spec': wandb.Image(x0_dec[0, 0].T.cpu().detach().numpy(), caption=image_name_png)}
    wandb.log(logging_dict)

    wandb_run.finish()

import argparse
import calendar
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch import inference_mode
import torchaudio
from tqdm import tqdm
import wandb
from ddm_inversion.inversion_utils import inversion_forward_process
from models import load_model
from pc_drift import forward_directional, PCStreamChoice, get_eigenvectors
from utils import plot_corrs, set_reproducability, get_text_embeddings, load_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract PCs for a real audio signal')
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help='Set random seed')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=3, help='Classifier-free guidance strength')
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps'],
                        default="cvssp/audioldm2-music", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--num_diffusion_steps", type=int, default=200,
                        help='Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps'
                             ', while AudioLDM is recommeneded to be used with 100 steps.')
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help='Prompt to accompany the inversion and generation process. Should describe the original audio.')
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help='Negative prompt to accompany the inversion and generation process.')  # ["low quality"])

    parser.add_argument("--corr_to_swap", type=float, default=0.8, help='Correlation threshold to swap eigenvector sign')
    parser.add_argument("--drift_start", type=int, default=None,
                        help='Timestep to start extracting PCs from. If not specified, will use the first timestep.')
    parser.add_argument("--drift_end", type=int, default=None,
                        help='Timestep to end extracting PCs from. If not specified, will use the last timestep.')
    parser.add_argument('--results_path', default='pc_extractions', help='path to dump results')

    parser.add_argument('-c', '--const', type=float, default=1e-3, help='Normalizing const for the power iterations')
    parser.add_argument('--n_evs', type=int, default=1, help='Number of eigenvectors to extract')
    parser.add_argument('-p', '--patch', nargs=2, default=None, type=int,
                        help='Set a specific patch to extract PC for. Format: x1 x2.')
    parser.add_argument('-t', '--iters', type=int, default=50, help='Amount of power iterations')
    parser.add_argument('-d', '--dry', action='store_true',
                        help='Dry run, just unconditional generation without PC extraction (fast)')

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true')

    args = parser.parse_args()
    
    # parser.add_argument('--pc_mode', type=str, choices=['both', 'text', 'uncond'], default='both')
    args.pc_mode = 'both'
    args.eta = 1.
    args.numerical_fix = True
    args.double_precision = False
    args.x_prev_mode = False
    args.test_rand_gen = False
    
    set_reproducability(args.seed)

    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    image_name_png = f's{args.seed}_' + \
        (f'p{args.patch[0]}-{args.patch[1]}_' if args.patch is not None else '') + \
        f'pc-{args.pc_mode}_cfgd{args.cfg_tar}_' + \
        f'drift{args.drift_start}-{args.drift_end}_it{args.iters}_c{args.const:.1e}' + \
        f'{"_dp" if args.double_precision else ""}_{time_stamp_name}'
    #  image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
    #     f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
    #     f'skip_{"-".join([str(x) for x in args.skip.numpy()])}_{time_stamp_name}'
    args.image_name_png = image_name_png

    wandb.login()
    wandb_run = wandb.init(project="AudInv", config={},
                           name=args.wandb_name if args.wandb_name is not None else image_name_png,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True),
                           job_type='pc_extraction_inv')
    wandb.config.update(args)

    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    # Load and set up model
    # ldm_stable, controller = load_model(args.model_id, device, args.num_diffusion_steps, args.double_precision)
    ldm_stable = load_model(args.model_id, device, args.num_diffusion_steps, args.double_precision)
    timesteps = ldm_stable.model.scheduler.timesteps

    if args.drift_start is None:
        args.drift_start = args.num_diffusion_steps
    if args.drift_end is None:
        args.drift_end = -1

    drift_start_it = args.num_diffusion_steps - args.drift_start  # 80 = 200 - 120
    drift_end_it = args.num_diffusion_steps - args.drift_end  # 140 = 200 - 60

    x0 = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device)
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    torch.cuda.empty_cache()
    # 0. Convert audio input length from seconds to spectrogram height
    # height = get_height_of_spectrogram(args.length, ldm_stable)

    # Get all noises now
    # batch_size = 1
    text_embeddings_class_labels, text_emb, uncond_emb = get_text_embeddings(
        args.source_prompt, args.target_neg_prompt, ldm_stable)

    # latents = []
    # for _ in range(len(timesteps) + 1):
    #     latents.append(ldm_stable.model.prepare_latents(
    #         batch_size,
    #         ldm_stable.model.unet.config.in_channels,
    #         height,
    #         text_embeddings_class_labels.dtype,
    #         device,
    #         None, None
    #     ))

    # find Zs and wts - forward process
    _, zs, wts = inversion_forward_process(ldm_stable, w0, etas=args.eta,
                                           prompts=args.source_prompt, cfg_scales=[args.cfg_tar],
                                           prog_bar=True,
                                           num_inference_steps=args.num_diffusion_steps,
                                           # cutoff_points=args.cutoff_points,
                                           numerical_fix=args.numerical_fix,
                                           x_prev_mode=args.x_prev_mode)

    wts = wts.flip(0)
    latents = [wts[0].unsqueeze(0), *[z.unsqueeze(0) for z in zs.flip(0)]]

    del wts, zs

    # Set mask
    mask = torch.zeros_like(latents[0])
    if args.patch is not None:
        mask[:, :, args.patch[0]: args.patch[1], :] = 1
    else:
        mask[:, :, :, :] = 1

    pc_mode = PCStreamChoice.BOTH
    if args.pc_mode == 'text':
        pc_mode = PCStreamChoice.TEXT
    elif args.pc_mode == 'uncond':
        pc_mode = PCStreamChoice.UNCOND

    # masks = torch.ones((batch_size, *xT.shape[1:]), device=model.device, dtype=xT.dtype)
    # cfg_scales_tensor = torch.ones((batch_size, *xT.shape[1:]), device=model.device, dtype=xT.dtype)
    # if batch_size > 1:
    #     if cutoff_points is None:
    #         cutoff_points = [i * 1 / batch_size for i in range(1, batch_size)]
    #     if len(cfg_scales) == 1:
    #         cfg_scales *= batch_size
    #     elif len(cfg_scales) < batch_size:
    #         raise ValueError("Not enough target CFG scales")
    #     cutoff_points = [int(x * cfg_scales_tensor.shape[2]) for x in cutoff_points]
    #     cutoff_points = [0, *cutoff_points, cfg_scales_tensor.shape[2]]
    #     for i, (start, end) in enumerate(zip(cutoff_points[:-1], cutoff_points[1:])):
    #         cfg_scales_tensor[i, :, end:] = 0
    #         cfg_scales_tensor[i, :, :start] = 0
    #         masks[i, :, end:] = 0
    #         masks[i, :, :start] = 0
    #         cfg_scales_tensor[i] *= cfg_scales[i]
    #     cfg_scales_tensor = T.functional.gaussian_blur(cfg_scales_tensor, kernel_size=15, sigma=1)
    #     masks = T.functional.gaussian_blur(masks, kernel_size=15, sigma=1)
    # else:
    #     cfg_scales_tensor *= cfg_scales[0]

    # generate with drift
    xt = latents[0]
    prev_pc = None
    corrs = []
    in_corrs = []
    in_norms = []
    xts = [xt.detach().clone()]
    eigdata = {}

    # save output
    save_path = os.path.join(args.results_path,
                             args.model_id.split('/')[1],
                             os.path.basename(args.init_aud).split('.')[0],
                             'pmt_' + "__".join([x.replace(" ", "_") for x in args.source_prompt]) +
                             "__neg__" + "__".join([x.replace(" ", "_") for x in args.target_neg_prompt]))
    os.makedirs(save_path, exist_ok=True)

    for it, t in tqdm(enumerate(timesteps), total=(len(timesteps))):
        xt_m1, x0_pred = forward_directional(ldm_stable, xt, t, latents[it+1], uncond_emb, text_emb, args.cfg_tar,
                                             eta=args.eta, double_precision=args.double_precision)

        if not args.dry and (it >= drift_start_it and it < drift_end_it):
            eigvecs, eigval, in_corr, in_norm, interm_eigvecs, interm_eigvals = get_eigenvectors(
                ldm_stable,
                xt, text_emb, uncond_emb, latents[it+1], mask,
                t, x0_pred,
                pc_mode, args.const, args.cfg_tar,
                args.iters, args.double_precision, args.eta, args.n_evs)
            if it > drift_start_it:
                corr = (prev_pc.reshape(args.n_evs, -1) @ eigvecs.reshape(args.n_evs, -1).T).diag()
                for ev_num in range(args.n_evs):
                    if corr[ev_num] <= -args.corr_to_swap:
                        eigvecs[ev_num] *= -1
                        print(f'swapped eigvec {ev_num +1}!')
                        # eigval *= -1 # TODO WHat happens to eigval
                        corr[ev_num] *= -1
                corrs.append(corr)
            prev_pc = eigvecs
            in_corrs.append(in_corr)
            in_norms.append(in_norm)
            if not args.wandb_disable:
                for ev_num in range(args.n_evs):
                    ev_in_corr = [inc[ev_num].detach().cpu().item() for inc in in_corr]
                    # ev_in_corr = [inc.detach().cpu().item() for inc in in_corr[ev_num]]
                    table = wandb.Table(data=[[x, y] for (x, y) in zip(np.arange(args.iters - 1), ev_in_corr)],
                                        columns=["iter", "corr"])
                    wandb.log({f'in_corr_{ev_num}': wandb.plot.line(
                        table, 'iter', 'corr', title=f'Current Subspace iterations correlations #PC {ev_num}'),
                               f'eigval_{ev_num}': eigval[ev_num].item()},
                              step=it, commit=ev_num == args.n_evs - 1)

            # Save drift data
            eigdata[t.item()] = {
                'eigvec': eigvecs.detach().cpu(),
                'eigval': eigval.detach().cpu(),
                'interm_eigvecs': {k: v.detach().cpu() for k, v in interm_eigvecs.items()},
                'interm_eigvals': {k: v.detach().cpu() for k, v in interm_eigvals.items()},
                'it': it,
                'ts': args.num_diffusion_steps - it,
                'norm_factor': torch.sqrt(ldm_stable.model.scheduler.alphas_cumprod[t])}
        xt = xt_m1
        xts.append(xt.detach().clone())

        if it % 10 == 0:
            torch.save({'eigdata': eigdata, 'args': args, 'corrs': corrs,
                        'in_corrs': in_corrs, 'latents': latents,
                        'in_norms': in_norms,
                        'xts': xts},
                       os.path.join(save_path, image_name_png + ".pt"))

    torch.save({'eigdata': eigdata, 'args': args, 'corrs': corrs,
                'in_corrs': in_corrs, 'latents': latents,
                'in_norms': in_norms,
                'xts': xts},
               os.path.join(save_path, image_name_png + ".pt"))

    save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
    save_full_path_wave = os.path.join(save_path, image_name_png + ".wav")
    save_full_path_origwave = os.path.join(save_path, "orig.wav")

    # wandb.save(os.path.join(save_path, image_name_png + ".pt")) # Too large :D

    del latents, xts, eigdata, x0_pred, xt_m1, prev_pc
    torch.cuda.empty_cache()

    with inference_mode():
        x0_dec = ldm_stable.vae_decode(xt)
        # x0_dec = ldm_stable.vae.decode(1 / ldm_stable.vae.config.scaling_factor * w0).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)
        orig_audio = ldm_stable.decode_to_mel(x0)

    plt.imsave(save_full_path_spec, x0_dec[0, 0].T.cpu().detach().numpy())
    torchaudio.save(save_full_path_wave, audio, sample_rate=16000)
    torchaudio.save(save_full_path_origwave, orig_audio, sample_rate=16000)

    logging_dict = {'orig': wandb.Audio(orig_audio.squeeze(), caption='orig', sample_rate=16000),
                    'orig_spec': wandb.Image(x0[0, 0].T.cpu().detach().numpy(), caption='orig'),
                    'gen': wandb.Audio(audio.squeeze(), caption=image_name_png, sample_rate=16000),
                    'gen_spec': wandb.Image(x0_dec[0, 0].T.cpu().detach().numpy(), caption=image_name_png)}

    if not args.dry:  # Only log if full run
        print('[+] Logging correlations')
        plot_corrs(args, corrs, in_corrs, in_norms, save_path, image_name_png, logging_dict, args.n_evs)

    wandb.log(logging_dict)
    wandb_run.finish()

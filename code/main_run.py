import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run text-based audio editing.')
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help="GPU device number")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps'],
                        default="cvssp/audioldm2-music", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='Classifier-free guidance strength for forward process')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[12],
                        help='Classifier-free guidance strength for reverse process')
    parser.add_argument("--num_diffusion_steps", type=int, default=200,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""], required=True,
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the forward process. Should describe the original audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument("--tstart", type=int, nargs='+', default=[100],
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs='*', default=None)
    parser.add_argument("--mode", default="ours", choices=['ours', 'ddim'],
                        help="Run our editing or DDIM inversion based editing.")
    parser.add_argument("--fix_alpha", type=float, default=0.1)

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true')

    args = parser.parse_args()
    args.eta = 1.
    args.numerical_fix = True
    args.x_prev_mode = False
    args.test_rand_gen = False

    set_reproducability(args.seed, extreme=False)
    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    model_id = args.model_id
    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar

    # same output
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if args.mode == 'ours':
        image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
            f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
            f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
    else:
        if args.tstart != args.num_diffusion_steps:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
        else:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'{args.num_diffusion_steps}timesteps_{time_stamp_name}'

    wandb.login(key='')
    wandb_run = wandb.init(project="AudInv", entity='', config={},
                           name=args.wandb_name if args.wandb_name is not None else image_name_png,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(args)

    eta = args.eta  # = 1
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1:
            args.tstart *= len(args.target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    ldm_stable = load_model(model_id, device, args.num_diffusion_steps)
    x0 = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device)
    torch.cuda.empty_cache()
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    # find Zs and wts - forward process
    if args.mode == "ddim":
        if len(cfg_scale_src) > 1:
            raise ValueError("DDIM only supports one cfg_scale_src value")
        wT = ddim_inversion(ldm_stable, w0, args.source_prompt, cfg_scale_src[0],
                            num_inference_steps=args.num_diffusion_steps, skip=skip[0])
    else:
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta,
                                                prompts=args.source_prompt, cfg_scales=cfg_scale_src,
                                                prog_bar=True,
                                                num_inference_steps=args.num_diffusion_steps,
                                                cutoff_points=args.cutoff_points,
                                                numerical_fix=args.numerical_fix,
                                                x_prev_mode=args.x_prev_mode)

    # iterate over decoder prompts
    save_path = os.path.join(f'./{args.results_path}/',
                             model_id.split('/')[1],
                             os.path.basename(args.init_aud).split('.')[0],
                             'src_' + "__".join([x.replace(" ", "_") for x in args.source_prompt]),
                             'dec_' + "__".join([x.replace(" ", "_") for x in args.target_prompt]) +
                             "__neg__" + "__".join([x.replace(" ", "_") for x in args.target_neg_prompt]))
    os.makedirs(save_path, exist_ok=True)

    if args.mode == "ours":
        # reverse process (via Zs and wT)
        w0, _ = inversion_reverse_process(ldm_stable,
                                          xT=wts if not args.test_rand_gen else torch.randn_like(wts),
                                          skips=args.num_diffusion_steps - skip,
                                          fix_alpha=args.fix_alpha,
                                          etas=eta, prompts=args.target_prompt,
                                          neg_prompts=args.target_neg_prompt,
                                          cfg_scales=cfg_scale_tar, prog_bar=True,
                                          zs=zs[:int(args.num_diffusion_steps - min(skip))]
                                          if not args.test_rand_gen else torch.randn_like(
                                              zs[:int(args.num_diffusion_steps - min(skip))]),
                                          #   zs=zs[skip:],
                                          cutoff_points=args.cutoff_points)
    else:  # ddim
        if skip != 0:
            warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                          "You are now running partial DDIM inversion.", RuntimeWarning)
        if len(cfg_scale_tar) > 1:
            raise ValueError("DDIM only supports one cfg_scale_tar value")
        if len(args.source_prompt) > 1:
            raise ValueError("DDIM only supports one args.source_prompt value")
        if len(args.target_prompt) > 1:
            raise ValueError("DDIM only supports one args.target_prompt value")
        w0 = text2image_ldm_stable(ldm_stable, args.target_prompt,
                                   args.num_diffusion_steps, cfg_scale_tar[0],
                                   wT,
                                   skip=skip)

    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)
        orig_audio = ldm_stable.decode_to_mel(x0)

    # same output
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if args.mode == 'ours':
        image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
            f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
            f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
    else:
        if skip != 0:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
        else:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'{args.num_diffusion_steps}timesteps_{time_stamp_name}'

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

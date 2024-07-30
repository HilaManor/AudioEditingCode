import argparse
from models import load_model
import os
from pc_drift import forward_directional, PCStreamChoice, apply_drift

from torch import inference_mode
import torch
import torchaudio
import wandb
from tqdm import tqdm
from utils import set_reproducability, get_text_embeddings
import gc
import torchvision.transforms as T
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply extracted PCs to audio")
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help='Set random seed')
    parser.add_argument("--extraction_path", type=str, required=True, help="Path to extraction checkpoint")
    parser.add_argument("--drift_start", type=int, required=True, help="Starting timestep to apply PCs from")
    parser.add_argument("--drift_end", type=int, required=True, help="Ending timestep to apply PCs to")
    parser.add_argument("--amount", type=float, required=True, help="Factor multiplying the PCs (Strength)")
    parser.add_argument("--use_specific_ts_pc", type=int, default=None, help="Use PCs from a specific timestep")
    parser.add_argument("--fix_alpha", type=float, default=None, help="Mask fix")
    parser.add_argument('--fade_length', type=float, default=0., help="Fading mask length")
    parser.add_argument("--evs", type=int, nargs='+', default=[1], help="PCs to apply")
    parser.add_argument('--combine_evs', action='store_true', help="Apply the specified PCs together")
    parser.add_argument('--evals_pt', type=str, default=None, help="Use precomputed eigvalues")

    parser.add_argument('--rand_v', action='store_true')

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true')

    args = parser.parse_args()
    args.shift_x0_for_np = True
    args.sub_iters = None
    
    set_reproducability(args.seed)

    args.extraction_path = args.extraction_path[:-3] if args.extraction_path.endswith('.pt') else args.extraction_path

    run_name = f'drift{args.drift_start}-{args.drift_end}' \
        f'{"_spts" + str(args.use_specific_ts_pc) if args.use_specific_ts_pc is not None else ""}' \
        f'{"_subiters" if args.sub_iters is not None else ""}' \
        f'{"_shiftx0-4np" if args.shift_x0_for_np else ""}' \
        f'{f"fix{args.fix_alpha}" if args.fix_alpha is not None else ""}' \
        f'{"_fade" + str(args.fade_length) if args.fade_length > 0 else ""}' \
        f'{"_avgeval" if args.evals_pt is not None else ""}' \
        f'{"_RAND" if args.rand_v else ""}' \
        f'_a{args.amount}'

    wandb.login(key='')
    wandb_run = wandb.init(project="AudInv", entity='', config={},
                           name=args.wandb_name if args.wandb_name is not None else run_name,
                           notes=args.extraction_path,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True),
                           job_type='images_pc_application')
    wandb.config.update(args)

    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    load_dict = torch.load(args.extraction_path + '.pt', map_location=device)
    extraction_args = load_dict['args']
    eigdata = load_dict['eigdata']

    if args.rand_v:
        for k in eigdata:
            norm = eigdata[k]['eigvec'].norm()
            eigdata[k]['eigvec'] = torch.randn_like(eigdata[k]['eigvec'])
            eigdata[k]['eigvec'] = eigdata[k]['eigvec'] / eigdata[k]['eigvec'].norm() * norm

    latents = load_dict['latents']
    corrs = load_dict['corrs']
    in_corrs = load_dict['in_corrs']
    in_norms = load_dict['in_norms']
    xts = None
    if args.fix_alpha is not None:
        xts = load_dict.get('xts', None)  # handels old checkpoints
    del load_dict

    # Set fade length
    if hasattr(extraction_args, 'length'):
        args.fade_length = int(args.fade_length * latents[0].shape[2] / extraction_args.length)
    else:
        args.fade_length = int(args.fade_length * latents[0].shape[2] / 15)  # TODO

    ldm_stable = load_model(extraction_args.model_id, device,
                            extraction_args.num_diffusion_steps,
                            extraction_args.double_precision)
    timesteps = ldm_stable.model.scheduler.timesteps

    drifts_path = args.extraction_path + '_driftgens'
    os.makedirs(drifts_path, exist_ok=True)

    text_embeddings_class_labels, text_emb, uncond_emb = get_text_embeddings(
        extraction_args.source_prompt, extraction_args.target_neg_prompt, ldm_stable)

    # Set mask
    if args.fix_alpha is not None:
        mask = torch.zeros_like(latents[0], device=device)
        if extraction_args.patch is not None:
            mask[:, :, extraction_args.patch[0]: extraction_args.patch[1], :] = 1
            if args.fade_length > 0:
                mask[:, :, extraction_args.patch[0] - args.fade_length: extraction_args.patch[0], :] = \
                    torch.linspace(0, 1, args.fade_length, device=device)[None, None, :, None]
                mask[:, :, extraction_args.patch[1]: extraction_args.patch[1] + args.fade_length, :] = \
                    torch.linspace(1, 0, args.fade_length, device=device)[None, None, :, None]
        else:
            mask[:, :, :, :] = 1

    pc_mode = PCStreamChoice.BOTH
    if extraction_args.pc_mode == 'text':
        pc_mode = PCStreamChoice.TEXT
    elif extraction_args.pc_mode == 'uncond':
        pc_mode = PCStreamChoice.UNCOND

    drift_start_it = extraction_args.num_diffusion_steps - args.drift_start  # 80 = 200 - 120
    drift_end_it = extraction_args.num_diffusion_steps - args.drift_end  # 140 = 200 - 60

    # generate with drift
    xt = latents[0]
    if args.fix_alpha is not None:
        if xts is not None:
            parallel_xt = xts[0]
        else:
            print('[+] Running parallel xt')
            parallel_xt = latents[0]

    if args.evals_pt is not None:
        args.evals_pt = torch.load(args.evals_pt)

    for it, t in tqdm(enumerate(timesteps), total=(len(timesteps))):
        xt_m1, x0_pred = forward_directional(ldm_stable, xt, t, latents[it+1], uncond_emb, text_emb,
                                             extraction_args.cfg_tar,
                                             eta=extraction_args.eta, double_precision=extraction_args.double_precision)
        if args.fix_alpha is not None:
            if xts is not None:
                parallel_xt = xts[it + 1]
            else:
                parallel_xt, _ = forward_directional(ldm_stable, parallel_xt, t, latents[it+1], uncond_emb, text_emb,
                                                     extraction_args.cfg_tar,
                                                     eta=extraction_args.eta,
                                                     double_precision=extraction_args.double_precision)

        if (it >= drift_start_it and it < drift_end_it):
            if args.combine_evs:
                xt_m1 = apply_drift(ldm_stable,
                                    xt_m1,
                                    x0_pred,
                                    t, timesteps,
                                    extraction_args.num_diffusion_steps,
                                    eigdata, latents[it+1], device,
                                    use_shifted_x0_for_noisepred=args.shift_x0_for_np,
                                    use_specific_ts_pc=args.use_specific_ts_pc,
                                    amount=args.amount, sub_iters=args.sub_iters,
                                    # use_cur_eigval=args.use_cur_eigval,
                                    eta=extraction_args.eta,
                                    ev_nums=args.evs,
                                    evals=args.evals_pt)
            else:
                xt_m1_ev = []
                for ev_idx, ev_num in enumerate(args.evs):
                    xt_m1_ev.append(apply_drift(ldm_stable,
                                                xt_m1[ev_idx].unsqueeze(0) if len(xt_m1) > 1 else xt_m1,
                                                x0_pred[ev_idx].unsqueeze(0) if len(x0_pred) > 1 else xt_m1,
                                                t, timesteps,
                                                extraction_args.num_diffusion_steps,
                                                eigdata, latents[it+1], device,
                                                use_shifted_x0_for_noisepred=args.shift_x0_for_np,
                                                use_specific_ts_pc=args.use_specific_ts_pc,
                                                amount=args.amount, sub_iters=args.sub_iters,
                                                # use_cur_eigval=args.use_cur_eigval,
                                                eta=extraction_args.eta,
                                                ev_nums=[ev_num],
                                                evals=args.evals_pt))
                xt_m1 = torch.concatenate(xt_m1_ev, dim=0).to(device)

            if args.fix_alpha is not None:
                xt_m1 = mask * xt_m1 + (1 - mask) * (args.fix_alpha * parallel_xt + (1 - args.fix_alpha) * xt_m1)
            # del xt_m1_ev
        del x0_pred
        xt = xt_m1

    if args.fix_alpha is not None:
        del parallel_xt, xts, mask
    gc.collect()
    torch.cuda.empty_cache()

    with inference_mode():
        with torch.no_grad():
            x0_dec = []
            for i in range(len(xt)):
                x0_dec.append(ldm_stable.vae_decode(xt[i].unsqueeze(0)))
            x0_dec = torch.cat(x0_dec, dim=0)
        # x0_dec = ldm_stable.vae.decode(1 / ldm_stable.vae.config.scaling_factor * w0).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        x0_dec = (x0_dec.clamp(-1, 1) + 1) / 2
        images = []
        for i in range(len(x0_dec)):
            images.append(T.functional.to_pil_image(x0_dec[i].cpu().detach()))

    if args.combine_evs:
        images[0].save(os.path.join(
            drifts_path,
            f'pcs{"".join([str(x) for x in args.evs])}_'
            f'drift{args.drift_start}-{args.drift_end}'
            f'{"_spts" + str(args.use_specific_ts_pc) if args.use_specific_ts_pc is not None else ""}'
            f'_it{extraction_args.iters if args.sub_iters is None else args.sub_iters}'
            f'_shiftednp{args.shift_x0_for_np}'
            f'{"_fade" + str(args.fade_length) if args.fade_length > 0 else ""}'
            f'{f"_fix{args.fix_alpha}" if args.fix_alpha is not None else ""}'
            f'{"_avgeval" if args.evals_pt is not None else ""}'
            f'{"_RAND" if args.rand_v else ""}'
            # f'{"_diffeigval" if not args.use_cur_eigval else ""}'
            f'_a{args.amount}.png'))

        logging_dict = {'image_gen': wandb.Image(np.array(images[0]),
                                                caption=f'pcs{"".join([str(x) for x in args.evs])}_{run_name}')}
        wandb.log(logging_dict)
    else:
        for ev_idx, ev_num in enumerate(args.evs):
            images[ev_idx].save(os.path.join(
                drifts_path,
                f'pc{ev_num}_'
                f'drift{args.drift_start}-{args.drift_end}'
                f'{"_spts" + str(args.use_specific_ts_pc) if args.use_specific_ts_pc is not None else ""}'
                f'_it{extraction_args.iters if args.sub_iters is None else args.sub_iters}'
                f'_shiftednp{args.shift_x0_for_np}'
                f'{"_fade" + str(args.fade_length) if args.fade_length > 0 else ""}'
                f'{f"_fix{args.fix_alpha}" if args.fix_alpha is not None else ""}'
                # f'{"_diffeigval" if not args.use_cur_eigval else ""}'
                f'{"_avgeval" if args.evals_pt is not None else ""}'
                f'{"_RAND" if args.rand_v else ""}'
                f'_a{args.amount}.png'))

            logging_dict = {'image_gen': wandb.Image(np.array(images[ev_idx]),
                                                     caption=f'pc{ev_num}_' + run_name)}
            wandb.log(logging_dict)
    wandb_run.finish()

import torch
from tqdm import tqdm
from torchvision import transforms as T
from typing import List, Optional, Dict, Union, Tuple
from models import PipelineWrapper


def inversion_forward_process(model: PipelineWrapper,
                              x0: torch.Tensor,
                              etas: Optional[float] = None,
                              prog_bar: bool = False,
                              prompts: List[str] = [""],
                              cfg_scales: List[float] = [3.5],
                              num_inference_steps: int = 50,
                              cutoff_points: Optional[List[float]] = None,
                              numerical_fix: bool = False,
                              extract_h_space: bool = False,
                              extract_skipconns: bool = False,
                              duration: Optional[float] = None,
                              first_order: bool = False) -> Tuple:
    if len(prompts) > 1 and extract_h_space:
        raise NotImplementedError("How do you split cfg_scales for hspace? TODO")

    if len(prompts) > 1 or prompts[0] != "":
        text_embeddings_hidden_states, text_embeddings_class_labels, \
            text_embeddings_boolean_prompt_mask = model.encode_text(prompts)

        # # classifier free guidance
        batch_size = len(prompts)
        cfg_scales_tensor = torch.ones((batch_size, *x0.shape[1:]), device=model.device, dtype=x0.dtype)

        if len(prompts) > 1:
            if cutoff_points is None:
                cutoff_points = [i * 1 / batch_size for i in range(1, batch_size)]
            if len(cfg_scales) == 1:
                cfg_scales *= batch_size
            elif len(cfg_scales) < batch_size:
                raise ValueError("Not enough target CFG scales")

            cutoff_points = [int(x * cfg_scales_tensor.shape[2]) for x in cutoff_points]
            cutoff_points = [0, *cutoff_points, cfg_scales_tensor.shape[2]]

            for i, (start, end) in enumerate(zip(cutoff_points[:-1], cutoff_points[1:])):
                cfg_scales_tensor[i, :, end:] = 0
                cfg_scales_tensor[i, :, :start] = 0
                cfg_scales_tensor[i] *= cfg_scales[i]
                if prompts[i] == "":
                    cfg_scales_tensor[i] = 0
            cfg_scales_tensor = T.functional.gaussian_blur(cfg_scales_tensor, kernel_size=15, sigma=1)
        else:
            cfg_scales_tensor *= cfg_scales[0]

    # In the forward negative prompts are not supported currently (TODO)
    uncond_embeddings_hidden_states, uncond_embeddings_class_lables, uncond_boolean_prompt_mask = model.encode_text(
        [""], negative=True)
    timesteps = model.model.scheduler.timesteps.to(model.device)
    variance_noise_shape = model.get_noise_shape(x0, num_inference_steps)

    if type(etas) in [int, float]:
        etas = [etas]*model.model.scheduler.num_inference_steps
    xts = model.sample_xts_from_x0(x0, num_inference_steps=num_inference_steps)
    zs = torch.zeros(size=variance_noise_shape, device=model.device)
    extra_info = [None] * len(zs)

    hspaces = []
    skipconns = []
    if timesteps[0].dtype == torch.int64:
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    elif timesteps[0].dtype == torch.float32:
        t_to_idx = {float(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps
    model.setup_extra_inputs(xt, init_timestep=timesteps[0], audio_end_in_s=duration)
    for t in op:
        idx = num_inference_steps - t_to_idx[int(t) if timesteps[0].dtype == torch.int64 else float(t)] - 1

        # 1. predict noise residual
        xt = xts[idx+1][None]
        xt_inp = model.model.scheduler.scale_model_input(xt, t)

        with torch.no_grad():
            out, out_hspace, out_skipconns = model.unet_forward(xt_inp, timestep=t,
                                                                encoder_hidden_states=uncond_embeddings_hidden_states,
                                                                class_labels=uncond_embeddings_class_lables,
                                                                encoder_attention_mask=uncond_boolean_prompt_mask)
            if len(prompts) > 1 or prompts[0] != "":
                cond_out, cond_out_hspace, cond_out_skipconns = model.unet_forward(
                    xt_inp.expand(len(prompts), -1, -1, -1)
                    if hasattr(model.model, 'unet') else xt_inp.expand(len(prompts), -1, -1),
                    timestep=t,
                    encoder_hidden_states=text_embeddings_hidden_states,
                    class_labels=text_embeddings_class_labels,
                    encoder_attention_mask=text_embeddings_boolean_prompt_mask)

        if len(prompts) > 1 or prompts[0] != "":
            # # classifier free guidance
            noise_pred = out.sample + \
                (cfg_scales_tensor * (cond_out.sample - (out.sample.expand(batch_size, -1, -1, -1)
                                                         if hasattr(model.model, 'unet')
                                                         else out.sample.expand(batch_size, -1, -1)
                                                         ))
                 ).sum(axis=0).unsqueeze(0)
            if extract_h_space or extract_skipconns:
                noise_h_space = out_hspace + cfg_scales[0] * (cond_out_hspace - out_hspace)
            if extract_skipconns:
                noise_skipconns = {k: [out_skipconns[k][j] + cfg_scales[0] *
                                       (cond_out_skipconns[k][j] - out_skipconns[k][j])
                                       for j in range(len(out_skipconns[k]))]
                                   for k in out_skipconns}
        else:
            noise_pred = out.sample
            if extract_h_space or extract_skipconns:
                noise_h_space = out_hspace
            if extract_skipconns:
                noise_skipconns = out_skipconns
        if extract_h_space or extract_skipconns:
            hspaces.append(noise_h_space)
        if extract_skipconns:
            skipconns.append(noise_skipconns)

        # xtm1 =  xts[idx+1][None]
        xtm1 = xts[idx][None]
        z, xtm1, extra = model.get_zs_from_xts(xt, xtm1, noise_pred, t,
                                               eta=etas[idx], numerical_fix=numerical_fix,
                                               first_order=first_order)
        zs[idx] = z
        # print(f"Fix Xt-1 distance -  NORM:{torch.norm(xts[idx] - xtm1):.4g}, MSE:{((xts[idx] - xtm1)**2).mean():.4g}")
        xts[idx] = xtm1
        extra_info[idx] = extra

    if zs is not None:
        # zs[-1] = torch.zeros_like(zs[-1])
        zs[0] = torch.zeros_like(zs[0])
        # zs_cycle[0] = torch.zeros_like(zs[0])

    if extract_h_space:
        hspaces = torch.concat(hspaces, axis=0)
        return xt, zs, xts, extra_info, hspaces

    if extract_skipconns:
        hspaces = torch.concat(hspaces, axis=0)
        return xt, zs, xts, extra_info, hspaces, skipconns

    return xt, zs, xts, extra_info


def inversion_reverse_process(model: PipelineWrapper,
                              xT: torch.Tensor,
                              tstart: torch.Tensor,
                              fix_alpha: float = 0.1,
                              etas: float = 0,
                              prompts: List[str] = [""],
                              neg_prompts: List[str] = [""],
                              cfg_scales: Optional[List[float]] = None,
                              prog_bar: bool = False,
                              zs: Optional[List[torch.Tensor]] = None,
                              cutoff_points: Optional[List[float]] = None,
                              hspace_add: Optional[torch.Tensor] = None,
                              hspace_replace: Optional[torch.Tensor] = None,
                              skipconns_replace: Optional[Dict[int, torch.Tensor]] = None,
                              zero_out_resconns: Optional[Union[int, List]] = None,
                              extract_h_space: bool = False,
                              extract_skipconns: bool = False,
                              duration: Optional[float] = None,
                              first_order: bool = False,
                              extra_info: Optional[List] = None) -> Tuple:

    batch_size = len(prompts)

    text_embeddings_hidden_states, text_embeddings_class_labels, \
        text_embeddings_boolean_prompt_mask = model.encode_text(prompts)
    uncond_embeddings_hidden_states, uncond_embeddings_class_lables, \
        uncond_boolean_prompt_mask = model.encode_text(neg_prompts, negative=True)
    # text_embeddings = encode_text(model, prompts)
    # uncond_embedding = encode_text(model, [""] * batch_size)

    masks = torch.ones((batch_size, *xT.shape[1:]), device=model.device, dtype=xT.dtype)
    cfg_scales_tensor = torch.ones((batch_size, *xT.shape[1:]), device=model.device, dtype=xT.dtype)

    if batch_size > 1:
        if cutoff_points is None:
            cutoff_points = [i * 1 / batch_size for i in range(1, batch_size)]
        if len(cfg_scales) == 1:
            cfg_scales *= batch_size
        elif len(cfg_scales) < batch_size:
            raise ValueError("Not enough target CFG scales")

        cutoff_points = [int(x * cfg_scales_tensor.shape[2]) for x in cutoff_points]
        cutoff_points = [0, *cutoff_points, cfg_scales_tensor.shape[2]]

        for i, (start, end) in enumerate(zip(cutoff_points[:-1], cutoff_points[1:])):
            cfg_scales_tensor[i, :, end:] = 0
            cfg_scales_tensor[i, :, :start] = 0
            masks[i, :, end:] = 0
            masks[i, :, :start] = 0
            cfg_scales_tensor[i] *= cfg_scales[i]
        cfg_scales_tensor = T.functional.gaussian_blur(cfg_scales_tensor, kernel_size=15, sigma=1)
        masks = T.functional.gaussian_blur(masks, kernel_size=15, sigma=1)
    else:
        cfg_scales_tensor *= cfg_scales[0]

    # xt = xT.expand(1, -1, -1, -1)
    xt = xT[tstart.max()].unsqueeze(0)

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas]*model.model.scheduler.num_inference_steps
    assert len(etas) == model.model.scheduler.num_inference_steps
    timesteps = model.model.scheduler.timesteps.to(model.device)

    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]
    if timesteps[0].dtype == torch.int64:
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    elif timesteps[0].dtype == torch.float32:
        t_to_idx = {float(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    hspaces = []
    skipconns = []
    model.setup_extra_inputs(xt, extra_info=extra_info, init_timestep=timesteps[-zs.shape[0]], audio_end_in_s=duration)

    for it, t in enumerate(op):
        idx = model.model.scheduler.num_inference_steps - t_to_idx[
            int(t) if timesteps[0].dtype == torch.int64 else float(t)] - \
                (model.model.scheduler.num_inference_steps - zs.shape[0] + 1)

        xt_inp = model.model.scheduler.scale_model_input(xt, t)

        # # Unconditional embedding
        with torch.no_grad():
            uncond_out, out_hspace, out_skipconns = model.unet_forward(
                xt_inp, timestep=t,
                encoder_hidden_states=uncond_embeddings_hidden_states,
                class_labels=uncond_embeddings_class_lables,
                encoder_attention_mask=uncond_boolean_prompt_mask,
                mid_block_additional_residual=(None if hspace_add is None else
                                               (1 / (cfg_scales[0] + 1)) *
                                               (hspace_add[-zs.shape[0]:][it] if hspace_add.shape[0] > 1
                                                else hspace_add)),
                replace_h_space=(None if hspace_replace is None else
                                 (hspace_replace[-zs.shape[0]:][it].unsqueeze(0) if hspace_replace.shape[0] > 1
                                  else hspace_replace)),
                zero_out_resconns=zero_out_resconns,
                replace_skip_conns=(None if skipconns_replace is None else
                                    (skipconns_replace[-zs.shape[0]:][it] if len(skipconns_replace) > 1
                                     else skipconns_replace))
                )  # encoder_hidden_states = uncond_embedding)

        # # Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out, cond_out_hspace, cond_out_skipconns = model.unet_forward(
                    xt_inp.expand(batch_size, -1, -1, -1)
                    if hasattr(model.model, 'unet') else xt_inp.expand(batch_size, -1, -1),
                    timestep=t,
                    encoder_hidden_states=text_embeddings_hidden_states,
                    class_labels=text_embeddings_class_labels,
                    encoder_attention_mask=text_embeddings_boolean_prompt_mask,
                    mid_block_additional_residual=(None if hspace_add is None else
                                                   (cfg_scales[0] / (cfg_scales[0] + 1)) *
                                                   (hspace_add[-zs.shape[0]:][it] if hspace_add.shape[0] > 1
                                                    else hspace_add)),
                    replace_h_space=(None if hspace_replace is None else
                                     (hspace_replace[-zs.shape[0]:][it].unsqueeze(0) if hspace_replace.shape[0] > 1
                                      else hspace_replace)),
                    zero_out_resconns=zero_out_resconns,
                    replace_skip_conns=(None if skipconns_replace is None else
                                        (skipconns_replace[-zs.shape[0]:][it] if len(skipconns_replace) > 1
                                         else skipconns_replace))
                    )  # encoder_hidden_states = text_embeddings)

        z = zs[idx] if zs is not None else None
        z = z.unsqueeze(0)
        if prompts:
            # # classifier free guidance
            # noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
            noise_pred = uncond_out.sample + \
                (cfg_scales_tensor * (cond_out.sample - (uncond_out.sample.expand(batch_size, -1, -1, -1)
                                                         if hasattr(model.model, 'unet') else
                                                         uncond_out.sample.expand(batch_size, -1, -1))
                                      )
                 ).sum(axis=0).unsqueeze(0)
            if extract_h_space or extract_skipconns:
                noise_h_space = out_hspace + cfg_scales[0] * (cond_out_hspace - out_hspace)
            if extract_skipconns:
                noise_skipconns = {k: [out_skipconns[k][j] + cfg_scales[0] *
                                       (cond_out_skipconns[k][j] - out_skipconns[k][j])
                                       for j in range(len(out_skipconns[k]))]
                                   for k in out_skipconns}
        else:
            noise_pred = uncond_out.sample
            if extract_h_space or extract_skipconns:
                noise_h_space = out_hspace
            if extract_skipconns:
                noise_skipconns = out_skipconns

        if extract_h_space or extract_skipconns:
            hspaces.append(noise_h_space)
        if extract_skipconns:
            skipconns.append(noise_skipconns)

        # 2. compute less noisy image and set x_t -> x_t-1
        xt = model.reverse_step_with_custom_noise(noise_pred, t, xt, variance_noise=z,
                                                  eta=etas[idx], first_order=first_order)

        # print(f'MSE: {((xt - xT[tstart.max() - it - 1])**2).mean()}')

        # "fix" xt due to mask
        apply_fix = ((tstart.max() - tstart) > it)
        if apply_fix.any():
            apply_fix = (apply_fix * fix_alpha).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(xT.device)
            xt = (masks * (xt.expand(batch_size, -1, -1, -1) * (1 - apply_fix) +
                           apply_fix * (xT[tstart.max() - it - 1].expand(batch_size, -1, -1, -1)
                                        if hasattr(model.model, 'unet') else
                                        xT[tstart.max() - it - 1].expand(batch_size, -1, -1)))
                  ).sum(axis=0).unsqueeze(0)

    if extract_h_space:
        return xt, zs, torch.concat(hspaces, axis=0)

    if extract_skipconns:
        return xt, zs, torch.concat(hspaces, axis=0), skipconns

    return xt, zs

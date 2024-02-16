import torch
from tqdm import tqdm
from torchvision import transforms as T
from typing import List, Optional, Dict, Union
from models import PipelineWrapper


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
        else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 + \
        ((alpha_t**0.5 * (1-alpha_prod_t_prev)) / (1 - alpha_bar)) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50, x_prev_mode=False):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.model.scheduler.alphas
    # betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps + 1,
            model.model.unet.config.in_channels,
            # model.unet.sample_size,
            # model.unet.sample_size)
            x0.shape[-2],
            x0.shape[-1])

    timesteps = model.model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    xts[0] = x0
    x_prev = x0
    for t in reversed(timesteps):
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps-t_to_idx[int(t)]
        if x_prev_mode:
            xts[idx] = x_prev * (alphas[t] ** 0.5) + torch.randn_like(x0) * ((1-alphas[t]) ** 0.5)
            x_prev = xts[idx].clone()
        else:
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    # xts = torch.cat([xts, x0 ],dim = 0)

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 \
    #     else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample, model_output, torch.LongTensor([next_timestep]))
    return next_sample


def inversion_forward_process(model: PipelineWrapper,
                              x0: torch.Tensor,
                              etas: Optional[float] = None,
                              prog_bar: bool = False,
                              prompts: List[str] = [""],
                              cfg_scales: List[float] = [3.5],
                              num_inference_steps: int = 50,
                              eps: Optional[float] = None,
                              cutoff_points: Optional[List[float]] = None,
                              numerical_fix: bool = False,
                              extract_h_space: bool = False,
                              extract_skipconns: bool = False,
                              x_prev_mode: bool = False):
    if len(prompts) > 1 and extract_h_space:
        raise NotImplementedError("How do you split cfg_scales for hspace? TODO")

    if len(prompts) > 1 or prompts[0] != "":
        text_embeddings_hidden_states, text_embeddings_class_labels, \
            text_embeddings_boolean_prompt_mask = model.encode_text(prompts)
        # text_embeddings = encode_text(model, prompt)

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

    uncond_embedding_hidden_states, uncond_embedding_class_lables, uncond_boolean_prompt_mask = model.encode_text([""])
    # uncond_embedding = encode_text(model, "")
    timesteps = model.model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.model.unet.config.in_channels,
        # model.unet.sample_size,
        # model.unet.sample_size)
        x0.shape[-2],
        x0.shape[-1])

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas]*model.model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps, x_prev_mode=x_prev_mode)
        alpha_bar = model.model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    hspaces = []
    skipconns = []
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx+1][None]

        with torch.no_grad():
            out, out_hspace, out_skipconns = model.unet_forward(xt, timestep=t,
                                                                encoder_hidden_states=uncond_embedding_hidden_states,
                                                                class_labels=uncond_embedding_class_lables,
                                                                encoder_attention_mask=uncond_boolean_prompt_mask)
            # out = model.unet.forward(xt, timestep= t, encoder_hidden_states=uncond_embedding)
            if len(prompts) > 1 or prompts[0] != "":
                cond_out, cond_out_hspace, cond_out_skipconns = model.unet_forward(
                    xt.expand(len(prompts), -1, -1, -1), timestep=t,
                    encoder_hidden_states=text_embeddings_hidden_states,
                    class_labels=text_embeddings_class_labels,
                    encoder_attention_mask=text_embeddings_boolean_prompt_mask)
                # cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings)

        if len(prompts) > 1 or prompts[0] != "":
            # # classifier free guidance
            noise_pred = out.sample + \
                (cfg_scales_tensor * (cond_out.sample - out.sample.expand(batch_size, -1, -1, -1))
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

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model.model, noise_pred, t, xt)
        else:
            # xtm1 =  xts[idx+1][None]
            xtm1 = xts[idx][None]
            # pred of x0
            if model.model.scheduler.config.prediction_type == 'epsilon':
                pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
            elif model.model.scheduler.config.prediction_type == 'v_prediction':
                pred_original_sample = (alpha_bar[t] ** 0.5) * xt - ((1 - alpha_bar[t]) ** 0.5) * noise_pred

            # direction to xt
            prev_timestep = t - model.model.scheduler.config.num_train_timesteps // \
                model.model.scheduler.num_inference_steps

            alpha_prod_t_prev = model.get_alpha_prod_t_prev(prev_timestep)
            variance = model.get_variance(t, prev_timestep)

            if model.model.scheduler.config.prediction_type == 'epsilon':
                radom_noise_pred = noise_pred
            elif model.model.scheduler.config.prediction_type == 'v_prediction':
                radom_noise_pred = (alpha_bar[t] ** 0.5) * noise_pred + ((1 - alpha_bar[t]) ** 0.5) * xt

            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * radom_noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)

            zs[idx] = z

            # correction to avoid error accumulation
            if numerical_fix:
                xtm1 = mu_xt + (etas[idx] * variance ** 0.5)*z
            xts[idx] = xtm1

    if zs is not None:
        # zs[-1] = torch.zeros_like(zs[-1])
        zs[0] = torch.zeros_like(zs[0])
        # zs_cycle[0] = torch.zeros_like(zs[0])

    if extract_h_space:
        hspaces = torch.concat(hspaces, axis=0)
        return xt, zs, xts, hspaces

    if extract_skipconns:
        hspaces = torch.concat(hspaces, axis=0)
        return xt, zs, xts, hspaces, skipconns

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.model.scheduler.config.num_train_timesteps // \
        model.model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.get_alpha_prod_t_prev(prev_timestep)
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if model.model.scheduler.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif model.model.scheduler.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = model.get_variance(timestep, prev_timestep)
    # std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    if model.model.scheduler.config.prediction_type == 'epsilon':
        model_output_direction = model_output
    elif model.model.scheduler.config.prediction_type == 'v_prediction':
        model_output_direction = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(model: PipelineWrapper,
                              xT: torch.Tensor,
                              skips: torch.Tensor,
                              fix_alpha: float = 0.1,
                              etas: float = 0,
                              prompts: List[str] = [""],
                              neg_prompts: List[str] = [""],
                              cfg_scales: Optional[List[float]] = None,
                              prog_bar: bool = False,
                              zs: Optional[List[torch.Tensor]] = None,
                            #   controller=None,
                              cutoff_points: Optional[List[float]] = None,
                              hspace_add: Optional[torch.Tensor] = None,
                              hspace_replace: Optional[torch.Tensor] = None,
                              skipconns_replace: Optional[Dict[int, torch.Tensor]] = None,
                              zero_out_resconns: Optional[Union[int, List]] = None,
                              asyrp: bool = False,
                              extract_h_space: bool = False,
                              extract_skipconns: bool = False):

    batch_size = len(prompts)

    text_embeddings_hidden_states, text_embeddings_class_labels, \
        text_embeddings_boolean_prompt_mask = model.encode_text(prompts)
    uncond_embedding_hidden_states, uncond_embedding_class_lables, \
        uncond_boolean_prompt_mask = model.encode_text(neg_prompts)
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

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas]*model.model.scheduler.num_inference_steps
    assert len(etas) == model.model.scheduler.num_inference_steps
    timesteps = model.model.scheduler.timesteps.to(model.device)

    # xt = xT.expand(1, -1, -1, -1)
    xt = xT[skips.max()].unsqueeze(0)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    hspaces = []
    skipconns = []

    for it, t in enumerate(op):
        # idx = t_to_idx[int(t)]
        idx = model.model.scheduler.num_inference_steps - t_to_idx[int(t)] - \
            (model.model.scheduler.num_inference_steps - zs.shape[0] + 1)
        # # Unconditional embedding
        with torch.no_grad():
            uncond_out, out_hspace, out_skipconns = model.unet_forward(
                xt, timestep=t,
                encoder_hidden_states=uncond_embedding_hidden_states,
                class_labels=uncond_embedding_class_lables,
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
                    xt.expand(batch_size, -1, -1, -1),
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
        # print(f'idx: {idx}')
        # print(f't: {t}')
        z = z.unsqueeze(0)
        # z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            # # classifier free guidance
            # noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
            noise_pred = uncond_out.sample + \
                (cfg_scales_tensor * (cond_out.sample - uncond_out.sample.expand(batch_size, -1, -1, -1))
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
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
        # if controller is not None:
            # xt = controller.step_callback(xt)

        # "fix" xt
        apply_fix = ((skips.max() - skips) > it)
        if apply_fix.any():
            apply_fix = (apply_fix * fix_alpha).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(xT.device)
            xt = (masks * (xt.expand(batch_size, -1, -1, -1) * (1 - apply_fix) +
                           apply_fix * xT[skips.max() - it - 1].expand(batch_size, -1, -1, -1))
                  ).sum(axis=0).unsqueeze(0)

    if extract_h_space:
        return xt, zs, torch.concat(hspaces, axis=0)

    if extract_skipconns:
        return xt, zs, torch.concat(hspaces, axis=0), skipconns

    return xt, zs

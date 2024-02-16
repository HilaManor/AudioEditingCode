import torch
try:
    from models import PipelineWrapper
except ModuleNotFoundError:
    from .models import PipelineWrapper
from typing import NamedTuple, Tuple, List, Dict, Optional
from enum import Enum


class PromptEmbeddings(NamedTuple):
    embedding_hidden_states: torch.Tensor
    embedding_class_lables: torch.Tensor
    boolean_prompt_mask: torch.Tensor


class PCStreamChoice(Enum):
    BOTH = 1
    TEXT = 2
    UNCOND = 3


def expand_for_evs(x: torch.Tensor, n_ev: int) -> torch.Tensor:
    if x is None:
        return x
    dev = x.device
    return x.repeat(n_ev, *[1]*(len(x.shape) - 1)).to(dev)


def forward_directional(ldm_stable: PipelineWrapper,
                        xt: torch.Tensor,
                        timestep: torch.Tensor,
                        latent: torch.Tensor,
                        uncond_emb: PromptEmbeddings,
                        text_emb: PromptEmbeddings,
                        cfg_tar: torch.Tensor,
                        eta: float = 1,
                        eigvecs: torch.Tensor = 0,
                        amount: float = 0,
                        double_precision: bool = False,
                        mode: PCStreamChoice = PCStreamChoice.BOTH) -> torch.Tensor:
    with torch.no_grad():
        input = xt + amount * eigvecs * torch.sqrt(ldm_stable.model.scheduler.alphas_cumprod[timestep])
        if double_precision:
            input = input.to(torch.double).to(xt.device)

    # expand embeddings if xt is batched and the embeddings are not expanded
    if len(xt) > 1 and \
        ((uncond_emb.boolean_prompt_mask is not None and len(uncond_emb.boolean_prompt_mask) == 1) or 
         (uncond_emb.embedding_hidden_states is not None and len(uncond_emb.embedding_hidden_states) == 1)):
        n_ev = len(xt)
        uncond_emb = PromptEmbeddings(
            embedding_hidden_states=expand_for_evs(uncond_emb.embedding_hidden_states, n_ev),
            boolean_prompt_mask=expand_for_evs(uncond_emb.boolean_prompt_mask, n_ev),
            embedding_class_lables=expand_for_evs(uncond_emb.embedding_class_lables, n_ev))

        text_emb = PromptEmbeddings(
            embedding_hidden_states=expand_for_evs(text_emb.embedding_hidden_states, n_ev),
            boolean_prompt_mask=expand_for_evs(text_emb.boolean_prompt_mask, n_ev),
            embedding_class_lables=expand_for_evs(text_emb.embedding_class_lables, n_ev))

    # predict the noise residual
    # # Unconditional embedding
    with torch.no_grad():
        uncond_out, _, _ = ldm_stable.unet_forward(
            input if mode == PCStreamChoice.BOTH or mode == PCStreamChoice.UNCOND else xt,
            timestep=timestep,
            encoder_hidden_states=uncond_emb.embedding_hidden_states,
            class_labels=uncond_emb.embedding_class_lables,
            encoder_attention_mask=uncond_emb.boolean_prompt_mask,
        )

    # # Conditional embedding
    with torch.no_grad():
        cond_out, _, _ = ldm_stable.unet_forward(
            input if mode == PCStreamChoice.BOTH or mode == PCStreamChoice.TEXT else xt,
            timestep=timestep,
            encoder_hidden_states=text_emb.embedding_hidden_states,
            class_labels=text_emb.embedding_class_lables,
            encoder_attention_mask=text_emb.boolean_prompt_mask,
        )

    # # classifier free guidance
    noise_pred = uncond_out.sample + cfg_tar * (cond_out.sample - uncond_out.sample)
    # noise_pred = uncond_out.sample + \
    #     (cfg_scales_tensor * (cond_out.sample - uncond_out.sample.expand(batch_size, -1, -1, -1))).sum(
    #                                                            axis=0).unsqueeze(0)

    # compute the previous noisy sample x_t -> x_t-1
    res = ldm_stable.model.scheduler.step(noise_pred, timestep, input, eta=eta, variance_noise=latent)

    xt_1 = res.prev_sample

    return xt_1, res.pred_original_sample


def get_eigenvectors(ldm_stable: PipelineWrapper,
                     xt: torch.Tensor,
                     text_emb: PromptEmbeddings,
                     uncond_emb: PromptEmbeddings,
                     latents: torch.Tensor,
                     mask: torch.Tensor,
                     t: torch.Tensor,
                     x0_pred: torch.Tensor,
                     pc_mode: PCStreamChoice = PCStreamChoice.BOTH,
                     const: float = 1e-3,
                     cfg_tar: float = 3,
                     iters: int = 50,
                     double_precision: bool = False,
                     eta: float = 1,
                     n_ev: int = 1) -> Tuple[torch.Tensor, torch.Tensor,
                                             List[torch.Tensor], List[torch.Tensor],
                                             Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:

    # Initialize a random vector

    if n_ev > 1:
        x0_pred = expand_for_evs(x0_pred, n_ev)
        xt = expand_for_evs(xt, n_ev)

        uncond_emb = PromptEmbeddings(
            embedding_hidden_states=expand_for_evs(uncond_emb.embedding_hidden_states, n_ev),
            boolean_prompt_mask=expand_for_evs(uncond_emb.boolean_prompt_mask, n_ev),
            embedding_class_lables=expand_for_evs(uncond_emb.embedding_class_lables, n_ev))

        text_emb = PromptEmbeddings(
            embedding_hidden_states=expand_for_evs(text_emb.embedding_hidden_states, n_ev),
            boolean_prompt_mask=expand_for_evs(text_emb.boolean_prompt_mask, n_ev),
            embedding_class_lables=expand_for_evs(text_emb.embedding_class_lables, n_ev))

    eigvecs = torch.randn_like(xt) * mask * const

    # Get 1st PC
    prev_ev = eigvecs.detach().clone()
    in_corr = []
    in_norm = []
    interm_eigvecs = {}
    interm_eigvals = {}

    # Apply power iterations
    with torch.no_grad():
        for i in range(iters):
            _, unmaksed_out = forward_directional(ldm_stable, xt, t, latents, uncond_emb, text_emb,
                                                  cfg_tar,
                                                  eta=eta,
                                                  eigvecs=eigvecs, amount=1,
                                                  double_precision=double_precision,
                                                  mode=pc_mode)
            out = unmaksed_out * mask
            Ab = out - x0_pred
            if n_ev > 1:
                if len(xt.shape) == 4:
                    permute_arg = (1, 2, 3, 0)
                elif len(xt.shape) == 3:
                    permute_arg = (1, 2, 0)
                elif len(xt.shape) == 2:
                    permute_arg = (1, 0)

                norm_of_Ab = Ab[:, mask[0].to(torch.bool)].norm(dim=1)
                # Now complete the power iteration:
                eigvecs = (Ab / norm_of_Ab.reshape(n_ev, *[1]*(len(xt.shape) - 1))) * mask

                # Now just make sure the evs are orthonormal
                Q, R = torch.linalg.qr(eigvecs.permute(*permute_arg).reshape(-1, n_ev), mode='reduced')
                swap = torch.prod(torch.linalg.diagonal(R))
                if swap < 0:  # TODO VERIFY
                    Q *= -1

                eigvecs = Q / Q.norm(dim=0)
                eigvecs = eigvecs.T.reshape(Ab.shape)

                # Organize eigvectors by eigenvalue
                _, tmp = (norm_of_Ab / const * (ldm_stable.get_sigma(t)**2)).reshape(n_ev, ).sort(
                    descending=True, stable=True)
                eigvecs = eigvecs[tmp, ...]
            else:
                norm_of_Ab = Ab[mask.to(torch.bool)].norm()
                eigvecs = (Ab / norm_of_Ab) * mask  # This completes the power iteration

            # Compute the correlation between the previous eigenvector and the current one
            if i > 0:
                corr = ((prev_ev.reshape(n_ev, -1)) @
                        (eigvecs.reshape(n_ev, -1).T)).diag()
                # if corr <= -corr_to_swap:
                #     eigvec *= -1
                in_corr.append(corr)
            in_norm.append(norm_of_Ab)
            prev_ev = eigvecs.detach().clone()

            if not (i % 10) and i > 15:
                interm_eigvecs[i] = eigvecs
                interm_eigvals[i] = norm_of_Ab / const * (ldm_stable.get_sigma(t) ** 2)

            eigvecs *= const  # This makes sure that in the next iteration the approximation will still hold

    eigval = (norm_of_Ab / const * (ldm_stable.get_sigma(t) ** 2))
    eigvecs /= const

    return eigvecs, eigval, in_corr, in_norm, interm_eigvecs, interm_eigvals


def apply_drift(ldm_stable: PipelineWrapper,
                xt_m1: torch.Tensor,
                x0_pred: torch.Tensor,
                t: torch.Tensor,
                timesteps: torch.Tensor,
                num_diff_steps: int,
                eigdata: Dict[int, Dict[str, torch.Tensor]],
                latent: torch.Tensor,
                device: torch.device,
                use_shifted_x0_for_noisepred: bool = True,
                use_specific_ts_pc: Optional[int] = None,
                amount: float = 1,
                sub_iters: Optional[int] = None,
                eta: float = 1,
                ev_nums: List[int] = [1],
                evals: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:

    if use_specific_ts_pc is None:
        use_t = t.item()
    else:
        use_t = timesteps[num_diff_steps - use_specific_ts_pc].item()
    eigvec = eigdata[use_t]['eigvec'].to(device)
    # eigval = eigdata[use_t]['eigval'].to(device)

    # if use_cur_eigval:
    if evals is None:
        eigval = eigdata[t.item()]['eigval'].to(device)
    else:
        eigval = torch.from_numpy(evals[t.item()]).to(device)

    if sub_iters is not None:
        eigvec = eigdata[use_t]['interm_eigvecs'][sub_iters].to(device)
        # eigval = eigdata[use_t]['interm_eigvals'][sub_iters].to(device)
        # if use_cur_eigval:
        if evals is not None:
            raise ValueError("evals should be None if sub_iters is not None")
        eigval = eigdata[t.item()]['interm_eigvals'][sub_iters].to(device)

    shift_by = 0
    for ev_num in ev_nums:
        ev_idx = ev_num - 1
        shift_by += amount * (eigval[ev_idx].unsqueeze(0).sqrt() * eigvec[ev_idx].unsqueeze(0))

    x0_pred_drift = x0_pred.clone() + shift_by

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    prev_timestep = t - ldm_stable.model.scheduler.config.num_train_timesteps // \
        ldm_stable.model.scheduler.num_inference_steps
    variance = ldm_stable.model.scheduler._get_variance(t, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    alpha_prod_t_prev = ldm_stable.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
        else ldm_stable.model.scheduler.final_alpha_cumprod
    alpha_prod_t = ldm_stable.model.scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t

    # Get original epsilon prediction
    if eta > 0:
        xt_m1 = xt_m1 - std_dev_t * latent

    # pred_sample_direction = xt_m1 - alpha_prod_t_prev ** (0.5) * x0_pred_drift  # This cancels everything out
    pred_sample_direction = xt_m1 - alpha_prod_t_prev ** (0.5) * x0_pred
    pred_epsilon = pred_sample_direction / ((1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5))

    if use_shifted_x0_for_noisepred:
        pred_epsilon = pred_epsilon - (alpha_prod_t ** (0.5)) / (beta_prod_t ** (0.5)) * shift_by

    # Get drifted xt_m1
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    xt_m1 = alpha_prod_t_prev ** (0.5) * x0_pred_drift + pred_sample_direction

    if eta > 0:
        xt_m1 = xt_m1 + std_dev_t * latent

    return xt_m1

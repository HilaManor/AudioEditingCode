# Code from inbarhub/DDPM_inversoin and from google/prompt-to-prompt

from typing import Union, Optional, List
import torch
import numpy as np
from tqdm import tqdm
from utils import get_text_embeddings


def next_step(ldm_model, model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    # timestep, next_timestep = min(timestep - ldm_model.model.scheduler.config.num_train_timesteps
    timestep, next_timestep = min(timestep - ldm_model.model.scheduler.config.num_train_timesteps
                                  // ldm_model.model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ldm_model.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else ldm_model.model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = ldm_model.model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred(ldm_model, latent, t, text_emb, uncond_emb, cfg_scale):
    noise_pred_uncond, _, _ = ldm_model.unet_forward(
            latent,
            timestep=t,
            encoder_hidden_states=uncond_emb.embedding_hidden_states,
            class_labels=uncond_emb.embedding_class_lables,
            encoder_attention_mask=uncond_emb.boolean_prompt_mask,
        )

    noise_prediction_text, _, _ = ldm_model.unet_forward(
            latent,
            timestep=t,
            encoder_hidden_states=text_emb.embedding_hidden_states,
            class_labels=text_emb.embedding_class_lables,
            encoder_attention_mask=text_emb.boolean_prompt_mask,
        )

    # noise_pred = ldm_model.unet_forward(latents_input, timestep=t, encoder_hidden_states=context).sample  #["sample"]
    # noise_pred_unconldm_d, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond.sample + cfg_scale * (noise_prediction_text.sample - noise_pred_uncond.sample)
    # latents = next_step(model, noise_pred, t, latent)
    return noise_pred


@torch.no_grad()
def ddim_inversion(ldm_model, w0, prompts, cfg_scale, num_inference_steps, skip):
    # uncond_embeddings, cond_embeddings = self.context.chunk(2)
    # all_latent = [latent]

    _, text_emb, uncond_emb = get_text_embeddings(prompts, [""], ldm_model)

    latent = w0.clone().detach()
    for i in tqdm(range(num_inference_steps)):
        if num_inference_steps - i <= skip:
            break
        t = ldm_model.model.scheduler.timesteps[len(ldm_model.model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred(ldm_model, latent, t, text_emb, uncond_emb, cfg_scale)
        latent = next_step(ldm_model, noise_pred, t, latent)
        # all_latent.append(latent)
    return latent


@torch.no_grad()
def text2image_ldm_stable(ldm_model, prompt: List[str], num_inference_steps: int = 50,
                          guidance_scale: float = 7.5, xt: Optional[torch.FloatTensor] = None, skip: int = 0):
    _, text_emb, uncond_emb = get_text_embeddings(prompt, [""], ldm_model)

    for t in tqdm(ldm_model.model.scheduler.timesteps[skip:]):
        noise_pred_uncond, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_emb.embedding_hidden_states,
                class_labels=uncond_emb.embedding_class_lables,
                encoder_attention_mask=uncond_emb.boolean_prompt_mask,
            )

        noise_prediction_text, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=text_emb.embedding_hidden_states,
                class_labels=text_emb.embedding_class_lables,
                encoder_attention_mask=text_emb.boolean_prompt_mask,
            )

        noise_pred = noise_pred_uncond.sample + guidance_scale * (noise_prediction_text.sample - noise_pred_uncond.sample)
        xt = ldm_model.model.scheduler.step(noise_pred, t, xt, eta=0).prev_sample

    return xt

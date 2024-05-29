"""Code based on https://github.com/facebookresearch/audiocraft/"""

from collections import namedtuple
import torch
import laion_clap
from laion_clap.training import data
import os
import julius
# import torch.nn.functional as F


class CLAP_base(torch.nn.Module):
    def __init__(self, checkpoint_path='clap/pretrained', requires_grad=False, model_arch='HTSAT-base',
                 chkpt='music_speech_epoch_15_esc_89.25.pt',
                 enable_fusion=False, device='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enable_fusion = enable_fusion
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch, device=device)
        self.model_sample_rate = 48_000
        # load_clap_state_dict(self.model, model_path)
        from clap_module.factory import load_state_dict  # type: ignore
        pkg = load_state_dict(os.path.join(checkpoint_path, chkpt))
        pkg.pop('text_branch.embeddings.position_ids', None)
        self.model.model.load_state_dict(pkg)
        self.model.eval()

        self.requires_grad = requires_grad
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_num_layers(self):
        return 4

    def forward(self, audio: torch.Tensor, sample_rates: torch.Tensor):
        assert torch.all(sample_rates == sample_rates[0].item()), "All items in batch should have the same sample rate"
        sample_rate = int(sample_rates[0].item())
        # convert audio batch to 48kHz monophonic audio with no channel dimension: [B, C, T] -> [B, T]
        audio = CLAP_base._convert_audio(audio, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1
                                         ).mean(dim=1).squeeze(0)

        temp_dict = {}
        temp_dict = data.get_audio_features(
            temp_dict, audio, 480000,
            data_truncating='fusion' if self.model.enable_fusion else 'rand_trunc',
            data_filling='repeatpad',
            audio_cfg=self.model.model_cfg['audio_cfg'],
            require_grad=audio.requires_grad and self.requires_grad
        )

        device = next(self.model.model.parameters()).device
        keys = temp_dict.keys()
        for k in keys:
            temp_dict[k] = temp_dict[k].unsqueeze(0).to(device)

        if self.model.model.audio_branch.enable_fusion and temp_dict["longer"].sum() == 0:
            x = temp_dict["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.model.model.audio_branch.bn0(x)
            x = x.transpose(1, 3)
            x = self.model.model.audio_branch.reshape_wav2img(x)
            output_dict = self._forward_features(x, longer_idx=[])
            return output_dict

        if not self.model.model.audio_branch.enable_fusion:
            x = temp_dict["waveform"].to(device=device, non_blocking=True)
            x = self.model.model.audio_branch.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.model.model.audio_branch.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
            x = self.model.model.audio_branch.bn0(x)
            x = x.transpose(1, 3)

            x = self.model.model.audio_branch.reshape_wav2img(x)
            output_dict = self._forward_features(x)
        else:
            longer_list = temp_dict["longer"].to(device=device, non_blocking=True)
            x = temp_dict["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.model.model.audio_branch.bn0(x)
            x = x.transpose(1, 3)
            longer_list_idx = torch.where(longer_list)[0]
            if self.model.model.audio_branch.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
                new_x = x[:, 0:1, :, :].clone().contiguous()
                if len(longer_list_idx) > 0:
                    # local processing
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1)).contiguous()
                    fusion_x_local = self.model.model.audio_branch.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(FB, FC, FF, fusion_x_local.size(-1))
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1, 3)).contiguous().flatten(2)
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat([fusion_x_local, torch.zeros((FB, FF, FT - fusion_x_local.size(-1)),
                                                                                device=device)], dim=-1)
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.model.model.audio_branch.fusion_model(new_x[longer_list_idx],
                                                                                        fusion_x_local)
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                else:
                    x = new_x

            elif self.model.model.audio_branch.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d', 'channel_map']:
                x = x  # no change

            x = self.model.model.audio_branch.reshape_wav2img(x)
            output_dict = self._forward_features(x, longer_idx=longer_list_idx)

        return output_dict

    def _forward_features(self, x, longer_idx=None):
        # A deprecated optimization for using a hierarchical output from different blocks

        x = self.model.model.audio_branch.patch_embed(x, longer_idx=longer_idx)
        if self.model.model.audio_branch.ape:
            x = x + self.model.model.audio_branch.absolute_pos_embed
        x = self.model.model.audio_branch.pos_drop(x)
        layer_outs = []
        for i, layer in enumerate(self.model.model.audio_branch.layers):
            x, attn = layer(x)
            layer_outs.append(x)

        clap_outputs = namedtuple("ClapOutputs", ['swin1', 'swin2', 'swin3', 'swin4'])
        return clap_outputs(layer_outs[0], layer_outs[1], layer_outs[2], layer_outs[3])

    @staticmethod
    def _convert_audio(wav: torch.Tensor, from_rate: float,
                       to_rate: float, to_channels: int) -> torch.Tensor:
        """Convert audio to new sample rate and number of audio channels."""
        wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
        wav = CLAP_base._convert_audio_channels(wav, to_channels)
        return wav

    @staticmethod
    def _convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
        """Convert audio to the given number of channels.

        Args:
            wav (torch.Tensor): Audio wave of shape [B, C, T].
            channels (int): Expected number of channels as output.
        Returns:
            torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
        """
        *shape, src_channels, length = wav.shape
        if src_channels == channels:
            pass
        elif channels == 1:
            # Case 1:
            # The caller asked 1-channel audio, and the stream has multiple
            # channels, downmix all channels.
            wav = wav.mean(dim=-2, keepdim=True)
        elif src_channels == 1:
            # Case 2:
            # The caller asked for multiple channels, but the input file has
            # a single channel, replicate the audio over all channels.
            wav = wav.expand(*shape, channels, length)
        elif src_channels >= channels:
            # Case 3:
            # The caller asked for multiple channels, and the input file has
            # more channels than requested. In that case return the first channels.
            wav = wav[..., :channels, :]
        else:
            # Case 4: What is a reasonable choice here?
            raise ValueError('The audio file has less channels than requested but is not mono.')
        return wav

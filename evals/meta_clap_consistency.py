# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from pathlib import Path
import typing as tp
import torchmetrics
from transformers import RobertaTokenizer  # type: ignore
import julius 


# Code from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py
def load_clap_state_dict(clap_model, path: tp.Union[str, Path]):
    """Wrapper around state dict loading of CLAP model
    addressing compatibility issues between CLAP and AudioCraft
    HuggingFace transformer version.
    See: https://github.com/LAION-AI/CLAP/issues/118
    """
    from clap_module.factory import load_state_dict  # type: ignore
    pkg = load_state_dict(path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    clap_model.model.load_state_dict(pkg)


# Code from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/data/audio_utils.py
def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
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

# Code from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/data/audio_utils.py
def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav

try:
    import laion_clap  # type: ignore
except ImportError:
    laion_clap = None


# Code from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/metrics/clap_consistency.py
class TextConsistencyMetric(torchmetrics.Metric):
    """Text consistency metric measuring consistency between audio and text pairs."""

    def update(self, audio: torch.Tensor, text: tp.List[str], sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        raise NotImplementedError("implement how to update the metric from the audio and text pairs.")

    def compute(self):
        raise NotImplementedError("implement how to compute the final metric score.")


# Code from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/metrics/clap_consistency.py
class CLAPTextConsistencyMetric(TextConsistencyMetric):
    """Text consistency metric relying on Contrastive Language-Audio Pretraining (CLAP).

    This metric is similar to the MuLan Cycle Consistency from MusicLM (https://arxiv.org/pdf/2301.11325.pdf)
    or the CLAP score used in Make-An-Audio (https://arxiv.org/pdf/2301.12661v1.pdf).

    As a joint audio-text embedding model, a pretrained CLAP model can be used to quantify the
    similarity between audio-text pairs. We compute the CLAP embeddings from the text descriptions as
    well as the generated audio based on them, and define the MCC metric as the average cosine similarity
    between these embeddings.

    Model implementation & pre-trained checkpoints: https://github.com/LAION-AI/CLAP
    """
    def __init__(self, model_path: tp.Union[str, Path], model_arch: str = 'HTSAT-tiny', enable_fusion: bool = False):
        super().__init__()
        if laion_clap is None:
            raise ImportError("Please install CLAP to compute text consistency: 'pip install laion_clap'")
        self.add_state("cosine_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")
        self._initialize_model(model_path, model_arch, enable_fusion)

 
    def _initialize_model(self, model_path: tp.Union[str, Path], model_arch: str, enable_fusion: bool):
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch)
        self.model_sample_rate = 48_000
        load_clap_state_dict(self.model, model_path)
        self.model.eval()

    def _tokenizer(self, texts: tp.Union[str, tp.List[str]]) -> dict:
        # we use the default params from CLAP module here as well
        return self.tokenize(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

    def update(self, audio: torch.Tensor, text: tp.List[str], sample_rates: torch.Tensor) -> None:
        """Compute cosine similarity between audio and text pairs and accumulate scores over the dataset."""
        assert audio.size(0) == len(text), "Number of audio and text samples should match"
        assert torch.all(sample_rates == sample_rates[0].item()), "All items in batch should have the same sample rate"
        sample_rate = int(sample_rates[0].item())
        # convert audio batch to 48kHz monophonic audio with no channel dimension: [B, C, T] -> [B, T]
        audio = convert_audio(audio, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1).mean(dim=1)
        audio_embeddings = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        text_embeddings = self.model.get_text_embedding(text, tokenizer=self._tokenizer, use_tensor=True)
        # cosine similarity between the text and the audio embedding
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings, dim=1, eps=1e-8)
        self.cosine_sum += cosine_sim.sum(dim=0)
        self.weight += torch.tensor(cosine_sim.size(0))

    def compute(self):
        """Computes the average cosine similarty across all audio/text pairs."""
        assert self.weight.item() > 0, "Unable to compute with total number of comparisons <= 0"  # type: ignore
        return (self.cosine_sum / self.weight).item()  # type: ignore
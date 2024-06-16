from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel

from streaming_avatar.models.whisper import Whisper
from .positional_encoding import PositionalEncoding


@dataclass
class MuseTalkConfig:
    vae_model_name_or_path: str = "stabilityai/sd-vae-ft-mse"

    unet_model_name_or_path: str = "stabilityai/sd-vae-ft-mse"

    whisper_model_name_or_path: str = "stabilityai/sd-vae-ft-mse"


class MuseTalk(nn.Module):
    def __init__(self, config: MuseTalkConfig):
        super().__init__()

        self.config = config

        self.vae = AutoencoderKL.from_pretrained(
            self.config.vae_model_name_or_path
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.unet_model_name_or_path
        )

        self.pe = PositionalEncoding(d_model=384)

        self.whisper = Whisper.from_pretrained(
            self.config.whisper_model_name_or_path
        )

    def preprocess_avatar(self):
        from .preprocess import preprocess_avatar

    def inpaint(self):
        ...

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .audio_utils import compute_log_mel_spectrogram


@dataclass
class WhisperConfig:
    n_mels: int = 80

    n_audio_ctx: int = 1500

    n_audio_state: int = 384

    n_audio_head: int = 6

    n_audio_layer: int = 4

    n_vocab: int = 51865

    n_text_ctx: int = 448

    n_text_state: int = 384

    n_text_head: int = 6

    n_text_layer: int = 4


@dataclass
class Segment:
    start: int

    end: int

    embeddings: torch.Tensor


@dataclass
class EncoderOutput:
    segments: list[Segment]


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: dict[nn.Module, torch.Tensor] | None = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache.get(self.key, self.key(xa))
            v = kv_cache.get(self.value, self.value(xa))

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: dict[nn.Module, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.positional_embedding: torch.Tensor

        self.blocks: list[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(
        self, x: torch.Tensor, include_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        include_embeddings: bool
            whether to include intermediate steps in the output
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        if include_embeddings:
            embeddings = [x.detach()]

        for block in self.blocks:
            x = block(x)
            if include_embeddings:
                embeddings.append(x.detach())

        x = self.ln_post(x)

        if include_embeddings:
            embeddings = torch.stack(embeddings, dim=1)
            return x, embeddings
        else:
            return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: list[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.mask: torch.Tensor

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor,
        kv_cache: dict[nn.Module, torch.Tensor] | None = None,
        include_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        include_embeddings : bool
            Whether to include intermediate values in the output to this function
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        if include_embeddings:
            embeddings = [x.detach()]

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            if include_embeddings:
                embeddings.append(x.detach())

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        if include_embeddings:
            embeddings = torch.stack(embeddings, dim=1)
            return logits, embeddings
        else:
            return logits


class Whisper(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="whisper",
    repo_url="https://github.com/vivym/streaming-avatar",
    docs_url="https://github.com/vivym/streaming-avatar",
):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        self.config = config

        self.encoder = AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
        )

        self.decoder = TextDecoder(
            self.config.n_vocab,
            self.config.n_text_ctx,
            self.config.n_text_state,
            self.config.n_text_head,
            self.config.n_text_layer,
        )

    def encode(self, audio: torch.Tensor) -> EncoderOutput:
        """
        audio : torch.Tensor, shape = (batch_size, T)
        """

        mel = compute_log_mel_spectrogram(audio, n_mels=self.config.n_mels)

        segments = []
        num_frames = mel.shape[-1]
        for start_idx in range(0, num_frames, 3000):
            end_idx = min(start_idx + 3000, num_frames)
            segment = mel[..., start_idx:end_idx]
            if segment.shape[-1] < 3000:
                segment = F.pad(segment, (0, 3000 - segment.shape[-1]))

            _, embeddings = self.encoder(segment, include_embeddings=True)

            segments.append(Segment(start=start_idx, end=end_idx, embeddings=embeddings))

        return EncoderOutput(segments=segments)

from pathlib import Path
from collections import OrderedDict
import os

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console
from whisper import load_model
import k2

console = Console()

from configs.args import ModelArgs


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight,
            None if self.bias is None else self.bias,
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight, None if bias is None else bias)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention=False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class SpeakerMLP(nn.Module):
    def __init__(self, n_state, n_layer, speaker_emb_dim):
        super().__init__()
        mlp = []
        mlp.append(Linear(n_state * 2, n_state * 4))
        mlp.append(nn.GELU())
        for i in range(n_layer - 1):
            mlp.append(Linear(n_state * 4, n_state * 4))
            mlp.append(nn.GELU())
        mlp.append(Linear(n_state * 4, speaker_emb_dim))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # mean + max pool (previous shape: (batch_size, 1500, n_state) -> (batch_size, n_state * 2))
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.mlp(x)


class AttributesMLP(nn.Module):
    def __init__(self, n_state, n_layer, n_attributes):
        super().__init__()
        mlp = []
        mlp.append(Linear(n_state * 2, n_state * 4))
        mlp.append(nn.GELU())
        for i in range(n_layer - 1):
            mlp.append(Linear(n_state * 4, n_state * 4))
            mlp.append(nn.GELU())
        mlp.append(Linear(n_state * 4, n_attributes))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # mean + max pool (previous shape: (batch_size, 1500, n_state) -> (batch_size, n_state * 2))
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.mlp(x)


class FrameProsodyMLP(nn.Module):
    def __init__(self, n_state, n_layer):
        super().__init__()
        mlp = []
        mlp.append(Linear(n_state, n_state))
        mlp.append(nn.GELU())
        for i in range(n_layer - 1):
            mlp.append(Linear(n_state, n_state))
            mlp.append(nn.GELU())
        mlp.append(Linear(n_state, 1))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # previous shape: (batch_size, 3000, n_state) -> (batch_size, 3000, 1))
        x = self.mlp(x)
        return x


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        n_mels = args.n_mels
        n_state = args.n_state
        n_head = args.n_head
        n_layer = args.n_layer
        n_ctx = args.n_ctx
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln_post = LayerNorm(n_state)

        speaker_postnet_layers = args.speaker_postnet_layers
        speaker_emb_dim = args.speaker_emb_dim
        self.speaker_mlp = SpeakerMLP(n_state, speaker_postnet_layers, speaker_emb_dim)

        self.postnet_phone_emb = nn.Embedding(args.n_phones, n_state)
        # position embedding for postnet cross attention
        self.register_buffer("phonenet_positional_embedding", sinusoids(500, n_state))

        self.postnet_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.pitch_mlp = FrameProsodyMLP(n_state, args.prosody_postnet_layers)
        self.energy_mlp = FrameProsodyMLP(n_state, args.prosody_postnet_layers)
        self.attributes_mlp = AttributesMLP(
            n_state, args.prosody_postnet_layers, args.n_attributes
        )

        self.phonenet = nn.Sequential(
            *[
                ResidualAttentionBlock(n_state, n_head, cross_attention=False)
                for _ in range(args.n_phonenet_layers)
            ]
        )

        self.postnet = nn.Sequential(
            *[
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(args.n_postnet_layers)
            ]
        )

        self.ln_phone = LayerNorm(n_state)

        self.phone_out = Linear(n_state, args.n_phones)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, phone_ids=None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = x + self.positional_embedding

        # phone_emb = self.postnet_phone_emb(phone_ids)
        # phone_emb = phone_emb + self.phonenet_positional_embedding
        # phone_emb = self.phonenet(phone_emb)

        for block in self.blocks:
            # x = block(x, xa=phone_emb)
            x = block(x)

        x = self.ln_post(x)

        # speaker embedding
        speaker_emb = self.speaker_mlp(x)

        # prev shape: (batch_size, 1500, n_state)
        # post shape: (batch_size, 3000, n_state)

        # upsample to 3000
        x = self.postnet_upsample(x.permute(0, 2, 1)).permute(0, 2, 1)

        # pitch
        pitch = self.pitch_mlp(x).squeeze(-1)

        # energy
        energy = self.energy_mlp(x).squeeze(-1)

        # attributes
        attributes = self.attributes_mlp(x)

        # postnet
        for block in self.postnet:
            # x = block(x, xa=phone_emb)
            x = block(x)

        x = self.ln_phone(x)
        # x = x @ torch.transpose(self.postnet_phone_emb.weight, 0, 1)
        x = self.phone_out(x)

        return {
            "phones": x,
            "speaker_emb": speaker_emb,
            "pitch": pitch,
            "energy": energy,
            "attributes": attributes,
        }

    def decode(self, log_prob, log_prob_len, targets):
        """
        Align targets to log_probs.

        Arguments
        ---------
        log_prob: torch.Tensor
            A tensor of shape (N, T, C) containing the log-probabilities.
            Please make sure that index 0 of the C dimension corresponds
            to the blank symbol.
        log_prob_len: torch.Tensor
            A tensor of shape (N,) containing the lengths of the log_probs.
            This is needed because the log_probs may have been padded.
            All elements in this tensor must be integers and <= T.
        targets: list
            A list of list of integers containing the targets.
            Note that the targets should not contain the blank symbol.
            The blank symbol is assumed to be index 0 in log_prob.
        Returns
        -------
        alignments: List[List[int]], containing the alignments.
        """
        assert log_prob.ndim == 3
        assert log_prob_len.ndim == 1
        assert log_prob.shape[0] == log_prob_len.shape[0]
        assert isinstance(targets, list)
        assert isinstance(targets[0], list)
        assert log_prob.shape[0] == len(targets)

        log_prob = torch.log_softmax(log_prob, dim=-1)

        N, T, C = log_prob.shape

        from time import time

        start = time()

        graph = k2.ctc_graph(targets, modified=True)

        lattice = k2.get_lattice(
            log_prob=log_prob,
            log_prob_len=log_prob_len,
            decoding_graph=graph,
        )

        best_path = k2.shortest_path(lattice, use_double_scores=True)
        labels = best_path.labels

        alignments = []
        alignment = []
        for e in labels.tolist():
            if e == -1:
                alignments.append(alignment)
                alignment = []
            else:
                alignment.append(e)

        print(f"decode time: {time() - start:.3f}s")

        return alignments

    def save_model(self, path, accelerator=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @classmethod
    def from_pretrained(cls, path_or_hubid, strict=True, freeze_whisper=True):
        # special cases
        if path_or_hubid == "distil-whisper/distil-large-v2":
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        model = cls(args)
        model.load_state_dict(torch.load(model_file), strict=strict)
        args.freeze_whisper = freeze_whisper
        if args.freeze_whisper:
            print("Freezing whisper model")
            # freeze conv1, conv2, and blocks
            for name, param in model.named_parameters():
                if (
                    name.startswith("conv1")
                    or name.startswith("conv2")
                    or name.startswith("blocks")
                    or name.startswith("ln_post")
                ):
                    param.requires_grad = False
        return model

    @classmethod
    def init_from_whisper(cls, whisper_name, args):
        whisper_model = load_model(whisper_name)
        dims = whisper_model.dims
        args.n_mels = dims.n_mels
        args.n_ctx = dims.n_audio_ctx
        args.n_state = dims.n_audio_state
        args.n_head = dims.n_audio_head
        args.n_layer = dims.n_audio_layer
        model = cls(args)
        model.load_state_dict(whisper_model.state_dict(), strict=False)
        if args.freeze_whisper:
            for name, param in model.named_parameters():
                if name in whisper_model.state_dict():
                    param.requires_grad = False
        del whisper_model
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        return [
            torch.randn(1, self.args.n_mels, self.args.n_ctx * 2),
            torch.zeros(1, 500, dtype=torch.long),
        ]

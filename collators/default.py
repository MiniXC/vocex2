import torch
from torch import nn
import numpy as np

from configs.args import CollatorArgs

import nemo.collections.asr as nemo_asr

import torchaudio


embs1 = speaker_model.get_embedding("figures/docs_diff_villefort.wav")

print(
    torch.abs(embs1 - embs2).mean(),
    torch.abs(embs1 - embs2).max(),
    torch.abs(embs1).max(),
    torch.abs(embs2).max(),
)


class VocexCollator(nn.Module):
    def __init__(
        self,
        args: CollatorArgs,
    ):
        super().__init__()

        self.args = args
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )

    def __call__(self, batch):
        speaker_model.get

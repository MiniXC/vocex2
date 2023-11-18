import matplotlib.pyplot as plt
import torch

from configs.args import TrainingArgs


def plot_first_batch(batch, args: TrainingArgs):
    print(batch)
    for i in range(len(batch["mel"])):
        mel = batch["mel"][i]
        phone_spans = batch["phone_spans"][i]
        silences = batch["silences"][i]
        mel_len = batch["mel_len"][i]
        trimmed_mel = mel[:, :mel_len]
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.imshow(torch.flip(trimmed_mel, dims=(0,)).numpy())
        for phone, (start, end) in phone_spans:
            # add a line for each phone
            ax.axvline(start, color="r")
            ax.axvline(end, color="r")
            ax.text(
                (start + end) / 2,
                10,
                phone,
                horizontalalignment="center",
                verticalalignment="center",
            )
        for start, end in silences:
            ax.axvline(start, color="b")
            ax.axvline(end, color="b")
        plt.savefig(f"figures/batch_{i}.png")

import matplotlib.pyplot as plt
import torch

from configs.args import TrainingArgs


def plot_item(mel, phone_spans, mel_len, args: TrainingArgs, silences=None):
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
    if silences is not None:
        for start, end in silences:
            ax.axvline(start, color="b")
            ax.axvline(end, color="b")
    return fig


def plot_first_batch(batch, args: TrainingArgs):
    for i in range(len(batch["mel"])):
        mel = batch["mel"][i]
        phone_spans = batch["phone_spans"][i]
        silences = batch["silences"][i]
        mel_len = batch["mel_len"][i]
        trimmed_mel = mel[:, :mel_len]
        fig = plot_item(mel, phone_spans, mel_len, args, silences)
        fig.savefig(f"figures/batch_{i}.png")


def plot_predictions(mel, mel_len, pred_phone_ids, id2phone, args: TrainingArgs):
    # convert phone_ids to phone_spans
    phone_spans = []
    start = 0
    if isinstance(pred_phone_ids, torch.Tensor):
        pred_phone_ids = pred_phone_ids.tolist()
    for phone_id in pred_phone_ids:
        if phone_id != 0:
            if len(phone_spans) == 0:
                phone_spans.append([phone_id, start, -1])
            elif phone_spans[-1][0] == phone_id:
                phone_spans[-1][2] = start - 1
            else:
                phone_spans.append([phone_id, start, -1])
                phone_spans[-2][2] = start
        start += 1
    phone_spans = [
        [id2phone[phone_id], start, end] for phone_id, start, end in phone_spans
    ]
    fig = plot_item(mel, phone_spans, mel_len, args)
    return fig

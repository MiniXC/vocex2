import matplotlib.pyplot as plt

from configs.args import TrainingArgs


def plot_first_batch(batch, args: TrainingArgs):
    fig, axes = plt.subplots(
        nrows=1, ncols=args.batch_size, figsize=(3 * args.batch_size, 3)
    )
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(batch["image"][i].view(28, 28), cmap="gray")
        ax.set_title(batch["target"][i].item())
        ax.axis("off")
    plt.tight_layout()
    return fig

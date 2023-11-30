import os
import sys
from collections import deque
from pathlib import Path
import typing
from dataclasses import fields
from torchmetrics.text import CharErrorRate

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset
import pandas as pd

# logging & etc
from torchinfo import summary
from torchview import draw_graph
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import yaml
from rich.console import Console

# plotting
import matplotlib.pyplot as plt
from util.plotting import plot_first_batch, plot_predictions, plot_item

console = Console()

# local imports
from configs.args import TrainingArgs, ModelArgs, CollatorArgs
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from model.whisper_encoder import WhisperAudioEncoder
from collators import get_collator

MODEL_CLASS = WhisperAudioEncoder

"""
TODO:
- [ ] add training for speaker embedding
- [ ] add training for prosody frame level prediction
- [ ] add training for utterance level prosody/channel prediction (mean, std)
- [ ] add evaluation for the above
- [ ] add evaluation plotting (alignment & spectrogram as in first batch)
- [ ] call espeak manually to avoid memory problems
"""


class DatasetFromDataframe(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row

    def __len__(self):
        return len(self.df)


def train_epoch(epoch):
    global global_step
    losses = deque(maxlen=training_args.log_every_n_steps)
    phone_losses = deque(maxlen=training_args.log_every_n_steps)
    speaker_losses = deque(maxlen=training_args.log_every_n_steps)
    attribute_losses = deque(maxlen=training_args.log_every_n_steps)
    pitch_losses = deque(maxlen=training_args.log_every_n_steps)
    energy_losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    console_rule(f"Epoch {epoch}")
    last_loss = None

    for batch in train_dl:
        if batch is None:
            continue
        mel_input = batch["mel"]
        phone_cond = batch["transcript_phonemized"]
        phone_mask = batch["transcript_mask"]

        # 25% of the time, mask out parts of the mel to encourage the model to learn from the phone condition
        if np.random.rand() < 0.25:
            mel_mask = torch.rand((mel_input.shape[0], mel_input.shape[2])) > 0.5
            mel_mask = mel_mask.unsqueeze(1).repeat(1, mel_input.shape[1], 1)
            mel_input = mel_input * mel_mask

        phone_target = batch["phone_ids"]
        speaker_target = batch["speaker_emb"]
        attribute_target = batch["attributes"]
        pitch_target = batch["pitch"]
        energy_target = batch["energy"]

        preds = model(
            mel_input,
            phone_cond,
            phone_mask,
        )

        # use cross entropy for phone targets (with label smoothing)
        phone_loss = torch.nn.functional.cross_entropy(
            preds["phones"].permute(0, 2, 1), phone_target, reduction="none"
        ) * batch["loss_mask"]
        phone_loss = phone_loss.sum() / batch["loss_mask"].sum()
        # use mse for all other targets
        speaker_loss = nn.MSELoss()(preds["speaker_emb"], speaker_target)
        attribute_loss = nn.MSELoss()(preds["attributes"], attribute_target)
        pitch_loss = nn.MSELoss(reduction="none")(preds["pitch"], pitch_target) * (
            batch["loss_mask"]
        )
        pitch_loss = pitch_loss.sum() / batch["loss_mask"].sum()
        energy_loss = nn.MSELoss(reduction="none")(preds["energy"], energy_target) * (
            batch["loss_mask"]
        )
        energy_loss = energy_loss.sum() / batch["loss_mask"].sum()

        loss = (
            phone_loss
            + speaker_loss * 5
            + attribute_loss * 5
            + pitch_loss * 5
            + energy_loss * 5
        ) / 5

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), training_args.gradient_clip_val)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.detach())
        phone_losses.append(phone_loss.detach())
        speaker_losses.append(speaker_loss.detach())
        attribute_losses.append(attribute_loss.detach())
        pitch_losses.append(pitch_loss.detach())
        energy_losses.append(energy_loss.detach())
        if (
            step > 0
            and step % training_args.log_every_n_steps == 0
            and accelerator.is_main_process
        ):
            last_loss = torch.mean(torch.tensor(losses)).item()
            last_phone_loss = torch.mean(torch.tensor(phone_losses)).item()
            last_speaker_loss = torch.mean(torch.tensor(speaker_losses)).item()
            last_attribute_loss = torch.mean(torch.tensor(attribute_losses)).item()
            last_pitch_loss = torch.mean(torch.tensor(pitch_losses)).item()
            last_energy_loss = torch.mean(torch.tensor(energy_losses)).item()
            wandb_log(
                "train",
                {
                    "loss": last_loss,
                    "phone_loss": last_phone_loss,
                    "speaker_loss": last_speaker_loss,
                    "attribute_loss": last_attribute_loss,
                    "pitch_loss": last_pitch_loss,
                    "energy_loss": last_energy_loss,
                },
                print_log=True,
            )
        if (
            training_args.do_save
            and global_step > 0
            and global_step % training_args.save_every_n_steps == 0
        ):
            save_checkpoint()
        if training_args.n_steps is not None and global_step >= training_args.n_steps:
            return
        if (
            training_args.eval_every_n_steps is not None
            and global_step > 0
            and global_step % training_args.eval_every_n_steps == 0
            and accelerator.is_main_process
        ):
            evaluate()
            console_rule(f"Epoch {epoch}")
        step += 1
        global_step += 1
        if accelerator.is_main_process:
            pbar.update(1)
            if last_loss is not None:
                pbar.set_postfix(
                    {
                        "loss": f"{last_loss:.3f}",
                        "phone_loss": f"{last_phone_loss:.3f}",
                        "speaker_loss": f"{last_speaker_loss:.3f}",
                        "attribute_loss": f"{last_attribute_loss:.3f}",
                        "pitch_loss": f"{last_pitch_loss:.3f}",
                        "energy_loss": f"{last_energy_loss:.3f}",
                    }
                )


@torch.no_grad()
def evaluate(device="cpu"):
    eval_model = create_latest_model_for_eval(device)
    eval_model.eval()
    if accelerator.is_main_process:
        cer_metric = CharErrorRate()
        cer_metric_transcript = CharErrorRate()
        y_true = []
        y_pred = []
        losses = []
        phone_losses = []
        speaker_losses = []
        attribute_losses = []
        pitch_losses = []
        energy_losses = []
        console_rule("Evaluation")
        for i in tqdm(range(10), desc="eval"):
            bs = training_args.batch_size
            batch = collator([val_ds[j + (i * bs)] for j in range(bs)])
            mel_len = batch["mel_len"]
            # forward pass
            from time import time

            start = time()
            pred = eval_model(
                batch["mel"].to(device),
                batch["transcript_phonemized"].to(device),
                batch["transcript_mask"].to(device),
            )
            print(f"forward pass took {time()-start:.3f}s")
            phone_ids = batch["phone_ids"].to(device)
            phone_pred = pred["phones"]
            phone_loss = torch.nn.functional.cross_entropy(
                phone_pred.permute(0, 2, 1), phone_ids, reduction="none"
            ) * batch["loss_mask"].to(device)
            phone_loss = phone_loss.sum() / batch["loss_mask"].sum()
            speaker_loss = nn.MSELoss()(
                pred["speaker_emb"], batch["speaker_emb"].to(device)
            )
            attribute_loss = nn.MSELoss()(
                pred["attributes"], batch["attributes"].to(device)
            )
            pitch_loss = nn.MSELoss(reduction="none")(
                pred["pitch"], batch["pitch"].to(device)
            ) * batch["loss_mask"].to(device)
            pitch_loss = pitch_loss.sum() / batch["loss_mask"].sum()
            energy_loss = nn.MSELoss(reduction="none")(
                pred["energy"], batch["energy"].to(device)
            ) * batch["loss_mask"].to(device)
            energy_loss = energy_loss.sum() / batch["loss_mask"].sum()
            loss = (
                phone_loss
                + speaker_loss * 5
                + attribute_loss * 5
                + pitch_loss * 5
                + energy_loss * 5
            ) / 5
            losses.append(loss.item())
            phone_losses.append(phone_loss.item())
            speaker_losses.append(speaker_loss.item())
            attribute_losses.append(attribute_loss.item())
            pitch_losses.append(pitch_loss.item())
            energy_losses.append(energy_loss.item())
            align_list = []
            for item in batch["transcript_phonemized"]:
                item = [x for x in item if x != 0]
                align_list.append(item)
            phone_pred_new = []
            for j in range(bs):
                pred_new = eval_model.decode(
                    phone_pred[j].unsqueeze(0),
                    torch.tensor(mel_len[j]).unsqueeze(0),
                    [align_list[j]],
                )
                if len(pred_new) == 0:
                    pred_new = [[0]]
                phone_pred_new.append(pred_new[0])
            # for each list in phone_pred, replace -1 with 0, and pad to max length
            phone_pred = [
                np.pad(
                    np.array([0 if x == -1 else x for x in item]),
                    (0, 3000 - len(item)),
                    constant_values=0,
                )
                for item in phone_pred_new
            ]
            print(phone_pred)
            for j in range(bs):
                if i == 0 and j == 0:
                    fig = plot_item(
                        batch["mel"][j],
                        batch["phone_spans"][j],
                        batch["mel_len"][j],
                        training_args,
                        batch["silences"][j],
                    )
                    fig.savefig("figures/eval.png")
                    fig = plot_predictions(
                        batch["mel"][j],
                        batch["mel_len"][j],
                        phone_pred[j],
                        collator.id2phone,
                        training_args,
                    )
                    fig.savefig("figures/eval_pred.png")
                    wandb.log(
                        {
                            "eval/alignment": wandb.Image("figures/eval.png"),
                            "eval/predictions": wandb.Image("figures/eval_pred.png"),
                        },
                        step=global_step,
                    )
                y_pred.append(phone_pred[j])
                y_true.append(phone_ids[j])
                # convert to strings for cer
                phone_pred_s = "".join([chr(x + 97) for x in phone_pred[j].tolist()])
                phone_true_s = "".join([chr(x + 97) for x in phone_ids[j].tolist()])
                phone_true_s_transcript = "".join(
                    [chr(x + 97) for x in batch["transcript_phonemized"][j].tolist()]
                )
                # remove repeated characters
                phone_pred_s = "".join(
                    [x for x, y in zip(phone_pred_s, phone_pred_s[1:]) if x != y]
                )
                phone_true_s = "".join(
                    [x for x, y in zip(phone_true_s, phone_true_s[1:]) if x != y]
                )
                phone_true_s_transcript = "".join(
                    [
                        x
                        for x, y in zip(
                            phone_true_s_transcript, phone_true_s_transcript[1:]
                        )
                        if x != y
                    ]
                )
                cer_metric(phone_pred_s, phone_true_s)
                cer_metric_transcript(phone_pred_s, phone_true_s_transcript)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        cer = cer_metric.compute()
        cer_transcript = cer_metric_transcript.compute()
        loss = np.mean(losses)
        phone_loss = np.mean(phone_losses)
        speaker_loss = np.mean(speaker_losses)
        attribute_loss = np.mean(attribute_losses)
        pitch_loss = np.mean(pitch_losses)
        energy_loss = np.mean(energy_losses)
        wandb_log(
            "val",
            {
                "cer": cer.item(),
                "cer_transcript": cer_transcript.item(),
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "loss": loss,
                "phone_loss": phone_loss,
                "speaker_loss": speaker_loss,
                "attribute_loss": attribute_loss,
                "pitch_loss": pitch_loss,
                "energy_loss": energy_loss,
            },
            print_log=True,
        )


def main():
    global accelerator, training_args, model_args, collator, val_ds, collator_args, train_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    accelerator = Accelerator()

    # parse args
    (
        training_args,
        model_args,
        collator_args,
    ) = parse_args([TrainingArgs, ModelArgs, CollatorArgs])

    # check if run name is specified
    if training_args.run_name is None:
        raise ValueError("run_name must be specified")
    if (
        training_args.do_save
        and (Path(training_args.checkpoint_path) / training_args.run_name).exists()
    ):
        raise ValueError(f"run_name {training_args.run_name} already exists")

    # wandb
    if accelerator.is_main_process:
        wandb_name, wandb_project, wandb_dir, wandb_mode = (
            training_args.run_name,
            training_args.wandb_project,
            training_args.wandb_dir,
            training_args.wandb_mode,
        )
        wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode)
        wandb.run.log_code()

    # log args
    console_rule("Arguments")
    console_print(training_args)
    console_print(model_args)
    console_print(collator_args)
    if accelerator.is_main_process:
        wandb_update_config(
            {
                "training": training_args,
                "model": model_args,
            }
        )
    validate_args(training_args, model_args, collator_args)

    # Distribution Information
    console_rule("Distribution Information")
    console_print(f"[green]accelerator[/green]: {accelerator}")
    console_print(f"[green]n_procs[/green]: {accelerator.num_processes}")
    console_print(f"[green]process_index[/green]: {accelerator.process_index}")

    # model
    if training_args.from_pretrained is not None:
        model = MODEL_CLASS.from_pretrained(
            training_args.from_pretrained,
            strict=False,
            freeze_whisper=model_args.freeze_whisper,
        )
    elif training_args.from_whisper is not None and training_args.from_whisper not in [
        "None",
        "none",
        "",
    ]:
        model = MODEL_CLASS.init_from_whisper(training_args.from_whisper, model_args)
    else:
        model = MODEL_CLASS(model_args)

    console_rule("Model")
    print_and_draw_model()

    # dataset
    console_rule("Dataset")

    data = pd.read_json(
        path_or_buf=Path(training_args.libriheavy_path)
        / "libriheavy_cuts_small.jsonl.gz",
        lines=True,
    )
    data["speaker_id"] = data["supervisions"].apply(lambda x: x[0]["speaker"])
    # remove all rows with "duration" >= 30
    len_pre = len(data)
    data = data[data["duration"] < 30]
    len_post = len(data)
    console_print(
        f"[green]removed rows due to duration >= 30s[/green]: {len_pre-len_post} ({(len_pre-len_post)/len_pre*100:.2f}%)"
    )
    # move 10 randomly selected speakers to val_ds
    speaker_ids = data["speaker_id"].unique()
    np.random.seed(training_args.seed)
    val_speaker_ids = np.random.choice(speaker_ids, 10, replace=False)
    val_ds = data[data["speaker_id"].isin(val_speaker_ids)]
    train_ds = data.drop(val_ds.index)
    collator_args.libriheavy_path = training_args.libriheavy_path
    # shuffle val_ds with seed
    val_ds = val_ds.sample(frac=1, random_state=training_args.seed)
    console_print(f"[green]dataset path[/green]: {training_args.libriheavy_path}")
    console_print(f"[green]train_split size[/green]: {len(train_ds)}")
    console_print(f"[green]val_split size[/green]: {len(val_ds)}")

    train_ds = DatasetFromDataframe(train_ds)
    val_ds = DatasetFromDataframe(val_ds)

    # collator
    collator = get_collator(collator_args)

    # plot first batch
    if accelerator.is_main_process:
        first_batch = collator([train_ds[i] for i in range(training_args.batch_size)])
        plot_first_batch(first_batch, training_args)
        plt.savefig("figures/first_batch.png")

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=os.cpu_count(),
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
    )

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )
    else:
        raise NotImplementedError(f"{training_args.lr_schedule} not implemented")

    # accelerator
    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    # evaluation
    if training_args.eval_only:
        console_rule("Evaluation")
        seed_everything(training_args.seed)
        evaluate()
        return

    # training
    console_rule("Training")
    seed_everything(training_args.seed)
    pbar_total = training_args.n_steps
    training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    console_print(f"[green]n_epochs[/green]: {training_args.n_epochs}")
    console_print(
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes}"
    )
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        train_epoch(i)
    console_rule("Evaluation")
    seed_everything(training_args.seed)
    evaluate()

    # save final model
    console_rule("Saving")
    if training_args.do_save:
        save_checkpoint()

    # wandb sync reminder
    if accelerator.is_main_process and training_args.wandb_mode == "offline":
        console_rule("Weights & Biases")
        console_print(
            f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
        )


# helper functions (change them in parent repository if needed)


def parse_args(argument_classes):
    parser = HfArgumentParser(argument_classes)
    type_dicts = []
    for arg_class in argument_classes:
        resolved_hints = typing.get_type_hints(arg_class)
        field_names = [field.name for field in fields(arg_class)]
        resolved_field_types = {name: resolved_hints[name] for name in field_names}
        type_dicts.append(resolved_field_types)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        # additonally parse args from command line
        parsed_args = parser.parse_args_into_dataclasses(sys.argv[2:])
        args_in_argv = sys.argv[2:]
        updated_args = []
        for arg in args_in_argv:
            if "--" in arg:
                arg = arg.replace("--", "")
                updated_args.append(arg)
            if "=" in arg:
                arg1, arg2 = arg.split("=")
                updated_args.append(arg1)
                updated_args.append(arg2)
        # remove every second element (values)
        updated_args = updated_args[::2]
        args_in_argv = updated_args
        # update args from yaml (only if not specified in command line)
        for k, v in args_dict.items():
            for j, arg_class in enumerate(parsed_args):
                if hasattr(arg_class, k) and k not in args_in_argv:
                    setattr(arg_class, k, type_dicts[j][k](v))
    else:
        parsed_args = parser.parse_args_into_dataclasses()
    return parsed_args


def print_and_draw_model():
    bsz = training_args.batch_size
    dummy_input = model.dummy_input
    # repeat dummy input to match batch size (regardless of how many dimensions)
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.repeat((bsz,) + (1,) * (len(dummy_input.shape) - 1))
        console_print(f"[green]input shape[/green]: {dummy_input.shape}")
    elif isinstance(dummy_input, list):
        dummy_input = [
            x.repeat((bsz,) + (1,) * (len(x.shape) - 1)) for x in dummy_input
        ]
        console_print(f"[green]input shapes[/green]: {[x.shape for x in dummy_input]}")

    model_summary = summary(
        model,
        input_data=dummy_input,
        verbose=0,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
    )
    console_print(model_summary)
    Path("figures").mkdir(exist_ok=True)
    if accelerator.is_main_process:
        _ = draw_graph(
            model,
            input_data=dummy_input,
            save_graph=True,
            directory="figures/",
            filename="model",
            expand_nested=True,
        )
        # remove "figures/model" file
        os.remove("figures/model")


def console_print(*args, **kwargs):
    if accelerator.is_main_process:
        console.print(*args, **kwargs)


def console_rule(*args, **kwargs):
    if accelerator.is_main_process:
        console.rule(*args, **kwargs)


def wandb_log(prefix, log_dict, round_n=3, print_log=True):
    if accelerator.is_main_process:
        log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
        wandb.log(log_dict, step=global_step)
        if print_log:
            log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
            console.log(log_dict)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(name_override=None):
    accelerator.wait_for_everyone()
    checkpoint_name = training_args.run_name
    if name_override is not None:
        name = name_override
    else:
        name = f"step_{global_step}"
    checkpoint_path = Path(training_args.checkpoint_path) / checkpoint_name / name
    if name_override is None:
        # remove old checkpoints
        if checkpoint_path.exists():
            for f in checkpoint_path.iterdir():
                os.remove(f)
    # model
    model.save_model(checkpoint_path, accelerator)
    if accelerator.is_main_process:
        # training args
        with open(checkpoint_path / "training_args.yml", "w") as f:
            f.write(yaml.dump(training_args.__dict__, Dumper=yaml.Dumper))
        # collator args
        with open(checkpoint_path / "collator_args.yml", "w") as f:
            f.write(yaml.dump(collator_args.__dict__, Dumper=yaml.Dumper))
    accelerator.wait_for_everyone()
    return checkpoint_path


def create_latest_model_for_eval(device="cpu"):
    checkpoint_path = save_checkpoint("latest")
    if accelerator.is_main_process:
        eval_model = MODEL_CLASS.from_pretrained(checkpoint_path)
        eval_model.eval()
        eval_model = eval_model.to(device)
        return eval_model
    return None


# main

if __name__ == "__main__":
    main()

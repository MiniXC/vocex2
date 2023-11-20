from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
    lr_warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    checkpoint_path: str = "checkpoints"
    output_path: str = "outputs"
    from_whisper: str = "tiny.en"
    run_name: str = None
    wandb_mode: str = "offline"
    wandb_project: str = "vocex2.5"
    wandb_dir: str = "wandb"
    libriheavy_size: str = "small"
    libriheavy_path: str = "/dev/shm/libriheavy-small"
    n_steps: int = 50000
    batch_size: int = 4
    seed: int = 0
    log_every_n_steps: int = 100
    do_full_eval: bool = True
    do_save: bool = False
    save_onnx: bool = False
    eval_only: bool = False
    eval_steps: int = 10
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 1000
    push_to_hub: bool = False
    hub_repo: str = None


@dataclass
class CollatorArgs:
    speaker_model: str = "titanet_large"
    name: str = "default"
    phone_len: int = 500


@dataclass
class ModelArgs:
    # default matches "tiny" version of whisper
    n_mels: int = 80
    n_ctx: int = 1500
    n_state: int = 384
    n_head: int = 6
    n_layer: int = 4
    n_postnet_layers: int = 4
    speaker_postnet_layers: int = 4
    speaker_emb_dim: int = 192
    n_phones: int = 403
    freeze_whisper: bool = True

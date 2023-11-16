import yaml

from .default import VocexCollator
from configs.args import CollatorArgs
import torch


def get_collator(args: CollatorArgs):
    return {
        "default": VocexCollator,
    }[
        args.name
    ](args)

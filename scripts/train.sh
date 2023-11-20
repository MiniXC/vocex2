#!/bin/bash
if [ "$1" = "--dryrun" ]; then
	cd /dev/shm
    git clone https://github.com/MiniXC/libriheavy-small.git
    cd /dev/shm/libriheavy-small
    bash run.sh --stage -1 --stop-stage -1
    bash run.sh --stage 1 --stop-stage 1
    sudo apt-get install espeak -y
fi
if [ "$1" = "--machine" ] && [ "$2" = "v3-1" ]; then
	/usr/bin/python scripts/train.py --freeze_whisper=True --from_whisper="tiny.en" --wandb_mode=online --run_name "tiny_frozen_align"
    /usr/bin/python scripts/train.py --freeze_whisper=False --from_whisper="tiny.en" --wandb_mode=online --run_name "tiny_unfrozen_align"
fi
if [ "$1" = "--machine" ] && [ "$2" = "v3-2" ]; then
    /usr/bin/python scripts/train.py --freeze_whisper=True --from_whisper="small.en" --wandb_mode=online --run_name "small_frozen_align"
    /usr/bin/python scripts/train.py --freeze_whisper=True --from_whisper="small.en" --n_postnet_layers=8 --wandb_mode=online --run_name "small_frozen_align_8"
fi
if [ "$1" = "--machine" ] && [ "$2" = "v3-3" ]; then
    /usr/bin/python scripts/train.py --freeze_whisper=True --from_whisper="medium.en" --wandb_mode=online --run_name "medium_frozen_align"
    /usr/bin/python scripts/train.py --freeze_whisper=True --from_whisper="medium.en" --n_postnet_layers=8 --wandb_mode=online --run_name "medium_frozen_align_8"
fi
if [ "$1" = "--machine" ] && [ "$2" = "v3-4" ]; then
    /usr/bin/python scripts/train.py --freeze_whisper=False --from_whisper="none" --wandb_mode=online --run_name "none_align"
    /usr/bin/python scripts/train.py --freeze_whisper=False --from_whisper="small.en" --wandb_mode=online --run_name "small_unfrozen_align"
    /usr/bin/python scripts/train.py --freeze_whisper=False --from_whisper="medium.en" --wandb_mode=online --run_name "medium_unfrozen_align"
fi
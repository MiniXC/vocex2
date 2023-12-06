# inference
python3 scripts/train.py --from_pretrained="checkpoints/final_s/latest" --eval_only --wandb_mode=offline --run_name "test" --seed 0

# train
python3 scripts/train.py --from_pretrained="checkpoints/final_s_cont/latest" --wandb_mode=online --run_name "final_s_cont2" --seed 0
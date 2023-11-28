[![tests](https://github.com/MiniXC/ml-template/actions/workflows/run_lint_and_test.yml/badge.svg)](https://github.com/MiniXC/vocex2/actions/workflows/run_lint_and_test.yml)
# Vocex2

## Dependencies

```bash
sudo apt-get install espeak -y
pip install k2==1.24.3.dev20230719+cpu.torch1.13.1 -f https://k2-fsa.github.io/k2/cpu.html
pip install -r requirements.txt
```

python3 scripts/train.py --freeze_whisper=False --from_whisper="base.en" --wandb_mode=online --run_name "base_new_cond"
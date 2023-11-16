import sys

sys.path.append(".")

from model.whisper_encoder import SimpleMLP
from configs.args import ModelArgs

model = SimpleMLP(ModelArgs())

OUT_SHAPE = (1, 10)


def test_forward_pass():
    x = model.dummy_input
    y = model(x)
    assert y.shape == OUT_SHAPE


def test_save_load_model(tmp_path):
    model.save_model(tmp_path / "test")
    model.from_pretrained(tmp_path / "test")
    x = model.dummy_input
    y = model(x)
    assert y.shape == OUT_SHAPE

# coding: utf-8
from dataclasses import dataclass
from pathlib import Path

from .eval import Predict as BasePredictor
import torch


@dataclass
class DummyConfig():
    model_type: str
    model_load_path: str
    dataset: str = None
    data_path: str = None
    batch_size: int = 1


project_dir = Path(__file__).parent.resolve()
MODEL_DIR = project_dir / 'models'


class MusicTagger(BasePredictor):

    def __init__(self, model_type, training_data, batch_size=1):

        model_load_path = MODEL_DIR / training_data / model_type / 'best_model.pth'

        config = DummyConfig(model_type, model_load_path, batch_size=batch_size)
        super(MusicTagger, self).__init__(config)

    def preprocess(self, x):
        """Ready an array x to input into the network."""
        length = len(x)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(x[i * hop:i * hop + self.input_length]).unsqueeze(0)
        return x

    def forward(self, x):
        """Forward pass of the net."""
        x = self.preprocess(x)
        return self.model(self.to_var(x))

    def __call__(self, x):
        return self.forward(x)
# coding: utf-8
from dataclasses import dataclass
from pathlib import Path

from .eval import Predict as BasePredictor
from .tag_metadata import MSD_TAGS, MTAT_TAGS, JAMENDO_TAGS
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
        self.training_data = training_data

        config = DummyConfig(model_type, model_load_path, batch_size=batch_size)
        super(MusicTagger, self).__init__(config)

    def preprocess(self, raw):
        """Ready an array x to input into the network."""
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length).to(raw.device)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i * hop:i * hop + self.input_length]).to(raw.device).unsqueeze(0)
        return x

    def forward(self, x):
        """Forward pass of the net."""
        x = self.preprocess(x)
        return self.model(self.to_var(x))

    def __call__(self, x):
        return self.forward(x)

    def _get_metadata(self):
        if self.training_data.lower() == 'mtat':
            return MTAT_TAGS
        elif self.training_data.lower() == 'msd':
            return MSD_TAGS
        elif self.training_data.lower() == 'jamendo':
            return JAMENDO_TAGS
        else:
            raise ValueError(f'Did not understand dataset: {self.training_data}.')

    def forward_labels(self, x):
        """Forward pass with labels."""
        predictions = self.forward(x)

        # TODO: Assume 1D output for now...
        predictions = list(predictions.detach().numpy().flatten())
        return list(zip(self._get_metadata(), predictions))


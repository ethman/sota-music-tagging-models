# coding: utf-8
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .eval import Predict as BasePredictor
from .tag_metadata import MSD_TAGS, MTAT_TAGS, JAMENDO_TAGS


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

    def __init__(self, model_type, training_data, return_feats=False):

        model_load_path = MODEL_DIR / training_data / model_type / 'best_model.pth'
        self.training_data = training_data

        config = DummyConfig(model_type, model_load_path)
        super(MusicTagger, self).__init__(config, return_feats)

    def preprocess(self, raw):
        """Ready an array x to input into the network."""
        length = len(raw)
        if isinstance(raw, np.ndarray):
            raw = torch.from_numpy(raw)
        x = raw

        batch = np.ceil(length / self.input_length).astype(int)
        if length % self.input_length:
            # Only pad if not a perfect window size already
            pad = self.input_length - length % self.input_length
            x = F.pad(x, (0, pad), 'constant', 0)

        x = x.reshape(batch, self.input_length)

        if self.is_cuda:
            return x.cuda()
        return x

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def forward(self, x):
        """Forward pass of the net."""
        x = self.preprocess(x)
        # TODO: Aggregate predictions if batch size is used to chop up long inputs
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return f'{self.model_type}_{self.training_data}'

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


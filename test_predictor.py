# coding: utf-8
import warnings

import librosa
import numpy as np
import torch
from sota_music_taggers.predict import MusicTagger
from sota_music_taggers.tag_metadata import MSD_TAGS, MTAT_TAGS, JAMENDO_TAGS


TAGGER_SR = 16000  # Hz

all_models = ['fcn', 'musicnn', 'crnn', 'sample', 'se', 'attention', 'hcnn', 'short', 'short_res']
all_datasets = ['jamendo', 'msd', 'mtat']


def test_all_combos():
    warnings.simplefilter('ignore')

    audio_path = 'example_audio/mixture.mp3'
    audio, sr = librosa.load(audio_path, sr=TAGGER_SR)

    for training_data in all_datasets:
        print(f'     {training_data}')
        print('  ------------------')
        for model_type in all_models:
            s = f'{model_type:9}  '
            try:
                model = MusicTagger(model_type, training_data, batch_size=1)
                s += f'  Yes  Loaded     '
            except:
                s += f'  Not  Loaded     - No  Outputs'
                continue

            try:
                outputs = model(audio)
                s += f'  Yes  Outputs'
            except Exception as e:
                s += f' - No  Outputs'
            print(s)
        print('\n\n')


def simple_test():
    audio_path = 'example_audio/mixture.mp3'
    audio, sr = librosa.load(audio_path, sr=TAGGER_SR, mono=True)

    audio = torch.Tensor(audio)

    training_data = 'mtat'
    model_type = 'fcn'

    model = MusicTagger(model_type, training_data, batch_size=1)
    # outputs = model(audio)

    labels = model.forward_labels(audio)

    i = 0


def load_tag_labels():



    i = 0

if __name__ == '__main__':
    # test_all_combos()
    simple_test()
    # load_tag_labels()
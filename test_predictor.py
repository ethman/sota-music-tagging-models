# coding: utf-8
from pathlib import Path

import librosa
from sota_music_taggers.predict import Predictor


TAGGER_SR = 16000  # Hz

all_models = ['fcn', 'musicnn', 'crnn', 'sample', 'se', 'attention', 'hcnn', 'short', 'short_res']
all_datasets = ['jamendo', 'msd', 'mtat']


def test_all_combos():
    audio_path = 'example_audio/mixture.mp3'
    audio, sr = librosa.load(audio_path, sr=TAGGER_SR)

    # training_data = 'mtat'
    # model_type = 'hcnn'

    for training_data in all_datasets:
        print(f'     {training_data}')
        print('  ------------------')
        for model_type in all_models:
            try:
                model = Predictor(model_type, training_data, batch_size=1)
                print(f'  Yes  Loaded    {model_type}')
            except:
                print(f'- No   Loading   {model_type}')
                continue

            try:
                outputs = model(audio)
                print(f'  Yes  Outputs   {model_type}')
            except Exception as e:
                print(f'- No   Outputs   {model_type}')

        print('\n\n')
    i = 0


if __name__ == '__main__':
    test_all_combos()
# coding: utf-8
import warnings

import librosa
from sota_music_taggers.predict import Predictor


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
                model = Predictor(model_type, training_data, batch_size=1)
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
    i = 0


if __name__ == '__main__':
    test_all_combos()
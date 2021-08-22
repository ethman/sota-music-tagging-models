
import numpy as np
from pathlib import Path


JAMENDO_TAGS = [
    'genre---downtempo',
    'genre---ambient',
    'genre---rock',
    'instrument---synthesizer',
    'genre---atmospheric',
    'genre---indie',
    'instrument---electricpiano',
    'genre---newage',
    'instrument---strings',
    'instrument---drums',
    'instrument---drummachine',
    'genre---techno',
    'instrument---guitar',
    'genre---alternative',
    'genre---easylistening',
    'genre---instrumentalpop',
    'genre---chillout',
    'genre---metal',
    'mood/theme---happy',
    'genre---lounge',
    'genre---reggae',
    'genre---popfolk',
    'genre---orchestral',
    'instrument---acousticguitar',
    'genre---poprock',
    'instrument---piano',
    'genre---trance',
    'genre---dance',
    'instrument---electricguitar',
    'genre---soundtrack',
    'genre---house',
    'genre---hiphop',
    'genre---classical',
    'mood/theme---energetic',
    'genre---electronic',
    'genre---world',
    'genre---experimental',
    'instrument---violin',
    'genre---folk',
    'mood/theme---emotional',
    'instrument---voice',
    'instrument---keyboard',
    'genre---pop',
    'instrument---bass',
    'instrument---computer',
    'mood/theme---film',
    'genre---triphop',
    'genre---jazz',
    'genre---funk',
    'mood/theme---relaxing'
]

_datafiles_path = Path(__file__).parent.resolve()
MTAT_TAGS = list(np.load(_datafiles_path / 'tag_datafiles/mtat_tags.npy'))


with open(_datafiles_path / 'tag_datafiles/msd_50tagList.txt') as f:
    MSD_TAGS = f.read().splitlines()

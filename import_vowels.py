import librosa
import numpy as np
import pathlib


def read_vowel(file, sr=None):
    """read vowel data from file
    """
    data, rate = librosa.load(file, sr=sr)

    # discard first 1000 and last 1000 points
    data = data[1000:-1000]

    total_power = np.square(data).sum()
    
    return data/total_power


def load_data(vowel_types, dir, sr):
    """load all available vowel data
    """
    data = []
    label = []
    gender = []
    # load men vowel data
    for vowel in vowel_types:
        files = [f for f in pathlib.Path().glob(dir + "m*" + vowel + ".wav")]
        for file in files:
            data.append(read_vowel(file, sr))
            label.append(vowel)
            gender.append('m')

    # load women vowel data
    for vowel in vowel_types:
        files = [f for f in pathlib.Path().glob(dir + "w*" + vowel + ".wav")]
        for file in files:
            data.append(read_vowel(file))
            label.append(vowel)
            gender.append('w')

    return data, label, gender
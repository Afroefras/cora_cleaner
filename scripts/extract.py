import librosa
import numpy as np
from pathlib import Path


def load_mfcc(sound_path: str, duration: int, sr: int) -> np.array:
    # use kaiser_fast technique for faster extraction
    X, sr = librosa.load(sound_path, sr=sr, duration=duration, res_type="kaiser_fast")

    dur = librosa.get_duration(y=X, sr=sr)
    if round(dur) < duration:
        input_length = sr * duration
        X = librosa.util.fix_length(data=X, size=input_length)

    # extract normalized mfcc feature from data
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=25)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.array(mfcc).reshape([-1, 1])

    return mfcc


def load_all_sounds(soundfiles_path: str, duration: int = 10, sr: int = 22050) -> list:
    data = []
    soundfiles_path = Path(soundfiles_path)
    wav_files = soundfiles_path.glob("**/*.wav")
    mp3_files = soundfiles_path.glob("**/*.mp3")

    for sound_path in list(wav_files) + list(mp3_files):
        mfcc = load_mfcc(str(sound_path), duration, sr)
        data.append(mfcc)
    return data


def load_heart_noised(clean_dir: str, noised_dir: str) -> list:
    clean_dir = Path(clean_dir)
    noised_dir = Path(noised_dir)

    noises = noised_dir.glob("**/*.mp3")
    noises = list(map(str, noises))

    paths = {}
    tensors = []
    for clean_path in clean_dir.glob("**/*.wav"):
        clean_name = clean_path.stem

        noises_related = filter(lambda x: clean_name in x, noises)
        noises_related = list(noises_related)
        paths[str(clean_path)] = noises_related

        clean_tensor = load_mfcc(clean_path, duration=10, sr=22050)
        for noise_path in noises_related:
            noise_tensor = load_mfcc(noise_path, duration=10, sr=22050)

            tensors.append((clean_tensor, noise_tensor))

    return paths, tensors


# paths_tensors = load_heart_noised(
#     clean_dir="data/heart_sound",
#     noised_dir="data/heart_noised",
# )

# import pickle

# FILE_PATH = 'data/tensors/paths_tensors_2023-08-06.xz'
# # Store data (serialize)
# with open(FILE_PATH, 'wb') as handle:
#     pickle.dump(paths_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

import librosa
import numpy as np
from pathlib import Path

def load_mfcc(sound_path: str, duration: int, sr: int) -> np.array:
    # use kaiser_fast technique for faster extraction
    X, sr = librosa.load(
        sound_path, sr=sr, duration=duration, res_type="kaiser_fast"
    )

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


def load_heart_noised(clean_dir: str, noised_dir: str) -> np.array:
    clean_dir = Path(clean_dir)
    noised_dir = Path(noised_dir)

    for clean in clean_dir.glob("**/*.wav"):
        clean_name = clean.stem

        
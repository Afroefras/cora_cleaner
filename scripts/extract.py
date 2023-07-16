import librosa
import numpy as np
from pathlib import Path


def load_file_data(files_path: str, duration: int = 10, sr: int = 22050):
    files_path = Path(files_path)
    input_length = sr * duration

    data = []
    for sound_file in files_path.glob("*.wav"):
        # use kaiser_fast technique for faster extraction
        X, sr = librosa.load(
            str(sound_file), sr=sr, duration=duration, res_type="kaiser_fast"
        )
        dur = librosa.get_duration(y=X, sr=sr)

        # pad audio file same duration
        if round(dur) < duration:
            X = librosa.util.fix_length(data=X, size=input_length)

        # extract normalized mfcc feature from data
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=25)
        mfccs = np.mean(mfccs.T, axis=0)

        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data

a = load_file_data("data/heart_sound/val/unhealthy")
print(a)
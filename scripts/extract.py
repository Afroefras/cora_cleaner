import librosa
import numpy as np
from pathlib import Path


def load_sounds(soundfiles_path: str, duration: int = 10, sr: int = 22050):
    soundfiles_path = Path(soundfiles_path)
    input_length = sr * duration

    data = []
    wav_files = soundfiles_path.glob("**/*.wav")
    mp3_files = soundfiles_path.glob("**/*.mp3")

    for sound_file in list(wav_files) + list(mp3_files):
        # use kaiser_fast technique for faster extraction
        X, sr = librosa.load(
            str(sound_file), sr=sr, duration=duration, res_type="kaiser_fast"
        )

        # pad audio file same duration
        dur = librosa.get_duration(y=X, sr=sr)
        if round(dur) < duration:
            X = librosa.util.fix_length(data=X, size=input_length)

        # extract normalized mfcc feature from data
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=25)
        mfccs = np.mean(mfccs.T, axis=0)

        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data

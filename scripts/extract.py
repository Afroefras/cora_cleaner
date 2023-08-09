import librosa
import numpy as np
from pathlib import Path


def load_heart_noised_dict(clean_dir: str, noised_dir: str) -> list:
    clean_dir = Path(clean_dir)
    noised_dir = Path(noised_dir)

    noises = noised_dir.glob("**/*.wav")
    noises = list(map(str, noises))

    paths = {}
    for clean_path in clean_dir.glob("**/*.wav"):
        clean_name = clean_path.stem

        noises_related = filter(lambda x: clean_name in x, noises)
        noises_related = list(noises_related)
        paths[str(clean_path)] = noises_related

    return paths

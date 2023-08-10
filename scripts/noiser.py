from pathlib import Path
from pydub import AudioSegment
from random import choices, randint


def noiser(audio_path: str, noise_path: str, noise_louder: int) -> AudioSegment:
    audio = AudioSegment.from_file(audio_path, format="wav")
    noise = AudioSegment.from_file(noise_path, format="wav")
    noise += noise_louder
    noised = audio.overlay(noise, position=0, loop=True)
    return noised


def noise_dir(audios_dir: str, noises_dir: str, output_dir: str, k_noises: int) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    noises = Path(noises_dir).glob("**/*.wav")
    noises = list(noises)

    for audio in Path(audios_dir).glob("**/*.wav"):
        audio = str(audio)
        choosed_noises = choices(noises, k=k_noises)

        for noise in choosed_noises:
            noise_louder = -randint(10, 20)
            noised = noiser(audio, noise, noise_louder)

            new_audio_name = audio.replace(audios_dir, "")
            new_audio_name = new_audio_name.replace("/", "-")
            new_audio_name = new_audio_name.replace(".wav", "")

            noise_name = noise.stem
            new_audio_name += f"_{noise_name}noise_{noise_louder}dB_louder.wav"

            noised.export(output_dir.joinpath(new_audio_name), format="wav")


# noise_dir(
#     audios_dir="data/heart_sound_test_small",
#     noises_dir="data/hospital_noise",
#     output_dir="data/heart_noised_test_small",
#     k_noises=5,
# )

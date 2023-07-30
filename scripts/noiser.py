from pydub import AudioSegment


def noiser(audio_dir: str, noise_dir: str, audio_louder: int) -> AudioSegment:
    audio = AudioSegment.from_file(audio_dir, format="wav")
    noise = AudioSegment.from_file(noise_dir, format="wav")

    audio += audio_louder
    mixed = audio.overlay(noise, position=0)

    return mixed


mixed = noiser(
    audio_dir="data/heart_sound/train/healthy/a0007.wav",
    noise_dir="data/hospital_noise/seg_1.wav",
    audio_louder=0,
)

mixed.export("output.mp3", format="mp3")

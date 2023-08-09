import numpy as np
import librosa.display
from torch import Tensor
import matplotlib.pyplot as plt

def plot_audio_sample(audio_data, title):
    """
    Grafica un registro de audio de tamaño 25x1, su espectrograma y su MFCC.

    Args:
        audio_data (torch.Tensor): Datos de audio de tamaño 25x1.
        title (str): Título para el gráfico.
    """
    plt.figure(figsize=(12, 7))

    # Convertir los datos de audio a formato de punto flotante
    audio_data_float = Tensor(audio_data).float()

    # Gráfico de la forma de onda
    plt.subplot(2, 2, 1)
    plt.plot(audio_data_float)
    plt.title("Forma de Onda - " + title)
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # Espectrograma
    plt.subplot(2, 1, 2)
    plt.subplot(2, 1, 2)
    spectrogram = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data_float.squeeze().numpy())), ref=np.max
    )
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma - " + title)

    # MFCC
    plt.subplot(2, 2, 2)
    mfcc = librosa.feature.mfcc(
        y=audio_data_float.squeeze().numpy(), sr=22050
    )
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title("MFCC - " + title)

    plt.tight_layout()
    plt.show()

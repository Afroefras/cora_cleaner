import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from torch import Tensor, no_grad


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
    mfcc = librosa.feature.mfcc(y=audio_data_float.squeeze().numpy(), sr=22050)
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title("MFCC - " + title)

    plt.tight_layout()
    plt.show()


def plot_prediction_comparison(model, dataloader, index):
    model.eval()
    with no_grad():
        batch = next(iter(dataloader))
        clean = batch[0][index]
        dirty = batch[-1][index]

        reconstructed = model(dirty.unsqueeze(0))  # Añadir dimensión de batch
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        axs[0].plot(dirty.squeeze())
        axs[0].set_title("Audio Sucio")
        
        axs[1].plot(clean.squeeze())
        axs[1].set_title("Audio Limpio")
        
        axs[2].plot(reconstructed.squeeze())
        axs[2].set_title("Audio Reconstruido")
        
        plt.tight_layout()
        plt.show()
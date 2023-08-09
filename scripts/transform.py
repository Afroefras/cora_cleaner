import torch
import librosa
import torchaudio


def create_spectrogram(audio, sample_rate, n_fft=400, hop_length=160, win_length=400):
    """
    Crea el espectrograma de un audio.

    Args:
        audio (torch.Tensor): Tensor que representa el audio.
        sample_rate (int): Tasa de muestreo del audio.
        n_fft (int): Tamaño de la ventana para la transformada de Fourier de tiempo corto.
        hop_length (int): Desplazamiento entre ventanas sucesivas en muestras.
        win_length (int): Tamaño de la ventana de análisis en muestras.

    Returns:
        torch.Tensor: Espectrograma del audio.
    """
    specgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return specgram(audio).reshape(1, 1, -1)


def compute_mfcc(audio, sample_rate, n_mfcc=13):
    """
    Calcula los coeficientes cepstrales de frecuencia mel (MFCC) de un audio.

    Args:
        audio (numpy.ndarray): Señal de audio.
        sample_rate (int): Tasa de muestreo del audio.
        n_mfcc (int): Número de coeficientes MFCC a calcular.

    Returns:
        numpy.ndarray: Coeficientes MFCC calculados.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = torch.Tensor(mfcc)
    return mfcc.reshape(1, 1, -1)


def spec_n_mfcc(audio, sample_rate):
    spec = create_spectrogram(audio, sample_rate)
    mfcc = compute_mfcc(audio.numpy(), sample_rate)

    joined = torch.cat((audio, spec, mfcc), dim=2)
    return joined
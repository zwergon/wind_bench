import numpy as np
import matplotlib.mlab as mp


def compute_psd(values, DT, normalized=False):
    """
    Compute Power Spectral Density

    Parameters
    ----------
    values : numpy.ndarray Array of values
    DT : double sampling in time
    normalized : boolean default True Normalize values before plot

    Returns
    -------
    Pxx : 1-D array Values for the power spectrum
    freqs : 1-D array Frequencies corresponding to the elements in Pxx

    See Also
    --------
    matplotlib.mlab.psd : PSD computation description
    """
    FMAX = 5.0
    NFFT = int(np.power(2, 9))
    FS = 1.0 / DT

    if normalized:
        valuesTmp = (values - np.min(values)) / (np.max(values) - np.min(values))
    else:
        valuesTmp = values

    # (Pxx, freqs) = mp.psd(valuesTmp, NFFT, FS, detrend='mean', window=np.hanning(NFFT), noverlap=NFFT/2, pad_to=5*NFFT)
    (Pxx, freqs) = mp.psd(
        valuesTmp, NFFT, FS, detrend="mean", window=np.hanning(NFFT), noverlap=NFFT // 2
    )

    freqs_filtered = np.where(freqs < FMAX)

    return Pxx[freqs_filtered], freqs[freqs_filtered]

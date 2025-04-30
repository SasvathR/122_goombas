import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

# ─── QAM MODULATION ────────────────────────────────────────────────────────
def qammod(symbols, M):
    m = int(np.sqrt(M))
    if m*m != M:
        raise ValueError("qammod: M must be a square number")
    real_idx = symbols % m
    imag_idx = symbols // m
    re = 2*real_idx - (m-1)
    im = 2*imag_idx - (m-1)
    constellation = re + 1j*im
    constellation /= np.sqrt((2/3)*(M-1))
    return constellation                                                                      

# ─── OFDM TX/RX ─────────────────────────────────────────────────────────────
def ofdm_transmitter(data_symbols, Nfft, Ncp):
    ofdm_sym  = np.fft.ifft(data_symbols, Nfft)
    return np.concatenate([ofdm_sym[-Ncp:], ofdm_sym])                         

def ofdm_receiver(rx_signal, Nfft, Ncp):
    return np.fft.fft(rx_signal[Ncp:], Nfft)                                                             

# ─── IMPAIRMENTS USED BY sim.py ─────────────────────────────────────────────
def add_phase_noise(x, std):
    return x * np.exp(1j*std*np.random.randn(*x.shape))

def add_quantization_noise(x, bits):
    # your fixed‐range quantizer from earlier
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x.copy()
    step = 2*max_abs/(2**bits-1)
    real_q = np.round(x.real/step)*step
    imag_q = np.round(x.imag/step)*step
    return real_q + 1j*imag_q   
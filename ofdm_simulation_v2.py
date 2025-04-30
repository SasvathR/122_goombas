#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

def plot_input_signal(x, fs, title="Input Signal"):
    """
    Plots real, imaginary, magnitude, and power spectrum of the transmit signal x.
    
    x      : 1‐D complex numpy array (time domain)
    fs     : sampling rate in Hz
    title  : figure title prefix
    """
    N = len(x)
    t = np.arange(N)/fs
    
    # Time‐domain real/imag
    plt.figure(figsize=(10,6))
    plt.plot(t, np.real(x), label="Real(x)")
    plt.plot(t, np.imag(x), label="Imag(x)", alpha=0.7)
    plt.title(f"{title} (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Envelope (magnitude)
    plt.figure(figsize=(10,4))
    plt.plot(t, np.abs(x))
    plt.title(f"{title} Envelope |x(t)|")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Power Spectral Density (Welch)
    from scipy.signal import welch
    f, Pxx = welch(x, fs=fs, nperseg=1024, return_onesided=True)
    plt.figure(figsize=(10,4))
    plt.semilogy(f, Pxx)
    plt.title(f"{title} PSD (Welch)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def validate_awgn(snr_db=20, N=100000):
    x = np.ones(N, dtype=complex)
    y = add_awgn(x, snr_db)
    noise = y-x
    measured = 10*np.log10(np.mean(np.abs(x)**2)/np.mean(np.abs(noise)**2))
    print(f"[AWGN] target {snr_db} dB, measured {measured:.8f} dB")
    # histogram of noise
    plt.figure()
    plt.hist(np.real(noise), bins=100, alpha=0.5, label="Re(noise)")
    plt.hist(np.imag(noise), bins=100, alpha=0.5, label="Im(noise)")
    plt.title("AWGN Noise Histogram"); plt.legend(); plt.grid(True)
    plt.show()

def validate_phase_noise(std=0.01, N=100000):
    x = np.ones(N, dtype=complex)
    y = add_phase_noise(x, std)
    phase_err = np.angle(y)
    measured = np.std(phase_err)
    print(f"[Phase] target σ={std:.8f}, measured σ={measured:.8f}")
    plt.figure()
    plt.hist(phase_err, bins=100)
    plt.title("Phase Noise Histogram"); plt.grid(True)
    plt.show()

def validate_quant(bits=10, N=100000):
    x = np.random.uniform(-1,1,N)
    y = add_quantization_noise(x, bits)
    err = y-x
    lsb2 = 1/(2**bits)/2
    print(f"[Quant] max error={np.max(np.abs(err)):.8f}, LSB/2={lsb2:.8f}")
    plt.figure()
    plt.hist(err, bins=100)
    plt.title("Quantization Error Histogram"); plt.grid(True)
    plt.show()

def validate_jakes(f_D, fs, N=2000, M=16):
    h = jakes_fading(f_D, fs, N, M)

    # compute and normalize empirical autocorrelation
    max_lag = 200
    R_emp = np.array([np.mean(h[:N-l] * np.conj(h[l:])) for l in range(max_lag)])
    R_emp /= R_emp[0]            # <-- normalize to 1 at lag 0

    tau  = np.arange(max_lag) / fs
    R_th = j0(2*np.pi*f_D * tau)

    mse = np.mean((R_emp - R_th)**2)
    print(f"[Jakes] normalized autocorr MSE = {mse:.3e}")

    # Plot only the normalized autocorr vs theory
    plt.figure()
    plt.plot(tau, R_emp,   label="Empirical (norm)")
    plt.plot(tau, R_th, '--', label="J0 theoretical")
    plt.title("Jakes Fading Autocorrelation (normalized)")
    plt.xlabel("Lag (s)"); plt.ylabel("Autocorr")
    plt.legend(); plt.grid(True)
    plt.show()
# ─── QAM MODULATION ────────────────────────────────────────────────────────────
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

# ─── OFDM TX/RX ─────────────────────────────────────────────────────────────────
def ofdm_transmitter(data_symbols, Nfft, Ncp):
    ofdm_sym  = np.fft.ifft(data_symbols, Nfft)
    return np.concatenate([ofdm_sym[-Ncp:], ofdm_sym])

def ofdm_receiver(rx_signal, Nfft, Ncp):
    return np.fft.fft(rx_signal[Ncp:], Nfft)

# ─── NOISE & IMPAIRMENTS ───────────────────────────────────────────────────────
def add_awgn(x, snr_db):
    P = np.mean(np.abs(x)**2)
    snr = 10**(snr_db/10)
    var = P/snr
    noise = np.sqrt(var/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x+noise
# replace add_awgn with this version:
def add_awgn_fixed_noise(tx_sig, snr_db):
    """
    Add AWGN assuming transmit power p_tx and fixed noise floor:
      noise_var = p_tx / snr_lin
    After fading, rx power = |h|^2 * p_tx, so SNR_rx = |h|^2 × snr_lin
    """
    p_tx = np.mean(np.abs(tx_sig)**2)
    snr_lin = 10**(snr_db/10)
    noise_var = p_tx / snr_lin
    noise = np.sqrt(noise_var/2) * (
        np.random.randn(*tx_sig.shape) + 1j*np.random.randn(*tx_sig.shape)
    )
    return tx_sig + noise, noise_var 

def add_phase_noise(x, std):
    return x * np.exp(1j*std*np.random.randn(*x.shape))

def add_quantization_noise(x, bits):
    levels = 2**bits
    return np.round(x*levels)/levels

# ─── FIXED JAKES FADING (unit‐power) ────────────────────────────────────────────
def jakes_fading(f_D, fs, N, M=16):
    """
    Sum‐of‐sinusoids Jakes model with unit average power.
    """
    t     = np.arange(N)/fs
    beta  = 2*np.pi*np.random.rand(M)
    alpha = np.linspace(0, 2*np.pi, M, endpoint=False)
    h     = np.zeros(N, dtype=complex)
    for m in range(M):
        h += np.exp(1j*(2*np.pi*f_D*t*np.cos(alpha[m]) + beta[m]))
    # normalize to unit power
    return h/np.sqrt(np.mean(np.abs(h)**2))

def jakes_rayleigh_channel(tx, f_D, delays, gains_db, fs=1e6):
    gains = 10**(np.array(gains_db)/20)
    rx = np.zeros_like(tx, dtype=complex)
    for g,τ in zip(gains, delays):
        h_t = jakes_fading(f_D,fs,len(tx))
        d = int(np.round(τ*fs))
        delayed = np.concatenate([np.zeros(d,complex), tx[:-d]]) if d>0 else tx
        rx += g*h_t*delayed
    return rx

# ─── TEST DATA GENERATORS ──────────────────────────────────────────────────────
def prbs_seq(degree, length):
    reg = np.ones(degree, dtype=int)
    out = np.zeros(length, dtype=int)
    for i in range(length):
        fb = reg[-1]^reg[0]
        out[i] = reg[-1]
        reg[1:] = reg[:-1]
        reg[0] = fb
    return out

def zc_seq(root, N_zc):
    n = np.arange(N_zc)
    return np.exp(-1j*np.pi*root*n*(n+1)/N_zc)

# ─── CHANNEL ESTIMATION ────────────────────────────────────────────────────────
def ls_channel_est(rx_p, tx_p):
    return rx_p/tx_p

def lmmse_channel_est(rx_p, tx_p, R_h, sigma2):
    X = np.diag(tx_p)
    inv_term = np.linalg.inv(X@R_h@X.conj().T + sigma2*np.eye(len(tx_p)))
    W = R_h@X.conj().T@inv_term
    return W@rx_p

# ─── PA & EVM ──────────────────────────────────────────────────────────────────
def pa_nonlinear(x, a1, a3):
    return a1*x + a3*(np.abs(x)**2)*x

def compute_evm(rx_s, tx_s):
    return np.sqrt(np.mean(np.abs(rx_s-tx_s)**2)/np.mean(np.abs(tx_s)**2))

def plot_pa_evm(Nfft, Ncp, mod_order, snr_list):
    for snr_db in snr_list:
        for label,a3 in [("Ideal PA",0),("Moderate",0.01),("High",0.05)]:
            # generate PRBS OFDM symbol
            bits = prbs_seq(7,Nfft*int(np.log2(mod_order)))
            syms_int = bits.reshape(Nfft,2)[:,0]*2+bits.reshape(Nfft,2)[:,1]
            mod_syms = qammod(syms_int,mod_order)
            tx = ofdm_transmitter(mod_syms,Nfft,Ncp)
            pa = pa_nonlinear(tx,1,a3)
            rx = add_awgn(pa,snr_db)
            rx_syms = ofdm_receiver(rx,Nfft,Ncp)
            evm = compute_evm(rx_syms, mod_syms)
            evm_db = 20*np.log10(evm)
            print(f"SNR={snr_db} dB | {label} PA: EVM={evm_db:.5f} dB")
            plt.figure()
            plt.scatter(np.real(rx_syms),np.imag(rx_syms),alpha=0.7,label="Rx")
            plt.scatter(np.real(mod_syms),np.imag(mod_syms),marker='x',c='r',label="Ideal")
            plt.title(f"{label} PA @ {snr_db} dB, EVM={evm_db:.5f} dB")
            plt.legend(); plt.grid(True)
    plt.show()

# ─── TEST DATA & JAKES VALIDATION ──────────────────────────────────────────────
def plot_test_data(Nfft, mod_order):
    bits = prbs_seq(7,200)
    plt.figure()
    plt.plot(bits,'o-');plt.title("PRBS bits (200)");plt.grid(True)
    # QPSK from PRBS
    syms_int = prbs_seq(7,Nfft*2).reshape(Nfft,2)[:,0]*2 + prbs_seq(7,Nfft*2).reshape(Nfft,2)[:,1]
    qpsk = qammod(syms_int,mod_order)
    plt.figure()
    plt.scatter(np.real(qpsk),np.imag(qpsk));plt.title("QPSK from PRBS");plt.grid(True)
    # ZC autocorr
    zc = zc_seq(5,63)
    ac = np.abs(np.correlate(zc,zc.conj(),mode='full'))
    lags = np.arange(-62,63)
    plt.figure()
    plt.stem(lags,ac)
    plt.title("ZC Autocorr");plt.grid(True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # 1) Define parameters (FFT size, CP, mod order, doppler, etc.)
    # 2) Pre-allocate MSE arrays for LS and LMMSE.
    # 3) Repeat num_mc times:
    #     a) Generate PRBS bits → QAM symbols → OFDM payload (IFFT+CP).
    #     b) (Once only) call plot_input_signal() to verify the raw transmit block.
    #     c) Pass through Jakes channel, phase noise, AWGN, quantization.
    #     d) Demodulate (remove CP + FFT) → rx_syms.
    #     e) Estimate channel (LS and LMMSE) → record MSE against true=1.
    #     f) On first iteration, save diagnostics (tx, rx, h_ls, mod_syms, rx_syms).
    # 4) Print average MSE for LS & LMMSE.
    # 5) Plot the first‐run constellation (rx_syms vs. mod_syms).
    # 6) Call the three “modern diagnostics”:
    #       plot_test_data(), plot_jakes_validation(), plot_pa_evm().
    # parameters
    Nfft,Ncp,mod_order = 64,16,4
    max_doppler = 30; delays=[0,1e-6,3e-6]; gains_db=[0,-3,-6]
    fs=1e6; snr_db=10; phi_std=0.01; qbits=10; num_mc=100

    mse_ls = np.zeros(num_mc); mse_lmmse=np.zeros(num_mc)
    # pick one run for diagnostics
    diag = {}

    for mc in range(num_mc):
        # PRBS payload + OFDM
        bits = prbs_seq(7,Nfft*2)
        data_syms = bits.reshape(Nfft,2)[:,0]*2+bits.reshape(Nfft,2)[:,1]
        mod_syms = qammod(data_syms,mod_order)
        # 2) OFDM modulate (IFFT + CP)
        tx_payload = ofdm_transmitter(mod_syms, Nfft, Ncp)

        # —— PLOT INPUT SIGNAL FOR FIRST ITERATION ONLY ——
        if mc == 0:
            plot_input_signal(tx_payload, fs,
                              title="PRBS-OFDM Payload (no channel)")
        
        # 3) (no preamble here) send tx_payload through channel
        tx = tx_payload
        # channel + noise
        rx = jakes_rayleigh_channel(tx,max_doppler,delays,gains_db,fs)
        rx = add_phase_noise(rx,phi_std)
        rx = add_awgn(rx,snr_db)
        rx = add_quantization_noise(rx,qbits)
        # receive
        rx_syms = ofdm_receiver(rx,Nfft,Ncp)
        # est
        pilot = qammod(np.arange(Nfft)%mod_order,mod_order)
        h_ls = ls_channel_est(rx_syms,pilot)
        sigma2 = 10**(-snr_db/10)
        h_lmmse = lmmse_channel_est(rx_syms,pilot,np.eye(Nfft),sigma2)
        mse_ls[mc]=np.mean(np.abs(1-h_ls)**2)
        mse_lmmse[mc]=np.mean(np.abs(1-h_lmmse)**2)
        if mc==0:
            diag.update(tx=tx,rx=rx,mod_syms=mod_syms,rx_syms=rx_syms,h_ls=h_ls)
    print("Avg MSE LS:",mse_ls.mean(), "LMMSE:",mse_lmmse.mean())

    plt.figure(); 
    plt.scatter(np.real(diag['rx_syms']),np.imag(diag['rx_syms']),label="Rx")
    plt.scatter(np.real(diag['mod_syms']),np.imag(diag['mod_syms']),marker='x',label="Tx")
    plt.title("Constellation");plt.legend();plt.grid(True)

    
    # pull out what you saved in diag:
    rx_raw    = diag['rx_syms']    # complex vector length Nfft
    tx_ideal  = diag['mod_syms']   # original QAM symbols
    h_ls      = diag['h_ls']       # your LS estimate of the channel

    # equalize:
    rx_eq = rx_raw / h_ls

    plt.figure()
    plt.scatter(rx_eq.real, rx_eq.imag, label="Equalized Rx", alpha=0.7)
    plt.scatter(tx_ideal.real, tx_ideal.imag, marker='x', c='r', label="Tx Ideal")
    plt.title("Constellation (Post-Equalization)")
    plt.xlabel("In-Phase"); plt.ylabel("Quadrature")
    plt.legend(); plt.grid(True)
    plt.show()

    

    # call diagnostics
    plot_test_data(Nfft,mod_order)
    plot_pa_evm(Nfft,Ncp,mod_order,[20,10])
    plt.show()

if __name__ == "__main__":
    main()
    validate_awgn(20)
    validate_phase_noise(0.01)
    validate_quant(10)
    validate_jakes(30,1e6,5000,16)

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from vehicle_mobility import ChannelFading
import ofdm_simulation_v3 as phy  # existing OFDM simulation module
from numba import njit
import zlib
from math import ceil
import time

# ——— PARAMETERS (tune these to match sim.py globals) ———
Ptx_dBm     = 12.0                           # transmit power [dBm]
Ptx_W       = 10**((Ptx_dBm-30)/10)          # → [W]
quant_bits  = [8, 12]                        # test 8-bit & 12-bit quantizers
pn_std_list = [0.0, 0.01, 0.025, 0.05]  
                                            # no PN vs your std 

phase_noise_default = 0.01

# CSMA-CA parameters
difs_slots = 2.5      # DIFS in slot units
slot_time = getattr(phy, 'SLOT_TIME', 20e-6)  # 50 μs slot default
cw_min = 15           # min contention window (slots)
cw_max = 1024       # max contention window (slots)

symbol_dur = getattr(phy, 'SYMBOL_DURATION', 64e-7)


# constants (e.g. at top of your file or inside __init__)
c   = 3e8                    # speed of light [m/s]
fc  = 5.9e9                  # carrier frequency [Hz] (DSRC band)
lam = c / fc                 # wavelength [m]
d0  = 1.0                    # reference distance [m]
alpha = 2.0                  # free-space exponent

k_B   = 1.38e-23       # Boltzmann’s constant
T0    = 290            # room temperature [K]
B     = 10e6           # DSRC channel bandwidth [Hz]
nf_dB = 5.0
noise_var = k_B*T0*B * (10**(nf_dB/10))   # total noise power in 10 MHz

Ptx_dBm = 10.0                          # choose your module’s typical output
Ptx_W   = 10**((Ptx_dBm-30)/10)         # convert dBm→watts

# precompute FSPL at d0
fspl_d0 = (lam / (4*np.pi*d0))**2

PA_models   = [
    ("Ideal",    0.00),
    ("Moderate", 0.01),
    ("Severe",   0.05),
]

def plot_constellations(df, rx_id=None, max_plots=4):
    """
    Fast constellation plots for up to `max_plots` packets.
    """
    # Debug: inspect DataFrame
    print(f"DataFrame shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    if 'rx' in df.columns:
        print("Unique rx IDs:", df['rx'].unique())
    else:
        print("No 'rx' column in df!")

    # Filter by receiver if requested
    if rx_id is not None:
        print(f"Filtering for rx_id = {rx_id!r}")
        df = df[df['rx'] == rx_id]

    if df.empty:
        print("No data to plot (after filtering).")
        return

    # now pick earliest packets
    sel = df.nsmallest(max_plots, 'time')

    # 2) pick the earliest max_plots rows without a full sort
    sel = df.nsmallest(max_plots, 'time')

    times    = sel['time'].values
    snrs     = sel['snr_dB'].values
    tx_syms  = sel['tx_syms'].tolist()
    rx_syms  = sel['rx_syms'].tolist()

    n_plots = len(times)
    if n_plots == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(4*n_plots, 4))
    for i, (t, snr, tx, rx) in enumerate(zip(times, snrs, tx_syms, rx_syms)):
        ax = plt.subplot(1, n_plots, i+1)
        ax.scatter(rx.real, rx.imag, s=20, alpha=0.6, label='Rx')
        ax.scatter(tx.real, tx.imag, marker='x', c='r', label='Tx')
        ax.set_title(f"t={t:.3f}s\nSNR={snr:.1f} dB")
        ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.grid(True); ax.axis('equal')
        if i == 0:
            ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_constellations_by_pa(df, rx_id=None, pa_values=None, max_plots=3):
    """
    For each pa_a3 in pa_values, plot up to max_plots constellations
    (ideal vs received) at the same time, so you can compare how increasing
    nonlinearity warps the clusters.
    """
    if pa_values is None:
        pa_values = sorted(df['pa_a3'].unique())

    fig, axes = plt.subplots(len(pa_values), max_plots,
                             figsize=(4*max_plots,4*len(pa_values)),
                             squeeze=False)

    for i, a3 in enumerate(pa_values):
        sub = df[df['pa_a3']==a3]
        if rx_id is not None:
            sub = sub[sub['rx']==rx_id]
        sub = sub.nsmallest(max_plots, 'time')
        for j, (_, row) in enumerate(sub.iterrows()):
            ax = axes[i][j]
            tx = row['tx_syms']; rx = row['rx_syms']
            ax.scatter(rx.real, rx.imag, s=20, alpha=0.6, label='Rx')
            ax.scatter(tx.real, tx.imag, marker='x', c='r', label='Tx')
            ax.set_title(f"a3={a3:.3f}\ntime={row['time']:.3f}s\nSNR={row['snr_dB']:.1f} dB")
            ax.axis('equal'); ax.grid(True)
            if i==0 and j==0:
                ax.legend(loc='upper right')
    fig.suptitle(f"Constellations for Rx={rx_id}, varying PA nonlinearity", y=1.02)
    plt.tight_layout()
    plt.show()


def prbs_seq(degree, length):
    """Max-length PRBS for payload generation."""
    reg = np.ones(degree, dtype=int)
    out = np.zeros(length, dtype=int)
    for i in range(length):
        fb = int(reg[-1] ^ reg[0])
        out[i] = reg[-1]
        reg[1:] = reg[:-1]
        reg[0]  = fb
    return out

def add_awgn_integrated(x):
    """
    Add complex AWGN based on the physical noise floor noise_var.
    Matches exactly the manual block:
      noise = sqrt(noise_var/2)*(randn + j randn)
      rx_sig = clean_sig + noise
    """
    noise = np.sqrt(noise_var/2) * (
        np.random.randn(*x.shape)
      + 1j*np.random.randn(*x.shape)
    )
    return x + noise

def pa_nonlinear(x, a1, a3):
    """Cubic PA model: y = a1·x + a3·|x|²·x."""
    return a1*x + a3*(np.abs(x)**2)*x

def compute_evm(rx_syms, tx_syms):
    """RMS EVM (linear) → dB."""
    evm_lin = np.mean(np.abs(rx_syms - tx_syms)**2) / np.mean(np.abs(tx_syms)**2)
    return 20*np.log10(np.sqrt(evm_lin))
def plot_pa_evm(Nfft, Ncp, mod_order):
    """
    Plot EVM for different PA nonlinearities, phase-noise and quantizers,
    using the fixed noise floor from add_awgn_integrated(sig).
    """
    # 1) prep IFFT normalization to remove the √Nfft gain
    ifft_norm = 1/np.sqrt(Nfft)

    # 2) generate one OFDM symbol (PRBS payload)
    n_bits   = Nfft * int(np.log2(mod_order))
    bits     = prbs_seq(7, n_bits)
    syms_int = bits.reshape(Nfft,2)[:,0]*2 + bits.reshape(Nfft,2)[:,1]
    tx_syms  = phy.qammod(syms_int, mod_order)

    plt.figure(figsize=(8,5))
    for label, a3 in PA_models:
        # build & scale TX waveform
        tx_bb = phy.ofdm_transmitter(tx_syms, Nfft, Ncp) * ifft_norm
        tx_bb *= np.sqrt(Ptx_W / np.mean(np.abs(tx_bb)**2))

        # apply PA nonlinearity
        pa_bb = pa_nonlinear(tx_bb, 1.0, a3)

        # add AWGN (fixed noise floor)
        rx_awgn = add_awgn_integrated(pa_bb)

        for pn_std in pn_std_list:
            # add phase noise if requested
            rx_pn = phy.add_phase_noise(rx_awgn, pn_std) if pn_std>0 else rx_awgn

            for qb in quant_bits:
                # add quantization
                rx_q = phy.add_quantization_noise(rx_pn, qb) if qb>0 else rx_pn

                # OFDM RX
                rx_syms = phy.ofdm_receiver(rx_q, Nfft, Ncp)

                # compute EVM
                evm_db = compute_evm(rx_syms, tx_syms)

                # plot
                legend = f"{label}, PN={pn_std:.2f}, Q={qb}b"
                plt.scatter(label, evm_db, s=80, alpha=0.7, label=legend)

    plt.title("PA & Impairment EVM (fixed noise floor)")
    plt.xlabel("PA Model")
    plt.ylabel("EVM (dB)")
    plt.grid(True)
    # one legend entry per unique label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()


def int_to_bits(x, width):
    return [int(b) for b in format(x & ((1 << width) - 1), f'0{width}b')]

def compute_fcs_from_bits(bit_array: list[int]) -> list[int]:
    # Pad bit array to make it a multiple of 8
    if len(bit_array) % 8 != 0:
        padding = 8 - (len(bit_array) % 8)
        bit_array += [0] * padding

    # Pack bits into bytes (LSB-first per byte)
    byte_array = bytearray()
    for i in range(0, len(bit_array), 8):
        byte = sum((bit_array[i + j] << j) for j in range(8))
        byte_array.append(byte)

    # Compute CRC32 with standard 802.11 parameters
    crc = zlib.crc32(byte_array) ^ 0xFFFFFFFF
    crc_bytes = crc.to_bytes(4, byteorder='little')

    # Convert CRC bytes to bit array (LSB-first per byte)
    fcs_bits = []
    for byte in crc_bytes:
        fcs_bits.extend([(byte >> i) & 1 for i in range(8)])

    return fcs_bits

def generate_mac():
    mac = [0x02,               # Locally administered, unicast
           random.randint(0x00, 0x7f),
           random.randint(0x00, 0xff),
           random.randint(0x00, 0xff),
           random.randint(0x00, 0xff),
           random.randint(0x00, 0xff)]
    return sum([mac[i] << ((5 - i) * 8) for i in range(len(mac))])

class Device:
    """
    MAC-layer device implementing simplified CSMA-CA.
    States: IDLE -> DIFS -> BACKOFF -> TRANSMIT -> IDLE
    """
    def __init__(self, v, env, mac, transmission_interval):
        self.vehicle = v
        self.n_collision = 0
        self.mac_addr = generate_mac()
        self.env = env
        self.mac = mac
        self.state = 'IDLE'
        self.difs = difs_slots
        self.backoff = 0
        env.process(self.tick())  # start FSM
        self.t_int = transmission_interval
        self.next_transmit = self.env.now + self.t_int

        self.frag_num = 0

    def start_difs(self):
        self.state = 'DIFS'
        self.difs = difs_slots

    def start_backoff(self):
        self.state = 'BACKOFF'
        self.backoff = random.randint(0, min(cw_max, (cw_min + 1) * (2 ** self.n_collision) - 1))

    def tick(self):
        """Run every us to update MAC state."""
        while True:
            if self.env.now < self.next_transmit and self.backoff <= 0:
                yield self.env.timeout(self.next_transmit - self.env.now)
            else:
                yield self.env.timeout(slot_time)
            busy = self.mac.channel_busy
            if self.backoff <= 0:
                if busy:
                    # collision
                    self.n_collision += 1
                    self.start_backoff()
                    continue
                else:
                    # start transmission
                    # 387 bits message, length is 39u
                    msg = self.vehicle.generate_bsm(self.env.now)
                    l = int(np.ceil((len(msg) + 98) / 10))
                    data = [
                        int_to_bits(0x0888, 16), # Frame Control
                        int_to_bits(l, 16), # Duration
                        int_to_bits(0x01005E7FFFFA, 6), # Reciever, broadcast
                        int_to_bits(self.mac_addr, 6), # Transmitter
                        int_to_bits(0xFFFFFFFFFFFF, 6), # BSSID
                        int_to_bits(self.frag_num, 16), # Seq #
                        msg # data
                    ]
                    bits = [int(bit) for field in data for bit in field]
                    bits += compute_fcs_from_bits(bits)
                    self.env.process(self.mac.handle_transmission(self.env.now, self.vehicle.id, bits))
                    self.state = 'IDLE'
                    self.n_collision = 0
                    self.next_transmit = self.env.now + self.t_int
                    self.frag_num += 1

            if self.state == 'IDLE' and not busy:
                self.start_difs()
            elif self.state == 'DIFS':
                if busy:
                    self.start_difs()
                else:
                    self.difs -= 1
                    if self.difs <= 0:
                        self.start_backoff()
            elif self.state == 'BACKOFF' and not busy:
                self.backoff -= 1


@njit
def popcount(x):
    """Count set bits in integer x."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

@njit
def fast_conv_encode_nb(bits, K, G1o, G2o):
    """
    Numba-accelerated rate-1/2 convolutional encoder (terminated flush).
    bits: 1D uint8 array of 0/1.
    Returns: 1D uint8 array of coded bits length = 2*(len(bits) + K - 1).
    """
    n = bits.size
    total = 2 * (n + K - 1)
    out = np.empty(total, dtype=np.uint8)
    st = 0  # state holds last K-1 bits
    idx = 0

    # Encode message bits
    for i in range(n):
        b = bits[i]
        inp = (b << (K - 1)) | st
        o1 = popcount(inp & G1o) & 1
        o2 = popcount(inp & G2o) & 1
        out[idx] = o1
        out[idx + 1] = o2
        idx += 2
        # Update state: shift right, insert b at MSB of K-1 bits
        st = (st >> 1) | (b << (K - 2))

    # Flush (K-1) zero bits
    for _ in range(K - 1):
        inp = st  # since b=0, inp = st
        o1 = popcount(inp & G1o) & 1
        o2 = popcount(inp & G2o) & 1
        out[idx] = o1
        out[idx + 1] = o2
        idx += 2
        st = st >> 1  # shift right

    return out

@njit
def fast_viterbi_decode_nb(coded, msg_len, K, G1o, G2o):
    """
    Numba-accelerated hard-decision Viterbi decoder for rate-1/2 conv code.
    coded: 1D uint8 array of coded bits.
    msg_len: original message bit length.
    Returns: 1D uint8 array of decoded bits length = msg_len.
    """
    n_states = 1 << (K - 1)
    # Build trellis
    next_state = np.zeros((n_states, 2), np.int32)
    output     = np.zeros((n_states, 2), np.uint8)
    for st in range(n_states):
        for b in (0, 1):
            inp = (b << (K - 1)) | st
            o1  = popcount(inp & G1o) & 1
            o2  = popcount(inp & G2o) & 1
            ns  = ((st >> 1) | (b << (K - 2))) & (n_states - 1)
            next_state[st, b] = ns
            output[st, b]     = (o1 << 1) | o2

    n_steps = coded.size // 2
    INF     = 10**9
    pm      = np.full(n_states, INF, np.int32)
    pm[0]   = 0
    prev    = np.zeros((n_steps, n_states), np.int32)

    # Forward pass
    for t in range(n_steps):
        r0 = coded[2*t]
        r1 = coded[2*t+1]
        new_pm = np.full(n_states, INF, np.int32)
        for st in range(n_states):
            for b in (0, 1):
                ns   = next_state[st, b]
                outb = output[st, b]
                exp0 = (outb >> 1) & 1
                exp1 = outb & 1
                metric = pm[st] + (r0 != exp0) + (r1 != exp1)
                if metric < new_pm[ns]:
                    new_pm[ns]  = metric
                    prev[t, ns] = st
        pm = new_pm

    # Traceback
    state = 0
    best  = pm[0]
    for s in range(1, n_states):
        if pm[s] < best:
            best  = pm[s]
            state = s

    dec = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        dec[t] = (state >> (K - 2)) & 1
        state   = prev[t, state]

    return dec[:msg_len]

class ConvCoder:
    """
    Numba-backed convolutional coder with matching encode/decode.
    """
    def __init__(self, K=7, G1o=0o133, G2o=0o171):
        self.K    = K
        self.G1o  = G1o
        self.G2o  = G2o

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.astype(np.uint8)
        return fast_conv_encode_nb(bits, self.K, self.G1o, self.G2o)

    def decode(self, coded: np.ndarray, msg_len: int) -> np.ndarray:
        coded = coded.astype(np.uint8)
        return fast_viterbi_decode_nb(coded, msg_len, self.K, self.G1o, self.G2o)

class MACSim:
    """
    Orchestrates CSMA-CA MAC and OFDM PHY.
    Logs events, stores trajectories, and provides visualization.
    Implements:
      - Generate multiple OFDM symbols per transmission
      - Compute true BER by bit comparison
    """
    def __init__(self, env, vehicles, pa_a3=0.0):
        self.env = env
        self.do_conv = True
        self.vehicles = {v.id: v for v in vehicles}
        self.devices = []
        self.initial_positions = {v.id: v.position.copy() for v in vehicles}
        self.channel_busy = False
        self.fading = {}    # per-link ChannelFading instances
        self.log = []       # records of PHY events
        # PHY parameters
        self.Nfft = getattr(phy, 'Nfft', 64)
        self.Ncp  = getattr(phy, 'Ncp', 16)
        self.mod_order = getattr(phy, 'mod_order', 4)
        self.bits_per_symbol = int(np.log2(self.mod_order))
        self.pa_a3 = pa_a3   # cubic‐term coefficient for this run
        # instantiate MAC devices
        for v in vehicles:
            dev = Device(v, env, self, 0.1)
            self.devices.append(dev)
            env.timeout(random.randint(1, 50) * 1e-3)

        self.coder = ConvCoder()

    def qam_demod(self, rx_syms):
        """Demodulate QAM symbols back to integer indices."""
        m = int(np.sqrt(self.mod_order))
        real = np.real(rx_syms)
        imag = np.imag(rx_syms)
        # reverse mapping
        scale = np.sqrt((2/3)*(self.mod_order-1))
        re_idx = np.clip(((real * scale) + (m-1))/2, 0, m-1)
        im_idx = np.clip(((imag * scale) + (m-1))/2, 0, m-1)
        real_idx = re_idx.round().astype(int)
        imag_idx = im_idx.round().astype(int)
        return (imag_idx * m + real_idx).astype(int)


    def handle_transmission(self, t, tx_id, bits):
        """
        Called when a device's backoff expires (packet transmission).
        Applies distance-based path loss, small-scale fading, and fixed-noise AWGN.
        """
        self.channel_busy = True
        tx = self.vehicles[tx_id]
        # Move all vehicles to current time
        for v in self.vehicles.values():
            v.move_to(t)
        # Prepare fading and Doppler per link
        for rx_id, rx in self.vehicles.items():
            if rx_id == tx_id: continue
            f_d = abs(rx.doppler_shift(tx))
            key = (tx_id, rx_id)
            if key not in self.fading:
                self.fading[key] = ChannelFading(f_max=f_d, init_time=t)
            self.fading[key].f_max = f_d
                # Simulation parameters

        
        # Path-loss model reference distance and exponent
        d0 = getattr(phy, 'REF_DISTANCE', 1.0)      # 1 meter reference
        alpha = getattr(phy, 'PATHLOSS_EXP', 2.0)    # free-space → 2.0
        # Transmit to each receiver
        if self.do_conv: 
            scale = 2 
        else: 
            scale = 1
        num_packets = ceil(len(bits) / (self.Nfft * self.bits_per_symbol // scale))

        for rx_id, rx in self.vehicles.items():
            if rx_id == tx_id: continue
            chan = self.fading[(tx_id, rx_id)]

            per_symbol_avg_snr = []

            total_err = 0
            total_bits = 0
            # Compute distance-based path loss
            disp = rx.position - tx.position
            dist = np.linalg.norm(disp)

            if dist > d0:
                pl_lin = fspl_d0 * (d0/dist)**alpha
            else:
                pl_lin = 1.0

            all_bits = []
            for k in range(num_packets):

                sym_time = t + k * symbol_dur
                h = chan.evolve_to(sym_time)

                # 1) QAM symbols
                data_bin = np.array(bits[k * self.bits_per_symbol * self.Nfft // scale: (k+1) * self.bits_per_symbol * self.Nfft // scale])
                required_len = self.bits_per_symbol * self.Nfft // scale
                if len(data_bin) < required_len:
                    pad_len = required_len - len(data_bin)
                    data_bin = np.concatenate([data_bin, np.zeros(pad_len, dtype=np.uint8)])
                else:
                    data_bin = data_bin[:required_len]

                if self.do_conv:
                    data_enc = self.coder.encode(data_bin)
                    data_syms = data_enc.reshape(-1, self.bits_per_symbol)
                else:
                    data_syms = data_bin.reshape(-1, self.bits_per_symbol)

                powers = 2 ** np.arange(self.bits_per_symbol)[::-1]
                data_syms = data_syms.dot(powers)
                mod_syms = phy.qammod(data_syms, self.mod_order)
                # 2) OFDM TX
                # 2a) generate and normalize the OFDM burst
                tx_sig = phy.ofdm_transmitter(mod_syms, self.Nfft, self.Ncp)
                # undo the IFFT’s √Nfft gain so “unit power” really means unit power
                tx_sig /= np.sqrt(self.Nfft)

                tx_sig = pa_nonlinear(tx_sig, 1.0, self.pa_a3)

                
                # normalize the OFDM burst to unit power, then scale to Ptx:
                tx_sig *= np.sqrt(Ptx_W / np.mean(np.abs(tx_sig)**2))

                # 3) Apply path loss and fading
                clean_sig = np.sqrt(pl_lin) * h * tx_sig


                rx_sig = add_awgn_integrated(clean_sig)
                rx_sig = phy.add_phase_noise(rx_sig, phase_noise_default)

                # quantization only degrades at like 4 or 2 bits
                rx_sig = phy.add_quantization_noise(rx_sig, 12)

                # 5) equalize each subcarrier by the known flat‐fading h
                # ─── frequency‐domain snapshots ───────────────────────
                rxFFT    = phy.ofdm_receiver(rx_sig,    self.Nfft, self.Ncp)
                cleanFFT = phy.ofdm_receiver(clean_sig, self.Nfft, self.Ncp)

                # ─── rest of RX chain ─────────────────────────────────
                
                # 6) equalize the *noisy* FFT by known channel
                rx_eq = rxFFT / (h * np.sqrt(pl_lin)/np.sqrt(self.Nfft))

                # # clean_eq = cleanFFT / h
                # clean_eq    = cleanFFT / h            # = √pl_lin × mod_syms
                # ideal_syms  = clean_eq / np.sqrt(pl_lin)  # = mod_syms

                # 7) record the very first symbol’s constellation **after** all impairments
                if k == 0:
                    first_tx_syms = mod_syms.copy()
                    first_rx_syms = rx_eq.copy()

                # if k == 0:
                #     first_tx_syms = ideal_syms.copy()
                #     first_rx_syms = rx_eq.copy()

                # 8) demodulate & BER count
                rx_idx  = self.qam_demod(rx_eq)
                rx_bits = ((rx_idx[:,None] >> np.arange(self.bits_per_symbol-1, -1, -1)) 
                        & 1).astype(int).reshape(-1)
                if self.do_conv:
                    rx_bits = self.coder.decode(rx_bits, msg_len=required_len)
                errs     = np.sum(data_bin != rx_bits)
                total_err += errs
                total_bits+= required_len

                # 9) (optional) symbol‐by‐symbol SNR logging
                noiseFFT    = rxFFT - cleanFFT
                snr_sub_lin = np.abs(cleanFFT)**2 / (np.abs(noiseFFT)**2 + 1e-12)
                per_symbol_avg_snr.append(np.mean(10*np.log10(snr_sub_lin)))

                all_bits += [int(b) for b in rx_bits]

                # count symbol errors by popcount over bits_per_symbol bits
                errs = np.sum(data_bin != rx_bits)
                total_err += errs
                total_bits += self.Nfft * self.bits_per_symbol

            all_bits = all_bits[:len(bits)]
            data = all_bits[0:-32]
            fcs = compute_fcs_from_bits(data)
            if fcs == all_bits[-32:]:
                rx.receive_bsm(data[66:-4])
            # Compute BER and instantaneous SNR
            ber = total_err / total_bits
            self.log.append({
            'time'        : t,                                    # packet start time
            'tx'          : tx_id,                                # transmitter ID
            'rx'          : rx_id,                                # receiver ID
            'distance_m'  : dist,                                # link distance [m]
            'path_loss'   : pl_lin,                              # linear path‐loss
            'pa_a3'       : self.pa_a3,                           # <<< new
            'snr_dB'      : float(np.mean(per_symbol_avg_snr)),  # mean per‐symbol SNR [dB]
            'snr_vec'     : per_symbol_avg_snr,                  # list of per‐symbol SNRs [dB]
            'total_err'   : total_err,                           # total bit‐errors in packet
            'total_bits'  : total_bits,                          # total bits in packet
            'ber'         : ber,                                 # bit‐error‐rate = total_err/total_bits
            'success'     : (ber < 1e-3),                        # simple success flag
            'tx_syms'     : first_tx_syms,                       # first‐symbol ideal QAM constellation
            'rx_syms'     : first_rx_syms                        # first‐symbol received constellation
        })
        # Release channel after packet
        yield self.env.timeout(symbol_dur * num_packets)
        self.channel_busy = False

    def run(self, until):
        """Run the environment and return log DataFrame."""
        self.env.run(until=until)
        df = pd.DataFrame(self.log)
        self.sim_duration = until
        return df

    def plot_trajectories(self, n_points=100):
        """Plot each vehicle's path as sampled points over time."""
        times = np.linspace(0, self.sim_duration, n_points)
        plt.figure(figsize=(6,6))
        for vid, v0 in self.initial_positions.items():
            vel = self.vehicles[vid].velocity
            traj = np.outer(times, vel) + v0
            plt.plot(traj[:,0], traj[:,1], '-o', markersize=3, label=vid)
        plt.title('Vehicle Trajectories')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend(); plt.grid(True); plt.show()


    def plot_ber_vs_time(self, df):
        """Plot BER over time for each receiver."""
        if df.empty:
            print("No data to plot BER.")
            return
        plt.figure()
        for rx_id in df['rx'].unique():
            sub = df[df['rx'] == rx_id]
            plt.plot(sub['time'], sub['ber'], label=f'RX {rx_id}')
        plt.title('BER vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('BER')
        plt.yscale('log')
        plt.legend(); plt.grid(True); plt.show()

    def plot_snr_vs_time(self, df):
        """
        Plot mean per-packet SNR and per-symbol SNR scatter for each receiver.

        Expects df to have columns:
          - 'time'   : packet start time [s]
          - 'snr_dB' : mean SNR over symbols [dB]
          - 'snr_vec': list/array of per-symbol SNRs [dB]
        """
        if df.empty or not {'time','snr_dB','snr_vec'}.issubset(df.columns):
            print("No data to plot SNR."); 
            return

        plt.figure(figsize=(8,4))

        for rx in sorted(df['rx'].unique()):
            sub = df[df['rx'] == rx]

            # 1) plot the flat, per-packet average SNR
            plt.plot(sub['time'], sub['snr_dB'],
                     '-o', label=f'RX {rx} avg SNR', markersize=4)

            # 2) scatter each symbol's SNR around that packet time
            for _, row in sub.iterrows():
                packet_t = row['time']
                vec      = np.asarray(row['snr_vec'])
                # symbol times spaced by symbol_dur
                times    = packet_t + np.arange(vec.size) * symbol_dur
                plt.scatter(times, vec,
                            s=10, alpha=0.3, label=f'RX {rx} symbols' 
                                                  if _==sub.index[0] else "")

        plt.title("Per-Subcarrier SNR vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("SNR (dB)")
        plt.grid(True)
        plt.legend(loc='upper right', ncol=2)
        plt.tight_layout()
        plt.show()
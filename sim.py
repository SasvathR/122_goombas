
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from vehicle_mobility import ChannelFading
import ofdm_simulation_v2 as phy  # existing OFDM simulation module
from numba import njit

# CSMA-CA parameters
difs_slots = 2.5      # DIFS in slot units
slot_time = getattr(phy, 'SLOT_TIME', 20e-6)  # 50 μs slot default
cw_min = 15           # min contention window (slots)
cw_max = 1024       # max contention window (slots)

class Device:
    """
    MAC-layer device implementing simplified CSMA-CA.
    States: IDLE -> DIFS -> BACKOFF -> TRANSMIT -> IDLE
    """
    def __init__(self, id, env, mac):
        self.id = id
        self.n_collision = 0
        self.env = env
        self.mac = mac
        self.state = 'IDLE'
        self.difs = difs_slots
        self.backoff = 0
        env.process(self.tick())  # start FSM
        self.n_transmit = 1e2

    def start_difs(self):
        self.state = 'DIFS'
        self.difs = difs_slots

    def start_backoff(self):
        self.state = 'BACKOFF'
        self.backoff = random.randint(0, min(cw_max, (cw_min + 1) * (2 ** self.n_collision) - 1))

    def tick(self):
        """Run every us to update MAC state."""
        while True:
            yield self.env.timeout(slot_time)
            busy = self.mac.channel_busy
            if self.backoff <= 0:
                if busy:
                    # collision
                    self.n_collision += 1
                    self.start_backoff()
                    continue
                else:
                    # if self.n_transmit < 0:
                    #     continue
                    # schedule transmission process
                    self.env.process(self.mac.handle_transmission(self.env.now, self.id))
                    self.state = 'IDLE'
                    self.n_collision = 0
                    self.n_transmit -= 1
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

class ConvCoder:
    def __init__(self, K=7, G1o=0o133, G2o=0o171):
        self.K = K
        self.G1o = G1o
        self.G2o = G2o
        self.n_states = 1 << (K - 1)
        self.G1 = np.array([(G1o >> i) & 1 for i in range(K - 1, -1, -1)], dtype=np.uint8)[::-1]
        self.G2 = np.array([(G2o >> i) & 1 for i in range(K - 1, -1, -1)], dtype=np.uint8)[::-1]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return fast_conv_encode_nb(bits, self.K, self.G1o, self.G2o)

    def decode(self, coded: np.ndarray, msg_len: int) -> np.ndarray:
        return fast_viterbi_decode_nb(coded, msg_len, self.K, self.G1o, self.G2o)


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
    def __init__(self, env, vehicles, symbols_per_packet=100):
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
        self.symbols_per_packet = symbols_per_packet
        # instantiate MAC devices
        for v in vehicles:
            dev = Device(v.id, env, self)
            self.devices.append(dev)
            env.timeout(random.randint(1, 15) * 1e-6)

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
    
    def generate_bsm(self, vehicle, t):
        """Generate a binary BSM message from vehicle state."""
        def int_to_bits(x, width):
            return [int(b) for b in format(x & ((1 << width) - 1), f'0{width}b')]

        lat = int(37.7749 * 1e7)
        lon = int(-122.4194 * 1e7)
        elev = int(15)
        speed = int(np.linalg.norm(vehicle.velocity) * 100)
        heading = int(np.degrees(np.arctan2(vehicle.velocity[1], vehicle.velocity[0])) * 100) % 36000
        ax, ay, az = [int(x * 1000) for x in [0.5, 0.0, -9.8]]
        size_l, size_w = int(4.5 * 100), int(1.8 * 100)
        fields = [
            int_to_bits(0x20, 8),                # Message ID
            int_to_bits(0x3F29A78B, 32),         # Temporary ID
            int_to_bits(int(t * 1000), 32),      # Timestamp in ms
            int_to_bits(lat, 32),
            int_to_bits(lon, 32),
            int_to_bits(elev, 16),
            [1],                                 # Accuracy = high
            int_to_bits(speed, 16),
            int_to_bits(heading, 16),
            int_to_bits(ax, 16),
            int_to_bits(ay, 16),
            int_to_bits(az, 16),
            int_to_bits(0, 16),                  # Yaw Rate
            int_to_bits(0, 8),                   # Brake Status
            int_to_bits(size_l, 16),
            int_to_bits(size_w, 16),
        ]
        bits = np.array([bit for field in fields for bit in field], dtype=np.uint8)
        return bits


    def handle_transmission(self, t, tx_id):
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
        symbol_dur = getattr(phy, 'SYMBOL_DURATION', 1e-4)
        # Path-loss model reference distance and exponent
        d0 = getattr(phy, 'REF_DISTANCE', 1.0)      # 1 meter reference
        alpha = getattr(phy, 'PATHLOSS_EXP', 2.0)    # free-space → 2.0
        # Transmit to each receiver
        for rx_id, rx in self.vehicles.items():
            if rx_id == tx_id: continue
            chan = self.fading[(tx_id, rx_id)]
            total_err = 0
            total_bits = 0
            # Compute distance-based path loss
            disp = rx.position - tx.position
            dist = np.linalg.norm(disp)
                        # Compute distance-based path loss (d0 reference)
            pl_lin = (d0/dist)**alpha if dist>0 else 1.0
            # Noise settings
            snr_db = getattr(phy, 'snr_db', 10)
            snr_lin = 10**(snr_db / 10)
            # Multiple OFDM symbols
            for k in range(self.symbols_per_packet):
                sym_time = t + k * symbol_dur
                h = chan.evolve_to(sym_time)
                # 1) QAM symbols
                if self.do_conv: 
                    scale = 2 
                else: 
                    scale = 1
                data_bin = self.generate_bsm(tx, sym_time)
                required_len = self.bits_per_symbol * self.Nfft // scale
                if len(data_bin) < required_len:
                    pad_len = required_len - len(data_bin)
                    data_bin = np.concatenate([data_bin, np.zeros(pad_len, dtype=np.uint8)])
                else:
                    data_bin = data_bin[:required_len]
                original_bits = data_bin.copy()

                if self.do_conv:
                    data_enc = self.coder.encode(data_bin)
                    data_syms = data_enc.reshape(-1, self.bits_per_symbol)
                else:
                    data_syms = data_bin.reshape(-1, self.bits_per_symbol)

                powers = 2 ** np.arange(self.bits_per_symbol)[::-1]
                data_syms = data_syms.dot(powers)
                mod_syms = phy.qammod(data_syms, self.mod_order)
                # 2) OFDM TX
                tx_sig = phy.ofdm_transmitter(mod_syms, self.Nfft, self.Ncp)
                # 3) Apply path loss and fading
                rx_sig = np.sqrt(pl_lin) * h * tx_sig
                # 4) Impairments: phase noise + AWGN + quantization
                rx_sig = phy.add_awgn(rx_sig, snr_db)
                rx_sig = phy.add_phase_noise(rx_sig, getattr(phy, 'phase_noise_std', 0.01))
                # rx_sig = phy.add_quantization_noise(rx_sig, getattr(phy, 'quant_bits', 10))
                # 5) OFDM RX & equalize
                rx_syms = phy.ofdm_receiver(rx_sig, self.Nfft, self.Ncp)
                rx_eq = rx_syms / h
                # 6) Demod & count bit errors
                rx_idx = self.qam_demod(rx_eq)
                # print(len(rx_idx))
                rx_bits = ((rx_idx[:, None] >> np.arange(self.bits_per_symbol - 1, -1, -1)) & 1).astype(int).reshape(-1)
                # print(rx_bits)
                # print(self.coder.decode(self.coder.encode(np.array([0,1,0,1,0,1,0,1,0,1])), msg_len=10))

                if self.do_conv:
                    rx_bits = self.coder.decode(rx_bits, msg_len=required_len)

                if k == 0 and rx_id == list(self.vehicles.keys())[1]:  # pick one receiver
                    mismatch = np.sum(original_bits != rx_bits)
                    print(f"[DEBUG] TX→RX BSM bit errors: {mismatch} / {required_len}")
                    print("Original (first 40):", original_bits[:40])
                    print("Decoded  (first 40):", rx_bits[:40])

                # count symbol errors by popcount over bits_per_symbol bits
                errs = np.sum(data_bin != rx_bits)
                total_err += errs * self.bits_per_symbol
                total_bits += self.Nfft * self.bits_per_symbol
            # Compute BER and instantaneous SNR
            ber = total_err / total_bits
            inst_snr_lin = pl_lin * abs(h)**2 * snr_lin
            inst_snr_db = 10 * np.log10(inst_snr_lin) if inst_snr_lin > 0 else -np.inf
            # Log
            self.log.append({
                'time': t,
                'tx': tx_id,
                'rx': rx_id,
                'snr_dB': inst_snr_db,
                'ber': ber,
                'success': ber < 1e-1
            })
        # Release channel after packet
        yield self.env.timeout(self.symbols_per_packet * symbol_dur)
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
            print("No data to plot.")
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
        """Plot SNR over time for each receiver."""
        if df.empty or 'snr_dB' not in df.columns:
            print('No data to plot.'); return
        plt.figure()
        for rx in df['rx'].unique():
            sub = df[df['rx'] == rx]
            plt.plot(sub['time'], sub['snr_dB'], label=f'RX {rx}')
        plt.title('SNR vs Time')
        plt.xlabel('Time (s)'); plt.ylabel('SNR (dB)'); plt.legend(); plt.grid(True); plt.show()
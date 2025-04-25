
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from vehicle_mobility import ChannelFading
import ofdm_simulation_v2 as phy  # existing OFDM simulation module

# CSMA-CA parameters
difs_slots = 2.5      # DIFS in slot units
slot_time = getattr(phy, 'SLOT_TIME', 20e-6)  # 50 μs slot default
cw_min = 15           # min contention window (slots)
cw_max = 1024       # max contention window (slots)

def int_to_bits(x, width):
    """Convert integer to binary list."""
    return [int(b) for b in format(x, f'0{width}b')]

class Device:
    """
    MAC-layer device implementing simplified CSMA-CA.
    States: IDLE -> DIFS -> BACKOFF -> TRANSMIT -> IDLE
    """
    def __init__(self, id, env, mac):
        self.id = id
        self.us = 0
        self.n_collision = 0
        self.env = env
        self.mac = mac
        self.state = 'IDLE'
        self.difs = difs_slots
        self.backoff = 0
        env.process(self.tick())  # start FSM
        self.n_transmit = 4

    def start_difs(self):
        self.state = 'DIFS'
        self.difs = difs_slots

    def start_backoff(self):
        self.state = 'BACKOFF'
        self.backoff = random.randint(0, min(cw_max, (cw_min + 1) * (2 ** self.n_collision) - 1))

    def tick(self):
        """Run every us to update MAC state."""
        while True:
            yield self.env.timeout(1e-6)
            if self.us < slot_time:
                self.us += 1e-6
                continue
            else:
                self.us = 0
            busy = self.mac.channel_busy
            if self.backoff <= 0:
                if busy:
                    # collision
                    self.n_collision += 1
                    self.start_backoff()
                    continue
                else:
                    if self.n_transmit < 0:
                        continue
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

        self.K = 7

        self.G1 = int_to_bits(0o133, self.K)  # [1, 0, 1, 1, 0, 1, 1]
        self.G1o = 0o133
        self.G2 = int_to_bits(0o171, self.K)  # [1, 1, 1, 1, 0, 0, 1]
        self.G2o = 0o171

        self.TRELLIS = self.precompute_trellis()

    def precompute_trellis(self):
        n_states = 2**(self.K - 1)
        trellis = np.zeros((n_states, 2), dtype=[('next_state', np.uint8), ('output', np.uint8)])
        for state in range(n_states):
            for bit in [0, 1]:
                input_state = (bit << (self.K - 1)) | state
                o1 = bin(input_state & self.G1o).count('1') % 2
                o2 = bin(input_state & self.G2o).count('1') % 2
                next_state = ((state >> 1) | (bit << (self.K - 2))) & (n_states - 1)
                trellis[state, bit] = (next_state, (o1 << 1) | o2)
        return trellis

    def conv_encode(self, msg_bits):
        """
        Rate 1/2 convolutional encoder (constraint length 7).
        Generators: G1 = 133 (octal), G2 = 171 (octal).
        """

        msg = np.array(msg_bits, dtype=int)
        padded = np.concatenate([msg, np.zeros(self.K- 1, dtype=int)])  # tail bits
        encoded = []

        for i in range(len(msg)):
            window = padded[i:i + self.K]
            out1 = np.sum(self.G1 * window) % 2
            out2 = np.sum(self.G2 * window) % 2
            encoded.extend([out1, out2])

        return np.array(encoded, dtype=int)
    
    def conv_decode(self, encoded_bits):
        n_states = 2**(self.K - 1)
        n_steps = len(encoded_bits) // 2

        path_metrics = np.full((n_states,), np.inf)
        path_metrics[0] = 0
        prev_states = np.zeros((n_steps, n_states), dtype=np.uint8)

        for t in range(n_steps):
            r = encoded_bits[2*t:2*t+2]
            new_metrics = np.full((n_states,), np.inf)

            for state in range(n_states):
                for bit in [0, 1]:
                    next_state = self.TRELLIS[state, bit]['next_state']
                    expected = self.TRELLIS[state, bit]['output']
                    expected_bits = [(expected >> 1) & 1, expected & 1]
                    metric = np.sum(r != expected_bits)
                    new_metric = path_metrics[state] + metric

                    if new_metric < new_metrics[next_state]:
                        new_metrics[next_state] = new_metric
                        prev_states[t, next_state] = state

            path_metrics = new_metrics

        # Traceback
        state = np.argmin(path_metrics)
        decoded = []
        for t in reversed(range(n_steps)):
            prev = prev_states[t, state]
            decoded_bit = (state >> (self.K - 2)) & 1
            decoded.append(decoded_bit)
            state = prev

        return np.array(decoded[::-1][:self.Nfft], dtype=int)

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
            snr_db = getattr(phy, 'snr_db', 20)
            snr_lin = 10**(snr_db / 10)
            # Multiple OFDM symbols
            for k in range(self.symbols_per_packet):
                sym_time = t + k * symbol_dur
                h = chan.evolve_to(sym_time)
                # 1) QAM symbols
                data_bin = np.random.randint(0, 2, self.bits_per_symbol * self.Nfft)
                data_enc = data_bin # self.conv_encode(data_bin)
                # print(data_enc)
                data_syms = data_enc.reshape(-1, self.bits_per_symbol)
                powers = 2 ** np.arange(self.bits_per_symbol)[::-1]
                data_syms = data_syms.dot(powers)
                mod_syms = phy.qammod(data_syms, self.mod_order)
                # 2) OFDM TX
                tx_sig = phy.ofdm_transmitter(mod_syms, self.Nfft, self.Ncp)
                # 3) Apply path loss and fading
                rx_sig = np.sqrt(pl_lin) * h * tx_sig
                # 4) Impairments: phase noise + AWGN + quantization
                # rx_sig = phy.add_phase_noise(rx_sig, getattr(phy, 'phase_noise_std', 0.01))
                # rx_sig = phy.add_awgn_fixed_noise(rx_sig, snr_db)
                # rx_sig = phy.add_quantization_noise(rx_sig, getattr(phy, 'quant_bits', 10))
                # 5) OFDM RX & equalize
                rx_syms = phy.ofdm_receiver(rx_sig, self.Nfft, self.Ncp)
                rx_eq = rx_syms / h
                # 6) Demod & count bit errors
                rx_idx = self.qam_demod(rx_eq)
                # print(len(rx_idx))
                rx_bits = ((rx_idx[:, None] >> np.arange(self.bits_per_symbol - 1, -1, -1)) & 1).astype(int).reshape(-1)
                # print(rx_bits)
                print(self.conv_decode(self.conv_encode(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))))

                rx_bin = rx_bits # self.conv_decode(rx_bits)
                # count symbol errors by popcount over bits_per_symbol bits
                errs = np.sum(data_bin != rx_bin)
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
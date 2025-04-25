
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

class Device:
    """
    MAC-layer device implementing simplified CSMA-CA.
    States: IDLE -> DIFS -> BACKOFF -> TRANSMIT -> IDLE
    """
    def __init__(self, id, env, mac):
        self.id = id
        self.env = env
        self.mac = mac
        self.state = 'IDLE'
        self.cw = cw_min
        self.difs = difs_slots
        self.backoff = 0
        env.process(self.tick())  # start FSM

    def start_difs(self):
        self.state = 'DIFS'
        self.difs = difs_slots

    def start_backoff(self):
        self.state = 'BACKOFF'
        self.backoff = random.randint(0, self.cw)
        # exponential backoff for next attempt
        self.cw = min(self.cw * 2, cw_max)

    def tick(self):
        """Run every slot_time to update MAC state."""
        while True:
            yield self.env.timeout(slot_time)
            busy = self.mac.channel_busy
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
                if self.backoff <= 0:
                    # schedule transmission process
                    self.env.process(self.mac.handle_transmission(self.env.now, self.id))
                    self.state = 'IDLE'

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
                data_syms = np.random.randint(0, self.mod_order, self.Nfft)
                mod_syms = phy.qammod(data_syms, self.mod_order)
                # 2) OFDM TX
                tx_sig = phy.ofdm_transmitter(mod_syms, self.Nfft, self.Ncp)
                # 3) Apply path loss and fading
                rx_sig = np.sqrt(pl_lin) * h * tx_sig
                # 4) Impairments: phase noise + AWGN + quantization
                rx_sig = phy.add_phase_noise(rx_sig, getattr(phy, 'phase_noise_std', 0.01))
                rx_sig = phy.add_awgn_fixed_noise(rx_sig, snr_db)
                rx_sig = phy.add_quantization_noise(rx_sig, getattr(phy, 'quant_bits', 10))
                # 5) OFDM RX & equalize
                rx_syms = phy.ofdm_receiver(rx_sig, self.Nfft, self.Ncp)
                rx_eq = rx_syms / h
                # 6) Demod & count bit errors
                rx_idx = self.qam_demod(rx_eq)
                # compute bit-wise errors only over symbols_per_packet bits
                diff = rx_idx ^ data_syms
                # count bit errors by popcount over bits_per_symbol bits
                errs = sum(bin(int(d)).count('1') for d in diff)
                total_err += errs
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
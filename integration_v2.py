import numpy as np
import random
import simpy
import pandas as pd
import matplotlib.pyplot as plt
from vehicle_mobility import Vehicle, ChannelFading, plot_vehicles_enriched
import ofdm_simulation_v2 as phy  # existing OFDM simulation module
from scipy.special import erfc

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
                rx_sig = phy.add_awgn(rx_sig, snr_db)
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

# === Validation & Testing Suite ===


def test_static_channel():
    """
    Test 1: Static channel (f_D=0), high SNR → expect BER = 0
    """
    print("[TEST] Static channel: no fading, high SNR -> BER should be 0")
    # Monkey-patch ChannelFading.evolve_to to always return unity gain
    orig_evolve = ChannelFading.evolve_to
    ChannelFading.evolve_to = lambda self, t: 1+0j
    # Disable impairments: AWGN, phase noise, quantization
    orig_awgn = phy.add_awgn
    orig_phase = phy.add_phase_noise
    orig_quant = phy.add_quantization_noise
    phy.add_awgn = lambda x, snr: x
    phy.add_phase_noise = lambda x, std: x
    phy.add_quantization_noise = lambda x, bits: x
    # Run simulation with static channel
    vehicles = [Vehicle('T1',0,0,0,0), Vehicle('R1',100,0,0,0)]
    env = simpy.Environment()
    mac = MACSim(env, vehicles, symbols_per_packet=200)
    df = mac.run(until=0.1)
    max_ber = df['ber'].max() if not df.empty else None
    print(f"Max BER on static channel: {max_ber}")
    assert max_ber == 0 or np.isnan(max_ber), "Static channel BER non-zero!"
    # Restore patched functions
    ChannelFading.evolve_to = orig_evolve
    phy.add_awgn = orig_awgn
    phy.add_phase_noise = orig_phase
    phy.add_quantization_noise = orig_quant
    ChannelFading.evolve_to = orig_evolve


def test_awgn_benchmark():
    """
    Test 2: AWGN-only benchmark vs theoretical QPSK BER
    """
    print("[TEST] AWGN-only benchmark: measure BER and compare to theory")
    # Generate known QPSK symbols
    M = 4; Nsym=10000
    bits = np.random.randint(0, M, Nsym)
    syms = phy.qammod(bits, M)
    for snr_db in [0, 5, 10, 15, 20]:
        rx = phy.add_awgn(syms, snr_db)
        # simple demodulation by nearest neighbor
        rx_inds = np.array([np.argmin(np.abs(s - syms)) for s in rx])
        ber = np.mean(rx_inds != bits)
        # theoretical BER for QPSK over AWGN
        snr = 10**(snr_db/10)
        ber_theory = 0.5 * erfc(np.sqrt(snr))
        print(f"SNR={snr_db} dB: Meas BER={ber:.5e}, Theoretical≈{ber_theory:.5e}")


def test_fading_autocorr():
    """
    Test 3: Fading autocorrelation vs J0 function
    """
    print("[TEST] Jakes fading autocorrelation validation")
    phy.validate_jakes(f_D=30, fs=1e6, N=5000, M=16)


def test_doppler():
    """
    Test 4: Doppler calculation correctness
    """
    print("[TEST] Doppler shift calculation check")
    tx = Vehicle('TX',0,0,30,0)
    rx = Vehicle('RX',100,0,0,0)
    f_c = tx.freq
    computed = rx.doppler_shift(tx)
    disp = tx.position - rx.position
    vrel = np.dot(tx.velocity - rx.velocity, disp/np.linalg.norm(disp))
    expected = (vrel/tx.c) * f_c
    print(f"Computed fD={computed:.1f} Hz, Expected ≈{expected:.1f} Hz")
    assert np.isclose(abs(computed), abs(expected), atol=1.0), "Doppler magnitude mismatch!"


def test_convergence():
    """
    Test 5: BER convergence with increasing packet length
    """
    print("[TEST] BER convergence vs packet length")
    lengths = [100, 1000, 5000]
    for L in lengths:
        env = simpy.Environment()
        mac = MACSim(env, vehicles, symbols_per_packet=L)
        df = mac.run(until=0.5)
        mean_ber = df['ber'].mean() if not df.empty else None
        print(f"Packet symbols={L}: Mean BER={mean_ber:.5e}")


def test_reproducibility():
    """
    Test 6: Random-seed reproducibility
    """
    print("[TEST] Seed reproducibility check")
    seed=12345
    np.random.seed(seed); random.seed(seed)
    env1 = simpy.Environment()
    mac1 = MACSim(env1, vehicles, symbols_per_packet=200)
    df1 = mac1.run(until=0.2)
    np.random.seed(seed); random.seed(seed)
    env2 = simpy.Environment()
    mac2 = MACSim(env2, vehicles, symbols_per_packet=200)
    df2 = mac2.run(until=0.2)
    same = df1.equals(df2)
    print(f"Reproducible runs identical? {same}")


def test_csma_collision():
    """
    Test 7: CSMA-CA collision scenario
    """
    print("[TEST] CSMA-CA collision behavior")
    # Both devices transmit immediately: zero backoff
    vehicles = [Vehicle('A',0,0,0,0), Vehicle('B',0,0,0,0)]
    env = simpy.Environment()
    mac = MACSim(env, vehicles, symbols_per_packet=10)
    # Force both to new DIFS at t=0
    for dev in mac.devices:
        dev.start_difs()
    df = mac.run(until=0.01)
    print(df[['time','tx','rx']])
    # Check that no two transmissions share identical time stamp
    times = df['time'].values
    unique = len(times)==len(np.unique(times))
    print(f"No simultaneous tx? {unique}")

def simulate_stoplight():
    """
    Stoplight Scenario:
    - Two vehicles approach an intersection at x=50 m.
    - Vehicles stop for 3 s (t=5→8 s) and then resume.
    - Runs the MAC/PHY sim for 10 s, then plots and prints diagnostics.
    """
    print("\n=== Stoplight Scenario ===")
    # Set up two vehicles heading toward x=50 m
    vehicles_sl = [
        Vehicle('V1', -200,   0, 20.0, 0),   # starting at x=0 m, speed 10 m/s
        Vehicle('V2', 0, -200, 0, 20.0)    # starting at x=-20 m, speed 10 m/s
    ]
    env_sl = simpy.Environment()
    mac_sl = MACSim(env_sl, vehicles_sl, symbols_per_packet=500)

    # Stoplight controller process
    def control(env):
        # At t=10.0 s, turn red: vehicles stop
        yield env.timeout(10.0)
        print("Stoplight RED at t=10.0 s")
        for v in vehicles_sl:
            if (v.id == 'V1'):
                v.velocity = np.array([3.0,0])
            else:
                v.velocity = np.array([0, 3.0])
        # At t=30.0 s, turn green: vehicles resume 10 m/s
        yield env.timeout(4.0)
        print("Stoplight1 GREEN at t=14.0 s")
        for v in vehicles_sl:
            if (v.id == 'V1'):
                v.velocity = np.array([20.0,0])
        # At t=30.0 s, turn green: vehicles resume 10 m/s
        yield env.timeout(4.0)
        print("Stoplight2 GREEN at t=18.0 s")
        for v in vehicles_sl:
            if (v.id == 'V2'):
                v.velocity = np.array([0,20.0])


    env_sl.process(control(env_sl))

    # Run the simulation
    df_sl = mac_sl.run(until=20.0)

    # Print first few PHY events
    print("=== PHY Log (Stoplight, first 1000 rows) ===")
    if df_sl.empty:
        print("<No events>")
    else:
        print(df_sl.head(100).to_string(index=False))

    # Plot trajectories and BER
    plot_vehicles_enriched(list(mac_sl.vehicles.values()))
    mac_sl.plot_ber_vs_time(df_sl)
    mac_sl.plot_snr_vs_time(df_sl)

    # Histogram of per-packet BER
    plt.figure()
    plt.hist(df_sl['ber'], bins=20, edgecolor='black')
    plt.title('Per-Packet BER Histogram (Stoplight)')
    plt.xlabel('BER')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def basic_simulation():
    vehicles = [Vehicle('A', 0,0,20,0), Vehicle('B',50,20,-10,0), Vehicle('C',100,40,-20,0)]
    env = simpy.Environment()
    mac = MACSim(env, vehicles, symbols_per_packet=1000)
    df = mac.run(until=1.0)
    # --- Print detailed log and summary ---
    print("=== PHY Event Log (first 10 rows) ===")
    if df.empty:
        print("<No events logged>")
    else:
        print(df.head(10).to_string(index=False))
        print("=== Average BER per Link ===")
        summary = df.groupby(['tx','rx'])['ber'].mean().reset_index()
        summary.columns = ['TX','RX','Mean_BER']

if __name__ == '__main__':
    # basic_simulation()
    # test_static_channel()
    # test_awgn_benchmark()
    # test_fading_autocorr()
    # test_doppler()
    # test_convergence()
    # test_reproducibility()
    # test_csma_collision()
    simulate_stoplight()
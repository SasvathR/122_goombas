import numpy as np
import random
import simpy
import matplotlib.pyplot as plt
from vehicle_mobility import Vehicle, ChannelFading, plot_vehicles_enriched
import ofdm_simulation_v3 as phy  # existing OFDM simulation module
from sim import MACSim
from sim import plot_pa_evm, plot_constellations
from scipy.special import erfc

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
    orig_awgn = phy.add_awgn_fixed_noise
    orig_phase = phy.add_phase_noise
    orig_quant = phy.add_quantization_noise
    phy.add_awgn_fixed_noise = lambda x, snr: x
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
    phy.add_awgn_fixed_noise = orig_awgn
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
        rx = phy.add_awgn_fixed_noise(syms, snr_db)
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
    mac_sl = MACSim(env_sl, vehicles_sl)

    # setattr(phy, 'snr_db', 20)
    # setattr(phy, 'quant_bits', 6)
    # setattr(phy, 'phase_noise_std', 0.01)

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


    print("got here")
    plot_constellations(df_sl, rx_id=1, max_plots=4)
    print("but not here")
    # Histogram of per-packet BER
    plt.figure()
    plt.hist(df_sl['ber'], bins=20, edgecolor='black')
    plt.title('Per-Packet BER Histogram (Stoplight)')
    plt.xlabel('BER')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def simulate_constant_speed():
    """
    Constant Speed Scenario:
    - Two vehicles moving at a constant speed in the same direction
    - Runs the MAC/PHY sim for 10 s, then plots and prints diagnostics.
    """
    print("\n=== Constant Speed Scenario ===")

    # Set up two vehicles heading toward x=50 m
    vehicles_sl = [
        Vehicle('V1', -200,   0, 20.0, 0),   # starting at x=-200 m, speed 10 m/s
        Vehicle('V2', -210, 0, 20.0, 0)    # starting at x=-210 m, speed 10 m/s
    ]
    env_sl = simpy.Environment()
    mac_sl = MACSim(env_sl, vehicles_sl)


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

    print(vehicles_sl[0].packets_received, vehicles_sl[0].packets_received)

    # Histogram of per-packet BER
    plt.figure()
    plt.hist(df_sl['ber'], bins=20, edgecolor='black')
    plt.title('Per-Packet BER Histogram (Stoplight)')
    plt.xlabel('BER')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def simulate_emergency():
    """
    Constant Speed Scenario:
    - Two vehicles moving at a constant speed in the same direction
    - Runs the MAC/PHY sim for 10 s, then plots and prints diagnostics.
    """
    print("\n=== Constant Speed Scenario ===")

    emergency_stopped = {}
    def react_emergency(v, data):
        # if data["emergency"]:
        #     print(data)
        #     # print(v.position, data["lat"] / 1e7, data["lon"] / 1e7)
        # print(data["emergency"], np.array([v.position[0] - data["lat"] / 1e7, v.position[1] - data["lon"] / 1e7]))
        if v.id not in emergency_stopped and data["emergency"] and np.linalg.norm(np.array([v.position[0] - data["lat"] / 1e7, v.position[1] - data["lon"] / 1e7])) < 40:
            emergency_stopped[v.id] = True

    # Set up grid of moving vehicles
    vehicles_sl = [Vehicle(f'V{i}', 100 + i * 10, (i % 2) * 4, 10, 0, receive_f=react_emergency) for i in range(20)] # grid of moving cars
    vehicles_sl.append(Vehicle("Emergency", 0, 0, 40, 0))
    
    env_sl = simpy.Environment()
    mac_sl = MACSim(env_sl, vehicles_sl)

    # Run the simulation
    df_sl = mac_sl.run(until=20.0)

    print(emergency_stopped)
    for v in vehicles_sl:
        print(v.id, v.position)
    # print(vehicles_sl[2].history)

    # Histogram of per-packet BER
    # plt.figure()
    # plt.hist(df_sl['ber'], bins=20, edgecolor='black')
    # plt.title('Per-Packet BER Histogram (Stoplight)')
    # plt.xlabel('BER')
    # plt.ylabel('Count')
    # plt.grid(True)
    # plt.show()

def simulate_far_fast_moving():
    """
    Constant Speed Scenario:
    - Two vehicles moving at a constant speed in the same direction
    - Runs the MAC/PHY sim for 10 s, then plots and prints diagnostics.
    """
    print("\n=== Constant Speed Scenario ===")

    # setattr(phy, 'snr_db', 4)
    # setattr(phy, 'quant_bits', 6)
    # setattr(phy, 'phase_noise_std', 0.05)

    # Set up two vehicles heading toward x=50 m
    vehicles_sl = [
        Vehicle('V1', 0,   0, 0, 0),   # starting at x=-200 m, speed 10 m/s
        Vehicle('V2', -1000, 0, 80.0, 0)    # starting at x=-210 m, speed 10 m/s
    ]
    env_sl = simpy.Environment()
    mac_sl = MACSim(env_sl, vehicles_sl)


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

    print(vehicles_sl[0].packets_received, vehicles_sl[0].packets_received)

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
    mac = MACSim(env, vehicles, symbols_per_packet=10)
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
    mac.plot_ber_vs_time(df)

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
    # plot_pa_evm(Nfft=64, Ncp=16, mod_order=4)

    # simulate_constant_speed()
    # simulate_emergency()
    # simulate_far_fast_moving()
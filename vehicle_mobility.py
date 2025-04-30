#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import simpy
from scipy.special import j0
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter

BSM_MSG_STRUCT = {
    "message_id": 8,
    "temporary_id": 32,
    "timestamp": 32,
    "lat": 32,
    "lon": 32,
    "elev": 32,
    "accuracy": 1,
    "speed": 16,
    "direction": 16,
    "ax": 16,
    "ay": 16,
    "az": 16,
    "yaw_rate": 16,
    "brake_status": 8,
    "length": 16,
    "width": 16,
    "emergency": 1
}

class Vehicle:
    """
    Represents a vehicle in 2D space with constant velocity.

    Attributes:
        id (str): Unique identifier for the vehicle.
        position (np.ndarray): Current [x, y] position in meters.
        velocity (np.ndarray): Current [vx, vy] velocity in m/s.
        freq (float): Carrier frequency in Hz (for Doppler calculations).
        c (float): Speed of light in m/s.
        last_update (float): Last time (s) the position was updated.
    """
    def __init__(self, env, id, x, y, vx, vy, ax=0.0, ay=0.0, l=5, w=2, lane=0, freq=5.9e9, c=3e8, receive_f=lambda a, b: a):
        self.env = env
        self.id = id
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.acceleration = np.array([ax, ay], dtype=float)
        self.size = np.array([l, w], dtype=float)
        self.lane = lane
        self.freq = freq
        self.c = c
        self.last_update = 0.0
        self.history = []
        self.original_acceleration = self.acceleration.copy()
        self.original_velocity = self.velocity.copy()
        self.original_position1 = self.position[1].copy()
        self.stop_segments = []  # List of (start, end) tuples
        self.current_stop_start = None
        self._last_speed = np.linalg.norm([vx, vy])
        self.receive_f = receive_f
        self.process = env.process(self.move())
        self.packets_received = 0
    
    def move(self):
        while True:
            yield self.env.timeout(0.1)
            self.velocity += self.acceleration * 0.1
            self.position += self.velocity * 0.1
            self.history.append((self.env.now, self.position.copy()))

    def move_to(self, t):
        dt = t - self.last_update
        if dt <= 0:
            return

        speed = np.linalg.norm(self.velocity)
        stopped_threshold = 0.1  # Consider stopped below this speed
        
        # Track stop segments
        if speed < stopped_threshold and self._last_speed >= stopped_threshold:
            # Transition to stopped
            self.current_stop_start = t
        elif speed >= stopped_threshold and self._last_speed < stopped_threshold:
            # Transition to moving
            if self.current_stop_start is not None:
                self.stop_segments.append((self.current_stop_start, t))
                self.current_stop_start = None
        
        self._last_speed = speed
        
        # Update position history
        self.history.append((t, self.position.copy()))
        self.last_update = t

    @property
    def stopped_time(self):
        total = sum(end - start for start, end in self.stop_segments)
        if self.current_stop_start is not None:
            total += self.last_update - self.current_stop_start
        return total

    def doppler_shift(self, transmitter):
        """
        Compute the Doppler shift (Hz) observed at this vehicle (receiver)
        for a transmission from 'transmitter'.

        Uses:
            v_rel = (v_tx - v_rx) ⋅ (p_rx - p_tx) / |p_rx - p_tx|
            f_D = (v_rel / c) * f_c

        Args:
            transmitter (Vehicle): The transmitting vehicle.
        Returns:
            float: Doppler frequency shift in Hz.
        """
        # Vector from transmitter to receiver
        disp = self.position - transmitter.position
        dist = np.linalg.norm(disp)
        if dist == 0:
            return 0.0
        direction = disp / dist
        # Relative velocity (projected)
        rel_vel = np.dot(transmitter.velocity - self.velocity, direction)
        return (rel_vel / self.c) * self.freq
    
    def generate_bsm(self, t):
        """Generate a binary BSM message from vehicle state."""
        def int_to_bits(x, width):
            return [int(b) for b in format(x & ((1 << width) - 1), f'0{width}b')]

        lat = int(self.position[0] * 1e7)
        lon = int(self.position[1] * 1e7)
        elev = int(15)
        speed = int(np.linalg.norm(self.velocity) * 100)
        heading = int(np.degrees(np.arctan2(self.velocity[1], self.velocity[0])) * 100) % 36000
        ax, ay, az = [int(x * 1000) for x in [0.5, 0.0, -9.8]]
        size_l, size_w = int(4.5 * 100), int(1.8 * 100)
        enriched_data = {
            "message_id": 0x20,
            "temporary_id": 0x3F29A78B,
            "timestamp": int(t * 1000),
            "lat": lat,
            "lon": lon,
            "elev": elev,
            "accuracy": 1,
            "speed": speed,
            "direction": heading,
            "ax": ax,
            "ay": ay,
            "az": az,
            "yaw_rate": 0,
            "brake_status": 0,
            "length": size_l,
            "width": size_w,
            "emergency": int(self.id == "Emergency")
        }
        # print("in", enriched_data)
        fields = [int_to_bits(enriched_data[k1], BSM_MSG_STRUCT[k2]) for k1, k2 in zip(enriched_data, BSM_MSG_STRUCT)]
        bits = np.array([bit for field in fields for bit in field], dtype=np.uint8)
        return bits
    
    def receive_bsm(self, bits):
        def bits_to_int(bits: list[int]) -> int:
            return sum(bit << (len(bits) - 1 - i) for i, bit in enumerate(bits))
        enriched_data = {}
        c = 0
        for field, data in BSM_MSG_STRUCT.items():
            enriched_data[field] = bits_to_int(bits[c: c+data])
            c += data
        # print("out", enriched_data)
        self.receive_f(self, enriched_data)
        self.packets_received += 1
        return


    def __repr__(self):
        return (f"Vehicle(id={self.id}, pos={self.position.tolist()}, "
                f"vel={self.velocity.tolist()})")


class ChannelFading:
    """
    Time-coherent Rayleigh fading process using Jakes' model autocorrelation.

    Attributes:
        f_max (float): Maximum Doppler frequency (Hz).
        last_time (float): Last timestamp the channel was evolved to.
        h (complex): Current complex channel gain.
    """
    def __init__(self, f_max, init_time=0.0):
        self.f_max = abs(f_max)
        self.last_time = init_time
        # Initialize channel gain as a random complex Gaussian (Rayleigh)
        real = np.random.randn()
        imag = np.random.randn()
        self.h = (real + 1j*imag) / np.sqrt(2)

    def evolve_to(self, t):
        """
        Evolve channel gain from last_time to t, preserving time correlation.

        Uses the theoretical autocorrelation:
            ρ(Δt) = J0(2π f_max Δt)
        and updates via:
            h_new = ρ h_old + √(1−ρ²)·η, η~CN(0,1)

        Args:
            t (float): Current simulation time in seconds.
        Returns:
            complex: Updated channel gain at time t.
        """
        dt = t - self.last_time
        if dt < 0:
            # If user queries past time, return existing state
            return self.h
        # Compute Bessel-based autocorrelation
        rho = j0(2 * np.pi * self.f_max * dt)
        # Generate new independent complex Gaussian sample
        eta = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        # Update channel with correct correlation
        self.h = rho * self.h + np.sqrt(1 - rho**2) * eta
        self.last_time = t
        return self.h

def firetruck_broadcast(env_sl, firetruck, vehicles, stop_radius=40):
        while True:
            yield env_sl.timeout(0.1)
            for v in vehicles:
                if v.id != firetruck.id:
                    dist = np.linalg.norm(v.position - firetruck.position)
                    if dist < stop_radius:
                        v.blinkers = True
                        if v.lane == 1:                   
                            if v.position[1] < 4 or v.velocity[0] != 0:
                                v.acceleration = np.array([-5.5, 0])
                                v.velocity[1] = 2
                            if v.position[1] > 6:
                                v.velocity[1] = 0
                            if v.velocity[0] == 0:
                                v.acceleration = v.original_acceleration()
                        else:
                            if v.position[1] > -4 or v.velocity[0] != 0:
                                v.acceleration = np.array([-5.5, 0])
                                v.velocity[1] = -2
                            if v.position[1] < -6:
                                v.velocity[1] = 0
                            if v.velocity[0] == 0:
                                v.acceleration = v.original_acceleration()
                    elif dist > stop_radius + 10:
                        v.blinkers = False
                        if v.lane == 1:
                            if v.position[1] > v.original_position1:
                                v.velocity[1] = -2
                            if v.position[1] <= v.original_position1 + 1:
                                v.velocity[1] = 0
                            if v.velocity[0] < v.original_velocity[0]:
                                v.acceleration = np.array([5, 0])
                            if v.velocity[0] >= v.original_velocity[0]:
                                v.acceleration = np.array([0, 0])
                        else:
                            if abs(v.position[1]) > abs(v.original_position1):
                                v.velocity[1] = 2
                            if abs(v.position[1]) <= abs(v.original_position1 - 1):
                                v.velocity[1] = 0
                            if v.velocity[0] < v.original_velocity[0]:
                                v.acceleration = np.array([5, 0])
                            if v.velocity[0] >= v.original_velocity[0]:
                                v.acceleration = np.array([0, 0])


def plot_vehicles_gif(env, vehicles):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    colors = ['red', 'blue', 'green', 'orange']
    rectangles = []
    for v in vehicles:
        l, w = v.size
        rect = Rectangle(
            (v.position[0] - l/2, v.position[1] - w/2),  # bottom-left corner
            l, w,
            color='red' if v.id == "Emergency" else 'blue',
            label=v.id,
            zorder = 2
        )
        ax.add_patch(rect)
        rectangles.append(rect)
    ax.set_xlim(-250, 100)
    ax.set_ylim(-80, 80)
    ax.set_title('Firetruck Simulation')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()

    ax.plot([-1000, 500], [0, 0], color='white', linestyle='--', linewidth = 2, zorder = 1)

    road_y_center = 0
    road_height = 16
    ax.add_patch(Rectangle(
        (-1000, road_y_center - road_height / 2),  # start x, start y
        1500,  # road length (adjust to fit simulation)
        road_height,
        color='gray',
        zorder = 0
    ))
    #shoulder_y = 10  # half of road height
    #ax.plot([-1000, 500], [shoulder_y, shoulder_y], color='white', linewidth=2, zorder=1)
    #ax.plot([-1000, 500], [-shoulder_y, -shoulder_y], color='white', linewidth=2, zorder=1)


    for y in [-6, 6]:  # lane boundaries
        ax.plot([-1000, 500], [y, y], color='yellow', linestyle='--', linewidth=1, zorder=1)

    # Center line (optional)
    
    print(vehicles[1].velocity)

    def update(frame):
        target_time = frame * 0.1    
        yield env.timeout(target_time - env.now)
        for i, v in enumerate(vehicles):
            l, w = v.size
            rectangles[i].set_xy((v.position[0] - l/2, v.position[1] - w/2))
        ax.set_title(f"Time: {env.now:.2f}s")
        return rectangles



    ani = FuncAnimation(fig, update, frames=200, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    ani.save("firetruck_simulation.gif", writer=writer)

    plt.show()


def plot_vehicles_old(vehicles, time=None, xlim=(-10, 100), ylim=(-10, 100)):
    """
    Plot all vehicles in 2D with velocity arrows.

    Args:
        vehicles (list[Vehicle]): Vehicles to plot.
        time (float, optional): Simulation time (for title).
        xlim (tuple): X-axis limits.
        ylim (tuple): Y-axis limits.
    """
    plt.figure(figsize=(6,6))
    for v in vehicles:
        plt.scatter(v.position[0], v.position[1], label=v.id)
        plt.arrow(
            v.position[0], v.position[1],
            v.velocity[0], v.velocity[1],
            head_width=0.5, head_length=1.0,
            fc='blue', ec='blue', alpha=0.6
        )
    title = "Vehicle Positions and Velocities"
    if time is not None:
        title += f" at t={time:.2f}s"
    plt.title(title)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_vehicles(vehicles, time=None, xlim=(-10, 110), ylim=(-10, 110)):
    """
    Enhanced vehicle plotting with trajectory history, speed indicators, and stop duration
    
    Args:
        vehicles (list[Vehicle]): Vehicles to plot
        time (float, optional): Simulation time (for title)
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Color map for different vehicles
    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))
    
    for idx, v in enumerate(vehicles):
        # Plot trajectory history
        if hasattr(v, 'history'):
            hist_pos = np.array([pos for t, pos in v.history])
            ax.plot(hist_pos[:,0], hist_pos[:,1], '--', 
                    color=colors[idx], alpha=0.4, 
                    label=f'{v.id} Path')
        
        # Current position and velocity vector
        current_pos = v.position
        speed = np.linalg.norm(v.velocity)
        arrow_scale = 2.0  # Scale factor for velocity vectors
        
        # Plot vehicle with speed-dependent marker size
        ax.scatter(*current_pos, s=100 + speed*10, 
                  color=colors[idx], edgecolors='k',
                  label=f'{v.id} Position', zorder=4)
        
        # Velocity vector
        ax.quiver(*current_pos, *v.velocity, 
                 scale=1/(0.1*arrow_scale), scale_units='xy',
                 color=colors[idx], angles='xy', 
                 width=0.002, headwidth=4, zorder=3)
        
        # Annotation box
        text_str = (f"ID: {v.id}\n"
                    f"Speed: {speed:.1f} m/s\n"
                    f"Vx: {v.velocity[0]:.1f} m/s\n"
                    f"Vy: {v.velocity[1]:.1f} m/s")
        
        if hasattr(v, 'stopped_time') and v.stopped_time > 0:
            text_str += f"\nStopped: {v.stopped_time:.1f}s"
            ax.add_patch(plt.Circle(current_pos, 3, 
                                   color='red', alpha=0.1))
        
        ax.annotate(text_str, current_pos + np.array([2, 2]),
                   color=colors[idx], fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.8))

    # Dynamic axis limits
    all_positions = np.concatenate([v.position[None] for v in vehicles])
    buffer = 15
    ax.set_xlim(min(all_positions[:,0].min(), xlim[0]) - buffer,
               max(all_positions[:,0].max(), xlim[1]) + buffer)
    ax.set_ylim(min(all_positions[:,1].min(), ylim[0]) - buffer,
               max(all_positions[:,1].max(), ylim[1]) + buffer)

    title = "Enhanced Vehicle Dynamics Visualization"
    if time is not None:
        title += f" at t={time:.2f}s"
    ax.set_title(title, pad=20)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

def plot_vehicles_enriched(vehicles, time=None):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))
    
    for idx, v in enumerate(vehicles):
        # Plot trajectory with stop segments
        if v.history:
            times, positions = zip(*v.history)
            x, y = zip(*positions)
            ax.plot(x, y, '--', color=colors[idx], alpha=0.4)
            
            # Mark stop segments on trajectory
            for start, end in v.stop_segments:
                mask = (np.array(times) >= start) & (np.array(times) <= end)
                if any(mask):
                    ax.scatter(np.array(x)[mask], np.array(y)[mask], 
                              color='red', s=20, alpha=0.7, zorder=2)

        # Current position and velocity
        current_pos = v.position
        speed = np.linalg.norm(v.velocity)
        
        # Vehicle marker with stop indicator
        marker = 's' if speed < 0.1 else 'o'
        ax.scatter(*current_pos, s=200, color=colors[idx],
                  edgecolors='k', zorder=4, marker=marker)

        # Velocity vector
        if speed > 0.1:
            ax.quiver(*current_pos, *v.velocity, 
                     scale=1/(0.05*speed), scale_units='xy',
                     color=colors[idx], width=0.003, 
                     headwidth=5, zorder=3)

        # Stop duration annotation
        if v.stop_segments or v.current_stop_start:
            stop_text = "\n".join([f"⏸️ {end-start:.1f}s" for start, end in v.stop_segments])
            if v.current_stop_start:
                current_stop = time - v.current_stop_start if time else 0
                stop_text += f"\n⏸️ {current_stop:.1f}s (current)"
            ax.annotate(stop_text, (current_pos[0], current_pos[1] - 8),
                       color='red', ha='center', va='top', fontsize=8)

        # Information panel
        text_str = (f"🚦 {v.id}\n"
                    f"Speed: {speed:.1f} m/s\n"
                    f"Position: {current_pos}\n"
                    f"Total stopped: {v.stopped_time:.1f}s")
        
        ax.annotate(text_str, current_pos + np.array([5, 5]),
                   color=colors[idx], fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.9))

    ax.set_title(f"Stoplight Scenario Visualization{' at t='+str(time)+'s' if time else ''}")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage block
if __name__ == "__main__":
    # Create some vehicles
    vehicles = [
        Vehicle("A", 0, 0, 20, 0),
        Vehicle("B", 100, 0, -15, 0)
    ]
    # Move vehicles to t=1 second
    for v in vehicles:
        v.move_to(1.0)
    # Compute Doppler
    dop = vehicles[1].doppler_shift(vehicles[0])
    print(f"Doppler at B from A: {dop:.2f} Hz (f_max)")

    # Initialize a channel fading process
    chan = ChannelFading(f_max=abs(dop), init_time=1.0)
    # Evolve channel over next 0.1s in steps and plot magnitude
    times = np.linspace(1.0, 1.1, 50)
    gains = []
    for t in times:
        gains.append(abs(chan.evolve_to(t)))
    plt.figure()
    plt.plot(times, gains)
    plt.title("Time-Coherent Fading Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("|h(t)|")
    plt.grid(True)
    plt.show()

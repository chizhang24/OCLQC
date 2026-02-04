import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def generate_qubit_stream(seed=42, days=7, shots_per_exp=5000, freq_mins=30):
    """
    Generate a reproducible stream of IQ points simulating qubit drift with ROTATION.

    Physics modeled:
    - 1/f Phase Noise: Rotation of the blobs around the IQ origin.
    - 1/f Amplitude/Offset Noise: Translation (i, q drift).
    - TLS Jumps: Sudden shifts in coordinates.
    - T1 Decay: Asymmetric state-mixing.
    """
    rs = np.random.RandomState(seed)
    steps = int((days * 24 * 60) / freq_mins)
    time_points = np.linspace(0, days, steps)

    # Physical Constants
    base_sigma = 0.52
    t1_decay_rate = 0.04

    # OU process constants for Translation
    ou_theta_trans = 0.05
    ou_sigma_trans = 0.02

    # OU process constants for Rotation (Phase Drift)
    ou_theta_rot = 0.03    # Reversion speed for phase
    ou_sigma_rot = 0.05    # Volatility of the rotation angle

    drift_i, drift_q = 0.0, 0.0
    drift_theta = 0.0  # Phase in radians

    centers_0 = []
    centers_1 = []

    # 1. Generate the Drift Trajectory (Translation + Rotation)
    for t in range(steps):
        # Update Translation Drift
        drift_i += ou_theta_trans * \
            (0.0 - drift_i) + ou_sigma_trans * rs.randn()
        drift_q += ou_theta_trans * \
            (0.0 - drift_q) + ou_sigma_trans * rs.randn()

        # Update Rotation (Phase) Drift
        drift_theta += ou_theta_rot * \
            (0.0 - drift_theta) + ou_sigma_rot * rs.randn()

        # Sudden TLS Jump
        if rs.rand() < 0.005:
            drift_i += rs.uniform(-0.2, 0.2)
            drift_q += rs.uniform(-0.2, 0.2)
            drift_theta += rs.uniform(-np.pi/8, np.pi/8)  # Sudden phase jump

        # Base centers (before rotation/drift)
        c0_raw = np.array([-1.0 + drift_i, 0.0 + drift_q])
        c1_raw = np.array([1.0 + drift_i, 0.0 + drift_q])

        # Apply Rotation Matrix around (0,0)
        cos_t, sin_t = np.cos(drift_theta), np.sin(drift_theta)
        rot_matrix = np.array([[cos_t, -sin_t],
                               [sin_t,  cos_t]])

        centers_0.append(rot_matrix @ c0_raw)
        centers_1.append(rot_matrix @ c1_raw)

    # 2. Generate the Data Stream
    stream = []
    for t in range(steps):
        num_each = shots_per_exp // 2

        points_0 = rs.normal(centers_0[t], base_sigma, size=(num_each, 2))
        points_1 = rs.normal(centers_1[t], base_sigma, size=(num_each, 2))

        # T1 Decay mixing
        mix_mask = rs.rand(num_each) < t1_decay_rate
        points_1[mix_mask] = rs.normal(
            centers_0[t], base_sigma, size=(np.sum(mix_mask), 2))

        stream.append({
            'timestamp': time_points[t],
            'x': np.vstack([points_0, points_1]).astype(np.float32),
            'y': np.hstack([np.zeros(num_each), np.ones(num_each)]).astype(np.longlong),
            'center_0': centers_0[t],
            'center_1': centers_1[t]
        })

    return stream


def animate_qubit_drift(data_stream, interval=50, subsample=500):
    """ Animate the qubit stream."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set plot limits based on the expected drift range
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('I (In-phase)', fontsize=12)
    ax.set_ylabel('Q (Quadrature)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Initialize plot elements
    # scatter_0 and scatter_1 for the 5000 raw shots
    scatter_0 = ax.scatter([], [], c='blue', alpha=0.3, s=2, label='State |0>')
    scatter_1 = ax.scatter([], [], c='red', alpha=0.3, s=2, label='State |1>')

    # path_0 and path_1 for the "trail" of the centers
    path_0, = ax.plot([], [], c='darkblue', linewidth=1.5, alpha=0.8)
    path_1, = ax.plot([], [], c='darkred', linewidth=1.5, alpha=0.8)

    # Text for timestamp
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                        fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Storage for center trajectories
    history_0 = []
    history_1 = []

    def update(frame):
        data = data_stream[frame]

        # 1. Update Raw Points (Subsampled for performance)
        # We assume labels 0 are first half, labels 1 are second half
        half = len(data['x']) // 2
        x0 = data['x'][:half]
        x1 = data['x'][half:]

        # Subsample to keep animation smooth
        idx0 = np.random.choice(len(x0), subsample, replace=False)
        idx1 = np.random.choice(len(x1), subsample, replace=False)

        scatter_0.set_offsets(x0[idx0])
        scatter_1.set_offsets(x1[idx1])

        # 2. Update Center Trajectories
        history_0.append(data['center_0'])
        history_1.append(data['center_1'])

        h0 = np.array(history_0)
        h1 = np.array(history_1)

        path_0.set_data(h0[:, 0], h0[:, 1])
        path_1.set_data(h1[:, 0], h1[:, 1])

        # 3. Update Title/Text
        time_text.set_text(f"Day: {data['timestamp']:.2f}")
        ax.set_title(f"LBNL AQT Qubit Drift Simulation - Exp {frame}")

        return scatter_0, scatter_1, path_0, path_1, time_text

    ani = FuncAnimation(fig, update, frames=len(data_stream),
                        interval=interval, blit=True)

    plt.show()
    # To save the animation, uncomment the line below:
    ani.save('qubit_drift.mp4', writer='ffmpeg')

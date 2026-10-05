"""Shared test fixtures."""

import numpy as np

from skyflow.data.urbanair500 import TruthLog


def straight_line_log(p0, v, n_total, dt=0.1, n_wx=4):
    """TruthLog with constant-velocity UAVs and all observation variates set
    to 'no loss, zero noise' unless modified by the caller. p0, v: (N, 3)."""
    p0 = np.asarray(p0, np.float32)
    v = np.asarray(v, np.float32)
    N = p0.shape[0]
    t = (np.arange(n_total, dtype=np.float32) * dt)[:, None, None]
    pos = p0[None] + v[None] * t
    return TruthLog(
        dt=dt, n_epochs=n_total, n_total=n_total,
        positions=pos.astype(np.float32),
        velocities=np.broadcast_to(v, (n_total, N, 3)).copy(),
        headings=np.zeros((n_total, N), np.float32),
        heading_rates=np.zeros((n_total, N), np.float32),
        accelerations=np.zeros((n_total, N, 3), np.float32),
        battery=np.ones((n_total, N), np.float32),
        battery_rates=np.zeros((n_total, N), np.float32),
        priorities=np.ones(N, np.int32),
        wind=np.zeros((n_total, 3), np.float32),
        local_wind_noise=np.zeros((n_total, N, 3), np.float32),
        gps_dop=np.full((n_total, N), 2.5, np.float32),
        corridor_pairs=[],
        weather_gain=np.ones((n_total, n_wx), np.float32),
        weather_vis=np.full((n_total, n_wx), 10000.0, np.float32),
        latency_u=np.zeros(N, np.float32),
        loss_u=np.ones((n_total, N), np.float32),          # never lost
        gps_noise_unit=np.zeros((n_total, N, 3), np.float32),
    )

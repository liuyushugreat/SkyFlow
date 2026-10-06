"""Observation-window bookkeeping shared by the trainer, the event-level
evaluator and the analysis scripts.

A dataset split is a flat list of snapshots: ``S`` consecutive snapshots per
scenario (1 Hz: ``scenario_duration_s * sim_freq_hz / SNAPSHOT_EPOCH_STEP``),
scenarios back to back.  Snapshots are processed in observation windows of
``K`` snapshots; the recurrent state of TR-GAT is

  * ``state_carry = "window"``   reset to zero at the start of every window
                                 (S8e and earlier: 10 s of memory at K=10);
  * ``state_carry = "scenario"`` carried (detached, i.e. truncated BPTT at
                                 window granularity) across the windows of a
                                 scenario and reset only at scenario starts
                                 (S8g: memory over the whole flight so far).
"""

from __future__ import annotations

from typing import List

STATE_CARRY_MODES = ("window", "scenario")
SNAPSHOT_EPOCH_STEP = 10      # TKG snapshots every 10 physics epochs (1 Hz at sim_freq_hz = 10); see
                              # UrbanAir500Simulator.dataset_from_logs(epoch_step=10)


def window_index_groups(n: int, K: int) -> List[List[int]]:
    """Consecutive windows of K indices; a shorter tail is covered by one
    extra window ending at n-1 (its leading snapshots are seen twice)."""
    groups = [list(range(s, s + K)) for s in range(0, n - K + 1, K)]
    if n >= K and n % K != 0:
        groups.append(list(range(n - K, n)))
    if not groups and n:
        groups.append(list(range(n)))
    return groups


def scenario_length(cfg) -> int:
    """Snapshots per scenario from the data config (60 for the default 60 s at 1 Hz)."""
    return max(int(round(float(cfg.data.scenario_duration_s) * float(cfg.data.sim_freq_hz) / SNAPSHOT_EPOCH_STEP)), 1)


def window_sequences(n: int, K: int, snapshots_per_scenario: int, state_carry: str) -> List[List[List[int]]]:
    """Windows grouped into state-carrying sequences.

    ``"window"``: every window is its own sequence.  ``"scenario"``: the
    windows whose first snapshot falls into the same scenario form one
    sequence, in time order.  Window boundaries are identical in both modes,
    so the set of scored snapshots does not depend on the switch."""
    if state_carry not in STATE_CARRY_MODES:
        raise ValueError(f"training.state_carry must be one of {STATE_CARRY_MODES}, got {state_carry!r}")
    groups = window_index_groups(n, K)
    if state_carry == "window":
        return [[g] for g in groups]
    S = max(int(snapshots_per_scenario), 1)
    seqs: List[List[List[int]]] = []
    last = None
    for g in groups:
        sid = g[0] // S
        if sid != last:
            seqs.append([])
            last = sid
        seqs[-1].append(g)
    return seqs

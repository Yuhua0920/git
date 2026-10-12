"""T15A-FIX train-only ensemble receive-power reference; no channel generator.

P_rx[r] = sum_tx Ps E_train |H[r,tx,k,t]|^2, Ps=0.5.
Every trajectory contributes equally. One common sigma2 for all RX branches,
REs and methods; never a per-frame/per-realization noise calibration.
"""
from dataclasses import dataclass
import numpy as np

SNR_DEFINITION="ensemble expected useful received signal power per RX branch / complex AWGN variance"
TRAIN_IDS=tuple(range(500))


def useful_power_per_rx(mean_link_powers, stream_power=.5):
    power=np.asarray(mean_link_powers,dtype=np.float64)
    if power.shape!=(2,2) or not np.isfinite(power).all() or np.any(power<0):
        raise ValueError("finite nonnegative mean link powers [rx=2,tx=2] required")
    if stream_power!=.5: raise ValueError("frozen two-stream symbol power Ps=0.5 required")
    return stream_power*power.sum(axis=1)


@dataclass(frozen=True)
class ReceivePowerReference:
    per_rx_signal_power: tuple[float,float]
    train_trajectory_ids: tuple[int,...]

    @property
    def reference(self):
        return float(np.mean(self.per_rx_signal_power))

    def noise_variance(self,snr_db):
        if not np.isfinite(snr_db): raise ValueError("finite SNR required")
        return float(self.reference/10**(float(snr_db)/10))

    def branch_offsets_db(self):
        return 10*np.log10(np.array(self.per_rx_signal_power)/self.reference)


class TrainReceivePowerAccumulator:
    """Accept clean train_dev H only; validation IDs fail before any mutation."""
    def __init__(self):
        self._ids=[]; self._link_power_sum=np.zeros((2,2),np.float64)

    def update(self,trajectory_id,h):
        if trajectory_id not in TRAIN_IDS or trajectory_id in self._ids:
            raise ValueError("only unique train_dev IDs 0..499 may calibrate SNR")
        channel=np.asarray(h)
        if channel.ndim!=4 or channel.shape[:2]!=(2,2) or min(channel.shape)<1 or not np.isfinite(channel).all():
            raise ValueError("finite H[rx=2,tx=2,K,time] required")
        self._link_power_sum+=np.mean(np.abs(channel)**2,axis=(2,3))
        self._ids.append(trajectory_id)

    def freeze(self):
        if set(self._ids)!=set(TRAIN_IDS): raise ValueError("all 500 train_dev trajectories required")
        per_rx=useful_power_per_rx(self._link_power_sum/len(self._ids))
        if np.any(per_rx<=0): raise ValueError("positive branch reference signal powers required")
        return ReceivePowerReference(tuple(map(float,per_rx)),tuple(sorted(self._ids)))

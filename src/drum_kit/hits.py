"""Find every hit on a separated drum track, score it, and pick the best distinct ones.

A hit starts at an onset of the piece's own track and ends at its next onset, or
at the drum's maximum length. Its score combines:

- isolation: how much louder this drum is than all other drums together during
  the hit (the other separated tracks). Where other drums play along, the
  separation leaves more of them behind in this drum's track;
- decay: how far the hit has died away when it ends, so a hit cut short by the
  next one ranks lower; one that has not died away MIN_DECAY_DB is not picked;
- strength: how close its peak is to the loudest hits of that drum. A hit more
  than STRENGTH_RANGE_DB below them (a ghost note, or bleed) is not picked.

The best hits are then picked greedily, skipping a hit that sounds the same as
one already picked (near-identical spectrum and level), so triggered samples do
not yield three copies of the same sound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

PIECES = ("kick", "snare", "toms", "hihat", "ride", "crash")

# seconds: maximum one-shot length, and the shortest gap between two hits
MAX_LENGTH = {"kick": 0.6, "snare": 0.8, "toms": 1.2, "hihat": 0.4, "ride": 1.5, "crash": 3.0}
MIN_GAP = {"kick": 0.07, "snare": 0.06, "toms": 0.06, "hihat": 0.04, "ride": 0.08, "crash": 0.15}

PRE_ROLL = 0.003        # seconds kept before the onset
FADE_IN = 0.001         # seconds
FADE_OUT = 0.010        # seconds
ENV_HOP = 0.001         # seconds per envelope frame (rounded down to whole samples)
ENV_HOLD = 0.012        # seconds the envelope holds a peak (longer than a 50 Hz half cycle)
ONSET_LOOKBACK = 0.015  # seconds back to find the level an attack rises from
FLUX_WINDOW = 1024      # samples per spectral flux frame
FLUX_HOP = 0.002        # seconds between flux frames
FLUX_MADS = 6.0         # flux peaks must exceed the running median by this many MADs
FLUX_SEARCH = 0.015     # seconds around a flux peak searched for the attack
FLUX_RANGE = 1e-8       # flux works on dB power floored 80 dB below the loudest bin
FLUX_FLOOR = 0.1        # flux peaks must reach this fraction of a strong attack (95th pct)
MASK_TIME = 0.15        # seconds after a hit in which ...
MASK_RATIO = 0.3        # ... a new hit needs this fraction of that hit's flux
ONSET_RANGE_DB = 30.0   # onsets this far below the drum's loud hits are ignored
STRENGTH_RANGE_DB = 12.0  # hits this far below the loudest score 0 on strength
MIN_DECAY_DB = 12.0     # a one-shot must have died away this far by its end
ABSENT_DB = 30.0        # a drum whose loud hits stay this far below the loudest drum is absent
DUPLICATE_DB = 1.5      # same sound: level within this ...
DUPLICATE_SIMILARITY = 0.99   # ... and spectra at least this similar


@dataclass(frozen=True)
class Hit:
    piece: str
    start: int          # sample index of the cut start
    end: int            # sample index of the cut end
    onset: float        # seconds
    peak_db: float      # peak level of the hit in dBFS
    isolation_db: float
    decay_db: float
    strength: float     # 0..1
    score: float

    def to_dict(self) -> dict:
        return {"piece": self.piece, "onset": round(self.onset, 4),
                "peak_db": round(self.peak_db, 1),
                "isolation_db": round(self.isolation_db, 1), "decay_db": round(self.decay_db, 1),
                "strength": round(self.strength, 2), "score": round(self.score, 2)}


def envelope_db(x: np.ndarray, rate: int) -> np.ndarray:
    """Peak envelope in dBFS per ENV_HOP frame of a (channels, frames) signal, held
    over ENV_HOLD so the cycles of a low drum do not read as new attacks."""
    mono = np.abs(x).max(axis=0)
    hop = max(1, int(rate * ENV_HOP))
    frames = len(mono) // hop
    peaks = mono[:frames * hop].reshape(frames, hop).max(axis=1)
    return 20 * np.log10(_past_max(peaks, int(ENV_HOLD / ENV_HOP)) + 1e-9)


def _past_max(x: np.ndarray, size: int) -> np.ndarray:
    """max(x[i - size + 1 .. i]) for every i."""
    padded = np.concatenate([np.full(size - 1, x.min() if len(x) else 0.0), x])
    return sliding_window_view(padded, size).max(axis=1)


def _past_min(x: np.ndarray, size: int) -> np.ndarray:
    """min(x[i - size .. i - 1]) for every i: the level before frame i."""
    padded = np.concatenate([np.full(size, x[0] if len(x) else 0.0), x[:-1]])
    return sliding_window_view(padded, size).min(axis=1)


def onsets(x: np.ndarray, rate: int, min_gap: float) -> np.ndarray:
    """Onset frames (ENV_HOP units) of a drum's own track.

    Candidates come from spectral flux (the summed rise of power in dB across all
    frequency bins) above a running median plus FLUX_MADS median absolute
    deviations; flux does not depend on the level dropping between hits, so
    fast double-bass or eighth-note hats are found too. Each candidate is then
    placed on the envelope: at the steepest jump within FLUX_SEARCH, walked back
    to where that jump starts. Within MASK_TIME after a hit, a candidate needs
    MASK_RATIO of that hit's flux (a ripple in its decay has far less). Hits
    more than ONSET_RANGE_DB below the drum's loud hits are dropped.
    """
    env = envelope_db(x, rate)
    hop = max(1, int(rate * ENV_HOP))
    flux_hop = max(1, int(rate * FLUX_HOP))
    _, times, z = signal.stft(x.mean(axis=0), rate, nperseg=FLUX_WINDOW,
                              noverlap=FLUX_WINDOW - flux_hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    log_power = 10 * np.log10(power + power.max() * FLUX_RANGE + 1e-20)   # dB, floored
    flux = np.r_[0.0, np.maximum(0.0, np.diff(log_power, axis=1)).sum(axis=0)]
    width = 2 * int(0.1 / FLUX_HOP) + 1
    median = signal.medfilt(flux, width)
    mad = signal.medfilt(np.abs(flux - median), width)
    threshold = median + FLUX_MADS * (mad + 1e-6)
    candidates, _ = signal.find_peaks(flux, height=threshold,
                                      distance=max(1, int(min_gap / FLUX_HOP)))
    if len(candidates):   # a typical strong attack sets the floor; tail ripples stay far below
        floor = FLUX_FLOOR * np.percentile(flux[candidates], 95)
        candidates = candidates[flux[candidates] >= floor]
    peaks: list[int] = []
    for c in candidates:   # a ripple in a hit's decay is far weaker than its attack
        if peaks and (c - peaks[-1]) * FLUX_HOP < MASK_TIME \
                and flux[c] < MASK_RATIO * flux[peaks[-1]]:
            continue
        peaks.append(c)

    jump = env - _past_min(env, int(ONSET_LOOKBACK / ENV_HOP))
    loud = np.percentile(env, 99.5)
    gap = max(1, int(min_gap / ENV_HOP))
    search = int(FLUX_SEARCH / ENV_HOP)
    starts: list[int] = []
    for t in times[peaks]:
        centre = int(t * rate / hop)
        lo, hi = max(1, centre - search), min(len(env), centre + search)
        if hi <= lo:
            continue
        q = lo + int(np.argmax(jump[lo:hi]))
        while q > 0 and jump[q - 1] >= 1.0:
            q -= 1
        if env[q:q + 20].max() >= loud - ONSET_RANGE_DB and (not starts or q - starts[-1] >= gap):
            starts.append(q)
    return np.asarray(starts, dtype=int)


def find_hits(piece: str, track: np.ndarray, others: np.ndarray, rate: int) -> list[Hit]:
    """All hits of one drum. `track` is its separated track, `others` the sum of the
    other drums' tracks, both (channels, frames)."""
    env = envelope_db(track, rate)
    frames = onsets(track, rate, MIN_GAP[piece])
    max_len = int(MAX_LENGTH[piece] / ENV_HOP)
    hop = max(1, int(rate * ENV_HOP))
    loudest = np.percentile([env[f:f + 20].max() for f in frames], 95) if len(frames) else 0.0

    hits = []
    for k, f in enumerate(frames):
        nxt = frames[k + 1] if k + 1 < len(frames) else len(env)
        stop = min(f + max_len, nxt, len(env))
        if stop - f < 20:
            continue
        window = env[f:stop]
        peak = float(window[:20].max())
        tail = float(window[-10:].mean())
        own = _power(track[:, f * hop:stop * hop])
        rest = _power(others[:, f * hop:stop * hop])
        isolation = 10 * np.log10((own + 1e-12) / (rest + 1e-12))
        decay = peak - tail
        strength = float(np.clip(1 - (loudest - peak) / STRENGTH_RANGE_DB, 0, 1))
        score = float(np.clip(isolation, -20, 40) + 0.5 * min(decay, 40) + 10 * strength)
        start = max(0, f * hop - int(PRE_ROLL * rate))
        hits.append(Hit(piece, start, stop * hop, f * hop / rate, peak, float(isolation),
                        float(decay), strength, score))
    return hits


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) if x.size else 0.0


def fingerprint(track: np.ndarray, hit: Hit, rate: int) -> np.ndarray:
    """Normalised log spectrum of the first 100 ms of a hit."""
    seg = track.mean(axis=0)[hit.start:hit.start + int(0.1 * rate)]
    if len(seg) < 64:
        return np.zeros(64)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=4096))
    bands = np.add.reduceat(spec, np.unique(np.geomspace(1, len(spec) - 1, 64).astype(int)))
    v = np.log10(bands + 1e-9)
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def loud_level(hits: list[Hit]) -> float:
    """Peak level (dBFS) of a drum's loud hits: the 95th percentile of its hit peaks."""
    return float(np.percentile([h.peak_db for h in hits], 95)) if hits else -np.inf


def pick_best(hits: list[Hit], track: np.ndarray, rate: int, count: int = 3) -> list[Hit]:
    """The `count` best hits by score among the full-strength, fully decayed ones,
    skipping near-duplicates of a hit already picked. Fewer when fewer qualify."""
    chosen: list[tuple[Hit, np.ndarray]] = []
    usable = (h for h in hits if h.strength > 0 and h.decay_db >= MIN_DECAY_DB)
    for hit in sorted(usable, key=lambda h: h.score, reverse=True):
        fp = fingerprint(track, hit, rate)
        if any(abs(hit.peak_db - c.peak_db) < DUPLICATE_DB and float(fp @ cfp) > DUPLICATE_SIMILARITY
               for c, cfp in chosen):
            continue
        chosen.append((hit, fp))
        if len(chosen) == count:
            break
    return [h for h, _ in chosen]


def cut(x: np.ndarray, hit: Hit, rate: int) -> np.ndarray:
    """The one-shot: the hit's samples with a short fade in and out."""
    seg = x[:, hit.start:hit.end].astype(np.float32).copy()
    n_in = min(int(FADE_IN * rate), seg.shape[1] // 2)
    n_out = min(int(FADE_OUT * rate), seg.shape[1] // 2)
    if n_in:
        seg[:, :n_in] *= np.linspace(0, 1, n_in, endpoint=False, dtype=np.float32)
    if n_out:
        seg[:, -n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return seg

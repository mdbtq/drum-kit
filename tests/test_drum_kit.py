"""Hit finding and selection on synthetic drum tracks, and the pipeline with fake
models. No separation model is needed."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stem_split import audio_io
from stem_split.audio_io import Audio

from drum_kit import hits as hitlib
from drum_kit.hits import PIECES
from drum_kit.pipeline import DrumKit

SR = 44100


def kick(level: float = 1.0, pitch: float = 50.0, length: float = 0.4) -> np.ndarray:
    n = np.arange(int(length * SR)) / SR
    return (np.sin(2 * np.pi * (pitch + 120 * np.exp(-n * 30)) * n) * np.exp(-n * 12) * level)


def place(total: float, events: list[tuple[float, np.ndarray]]) -> np.ndarray:
    y = np.zeros(int(total * SR), np.float32)
    for at, x in events:
        s = int(at * SR)
        e = min(len(y), s + len(x))
        y[s:e] += x[:e - s]
    return np.stack([y, y])


class FindHitsTest(unittest.TestCase):
    def test_finds_every_hit_at_its_onset(self):
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        track = place(3.5, [(t, kick()) for t in times])
        found = hitlib.find_hits("kick", track, np.zeros_like(track), SR)
        self.assertEqual(len(found), len(times))
        for hit, t in zip(found, times):
            self.assertAlmostEqual(hit.onset, t, delta=0.003)
            self.assertLessEqual(hit.start / SR, t)          # cut starts before the attack

    def test_ranks_isolated_full_decay_hits_first(self):
        times = [0.5, 1.5, 2.5, 3.5, 3.62]               # the last two overlap
        track = place(5.0, [(t, kick()) for t in times])
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(int(0.4 * SR)).astype(np.float32) * 0.3
        others = place(5.0, [(1.5, noise)])               # loud cymbal over the 2nd hit
        found = hitlib.find_hits("kick", track, others, SR)
        by_onset = {round(h.onset, 1): h for h in found}
        self.assertGreater(by_onset[0.5].score, by_onset[1.5].score)   # isolation
        self.assertGreater(by_onset[2.5].score, by_onset[3.5].score)   # cut short by 3.62
        self.assertLess(by_onset[3.5].decay_db, by_onset[2.5].decay_db)

    def test_ghost_notes_are_not_strong(self):
        track = place(3.0, [(0.5, kick()), (1.5, kick(0.1))])
        found = hitlib.find_hits("kick", track, np.zeros_like(track), SR)
        self.assertEqual([round(h.strength, 1) for h in found], [1.0, 0.0])


class PickBestTest(unittest.TestCase):
    def test_skips_identical_copies(self):
        # six identical samples and two different ones: the picks must all differ
        events = [(0.5 + i * 0.6, kick()) for i in range(6)]
        events += [(4.2, kick(0.5)), (4.8, kick(pitch=80))]
        track = place(6.0, events)
        found = hitlib.find_hits("kick", track, np.zeros_like(track), SR)
        best = hitlib.pick_best(found, track, SR, 3)
        self.assertEqual(len(best), 3)
        onsets = sorted(round(h.onset, 1) for h in best)
        self.assertEqual(sum(t < 4.0 for t in onsets), 1)   # one of the identical six
        self.assertIn(4.2, onsets)
        self.assertIn(4.8, onsets)

    def test_never_picks_weak_hits(self):
        track = place(4.0, [(0.5, kick()), (1.5, kick(0.15)), (2.5, kick(0.12, pitch=90))])
        found = hitlib.find_hits("kick", track, np.zeros_like(track), SR)
        best = hitlib.pick_best(found, track, SR, 3)
        self.assertEqual([round(h.onset, 1) for h in best], [0.5])

    def test_never_picks_a_hit_cut_short(self):
        track = place(3.0, [(0.5, kick()), (0.58, kick()), (1.5, kick(pitch=70))])
        found = hitlib.find_hits("kick", track, np.zeros_like(track), SR)
        best = hitlib.pick_best(found, track, SR, 3)
        self.assertNotIn(0.5, [round(h.onset, 2) for h in best])   # rings into the next hit

    def test_cut_fades_both_ends(self):
        track = place(2.0, [(0.5, kick())])
        hit = hitlib.find_hits("kick", track, np.zeros_like(track), SR)[0]
        shot = hitlib.cut(track, hit, SR)
        self.assertEqual(shot[0, 0], 0.0)
        self.assertEqual(shot[0, -1], 0.0)
        self.assertEqual(shot.shape[1], hit.end - hit.start)


class FakeSplitter:
    """Every drum gets its own pattern; the 'drum stem' is their sum."""

    def __init__(self, tracks: dict[str, np.ndarray]):
        self.tracks = tracks

    def separate(self, drums: Audio) -> dict[str, Audio]:
        return {p: Audio(self.tracks[p], drums.sample_rate) for p in PIECES}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def test_writes_drums_pieces_hits_and_report(self):
        tracks = {p: place(4.0, [(0.3 + 0.7 * i + 0.05 * k, kick(pitch=40 + 20 * k + 5 * i))
                                 for i in range(5)])
                  for k, p in enumerate(PIECES)}
        drums = np.sum(list(tracks.values()), axis=0)
        src = self.dir / "drums.wav"
        audio_io.write(src, Audio(drums, SR))
        out = self.dir / "kit"
        report = DrumKit(None, FakeSplitter(tracks), count=3).process(src, out, ".wav")
        self.assertFalse(report.drums_extracted)
        for p in PIECES:
            self.assertTrue((out / "pieces" / f"{p}.wav").exists())
            for n in (1, 2, 3):
                self.assertTrue((out / "hits" / "isolated" / f"{p}_{n}.wav").exists())
                self.assertTrue((out / "hits" / "drums" / f"{p}_{n}.wav").exists())
        data = json.loads((out / "report.json").read_text())
        self.assertEqual([p["piece"] for p in data["pieces"]], list(PIECES))
        self.assertEqual(len(data["pieces"][0]["picked"]), 3)

    def test_skips_a_drum_that_is_only_leakage(self):
        tracks = {p: place(4.0, [(0.3 + 0.7 * i, kick(pitch=40 + 20 * k + 5 * i))
                                 for i in range(5)])
                  for k, p in enumerate(PIECES)}
        tracks["ride"] = tracks["ride"] * 0.01          # 40 dB below the rest
        src = self.dir / "drums.wav"
        audio_io.write(src, Audio(np.sum(list(tracks.values()), axis=0), SR))
        report = DrumKit(None, FakeSplitter(tracks), count=3).process(src, self.dir / "kit", ".wav")
        ride = next(r for r in report.pieces if r.piece == "ride")
        self.assertEqual(ride.picked, [])
        self.assertIn("not in this track", ride.note)
        self.assertFalse((self.dir / "kit" / "hits" / "isolated" / "ride_1.wav").exists())


if __name__ == "__main__":
    unittest.main()

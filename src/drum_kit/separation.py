"""Model steps: the drum stem from the mix, and the drum stem split per drum.

The drum stem is the average of several Demucs models' drum outputs. Averaging
cancels part of each model's own leakage and artifacts. Removing the vocals
first (Mel-Band RoFormer) is optional: on a test track it did not reduce the
tonal leakage into the drums. Nothing else is kept, so no compromise is made
for the other instruments.

The split per drum uses MDX23C DrumSep (aufr33 and jarredou) through
audio-separator: kick, snare, toms, hi-hat, ride and crash.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from stem_split import audio_io
from stem_split.audio_io import Audio
from stem_split.separators import (ROFORMER_CACHE, DemucsSeparator, RoformerVocals,
                                   SeparationError, _conform)

from .hits import PIECES

DEMUCS_MODELS = ("htdemucs_ft", "hdemucs_mmi", "htdemucs")
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
MODEL_CACHE = ROFORMER_CACHE   # shared with stem-split, so weights are downloaded once
MODEL_LABELS = {"hihat": "hh"}   # DrumSep's own name where it differs from ours


class DrumExtractor:
    def __init__(self, models: tuple[str, ...] = DEMUCS_MODELS, remove_vocals: bool = False,
                 device: str = "auto"):
        self.models = models
        self._vocals = RoformerVocals() if remove_vocals else None
        self._demucs = [DemucsSeparator(m, device) for m in models]

    def extract(self, mix: Audio) -> Audio:
        source = mix
        if self._vocals is not None:
            vocals = self._vocals.separate(mix)["vocals"]
            source = Audio(mix.samples - vocals.samples, mix.sample_rate)
        drums = [sep.separate(source)["drums"].samples for sep in self._demucs]
        return Audio(np.mean(drums, axis=0).astype(np.float32), mix.sample_rate)


class DrumSep:
    """Kick, snare, toms, hh, ride and crash from a drum stem (audio-separator works on
    files, so the audio goes through a private temp folder)."""

    def __init__(self, model: str = DRUMSEP_MODEL):
        try:
            from audio_separator.separator import Separator as _Separator
        except ImportError as e:
            raise SeparationError(f"audio-separator is not installed: {e}") from e
        self._tmp = tempfile.TemporaryDirectory(prefix="drum-kit-")
        try:
            self._sep = _Separator(log_level=logging.WARNING, model_file_dir=str(MODEL_CACHE),
                                   output_dir=self._tmp.name, output_format="WAV",
                                   normalization_threshold=1.0)  # only scale a stem that clips
            self._sep.load_model(model)
        except Exception as e:
            raise SeparationError(f"cannot load model '{model}': {e}") from e

    def separate(self, drums: Audio) -> dict[str, Audio]:
        import torch

        tmp = Path(self._tmp.name)
        src = tmp / "drums.wav"
        audio_io.write(src, drums)
        written = [tmp / Path(f).name for f in self._sep.separate(str(src))]
        pieces = {}
        for piece in PIECES:
            label = MODEL_LABELS.get(piece, piece)
            match = [f for f in written if f"({label})" in f.name.lower()]
            if len(match) != 1:
                raise SeparationError(f"DrumSep gave no single '{piece}' output: "
                                      + ", ".join(f.name for f in written))
            stem = audio_io.read(match[0])
            pieces[piece] = _conform(torch.from_numpy(stem.samples), stem.sample_rate, drums)
        for f in [src, *written]:
            f.unlink(missing_ok=True)
        return pieces

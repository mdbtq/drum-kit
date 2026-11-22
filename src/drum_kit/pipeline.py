"""Track -> drum stem -> one track per drum -> the best distinct one-shots per drum."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stem_split import audio_io
from stem_split.audio_io import Audio

from . import hits as hitlib
from .hits import PIECES


@dataclass
class PieceResult:
    piece: str
    hits_found: int
    loud_db: float               # peak level of the drum's loud hits (dBFS)
    picked: list[dict] = field(default_factory=list)
    note: str = ""               # why nothing was picked


@dataclass
class Report:
    source: str
    out_dir: str
    drums_extracted: bool
    pieces: list[PieceResult]

    def to_dict(self) -> dict:
        return {"source": self.source, "out_dir": self.out_dir,
                "drums_extracted": self.drums_extracted,
                "pieces": [vars(p) for p in self.pieces]}


class DrumKit:
    """`extractor` makes the drum stem from a mix (None: the input is a drum stem);
    `splitter` splits a drum stem per drum."""

    def __init__(self, extractor, splitter, count: int = 3):
        self.extractor, self.splitter, self.count = extractor, splitter, count

    def process(self, src: Path, out_dir: Path, ext: str) -> Report:
        audio = audio_io.read(src)
        drums = self.extractor.extract(audio) if self.extractor else audio
        audio_io.write(out_dir / f"drums{ext}", drums)
        pieces = self.splitter.separate(drums)
        for piece in PIECES:
            audio_io.write(out_dir / "pieces" / f"{piece}{ext}", pieces[piece])

        for folder in ("isolated", "drums"):   # one-shots of an earlier run would mix in
            for old in (out_dir / "hits" / folder).glob(f"*{ext}"):
                if old.stem.rsplit("_", 1)[0] in PIECES:
                    old.unlink()

        rate = drums.sample_rate
        found = {}
        for piece in PIECES:
            others = np.sum([pieces[p].samples for p in PIECES if p != piece], axis=0)
            found[piece] = hitlib.find_hits(piece, pieces[piece].samples, others, rate)
        loud = {p: hitlib.loud_level(found[p]) for p in PIECES}
        loudest = max(loud.values())

        results = []
        for piece in PIECES:
            track = pieces[piece].samples
            result = PieceResult(piece, len(found[piece]), round(loud[piece], 1))
            results.append(result)
            if loud[piece] < loudest - hitlib.ABSENT_DB:
                result.note = (f"not in this track: its hits stay {loudest - loud[piece]:.0f} dB "
                               f"below the loudest drum (leakage only)")
                continue
            best = hitlib.pick_best(found[piece], track, rate, self.count)
            if not best:
                result.note = "no hit at full strength that decays before the next one"
            for n, hit in enumerate(best, 1):
                name = f"{piece}_{n}{ext}"
                audio_io.write(out_dir / "hits" / "isolated" / name,
                               Audio(hitlib.cut(track, hit, rate), rate))
                audio_io.write(out_dir / "hits" / "drums" / name,
                               Audio(hitlib.cut(drums.samples, hit, rate), rate))
                result.picked.append({**hit.to_dict(), "file": name,
                                      "length": round((hit.end - hit.start) / rate, 3)})

        report = Report(src.name, str(out_dir), self.extractor is not None, results)
        (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        return report

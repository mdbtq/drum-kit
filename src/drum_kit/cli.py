"""Command line interface: drum-kit <files> [options]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

FORMATS = ("wav", "flac", "aiff")


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def _print(report) -> None:
    kind = "drums extracted" if report.drums_extracted else "input used as drum stem"
    print(f"{report.source}: {kind} -> {report.out_dir}")
    for r in report.pieces:
        picks = ", ".join(f"{_clock(h['onset'])} ({h['isolation_db']:+.0f} dB clean, "
                          f"{h['decay_db']:.0f} dB decay)" for h in r.picked)
        print(f"  {r.piece:<6} {r.hits_found:5d} hits, " + (f"best: {picks}" if picks else r.note))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drum-kit",
        description="Extract the drums from a track, split them per drum (kick, snare, toms, "
                    "hihat, ride, crash) and pick the best distinct one-shots of each.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("files", nargs="+", type=Path, help="audio files")
    p.add_argument("-n", "--count", type=int, default=3, help="one-shots per drum (default 3)")
    p.add_argument("--drums", action="store_true",
                   help="the input is already a drum stem: skip the drum extraction")
    p.add_argument("--remove-vocals", action="store_true",
                   help="remove the vocals (Mel-Band RoFormer) before extracting the drums")
    p.add_argument("--models", default=None, metavar="A,B,...",
                   help="Demucs models averaged for the drum stem "
                        "(default htdemucs_ft,hdemucs_mmi,htdemucs)")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"),
                   help="Demucs device (default: first of mps, cuda, cpu)")
    p.add_argument("-o", "--out-dir", type=Path,
                   help="output folder (default: <name>_drumkit next to the input; one "
                        "subfolder per file when several are given)")
    p.add_argument("-f", "--format", choices=FORMATS, help="output format (default: wav)")
    p.add_argument("--json", type=Path, help="also write all reports to this file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .pipeline import DrumKit
    from .separation import DEMUCS_MODELS, DrumExtractor, DrumSep

    models = tuple(m.strip() for m in args.models.split(",")) if args.models else DEMUCS_MODELS
    extractor = None if args.drums else DrumExtractor(models, args.remove_vocals, args.device)
    kit = DrumKit(extractor, DrumSep(), args.count)
    ext = f".{args.format or 'wav'}"

    reports, failed = [], 0
    for src in args.files:
        out = args.out_dir or src.parent / f"{src.stem}_drumkit"
        if args.out_dir and len(args.files) > 1:
            out = out / src.stem
        try:
            report = kit.process(src, out, ext)
        except Exception as exc:  # report and continue with the next file
            print(f"{src}: {exc}", file=sys.stderr)
            failed += 1
            continue
        _print(report)
        reports.append(report.to_dict())
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    return 1 if failed else 0


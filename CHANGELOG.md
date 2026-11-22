# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-11-23

### Fixed
- Requires stem-split 0.3.1: cached Demucs models load without contacting the
  Hugging Face hub, so the warning about unauthenticated requests is gone.

## [0.2.0] - 2026-11-22

### Changed
- One-shots are only cut from the drum's own separated track and written to
  `hits/` (`hits/kick_1.wav`). The versions cut from the full drum stem had too
  much bleed from the rest of the kit.

### Removed
- `hits/drums/` (one-shots from the full drum stem) and `pieces/` (the
  separated track per drum).

## [0.1.0] - 2026-11-22

### Added
- `drum-kit` command: extracts the drums from a track (average of the drum
  stems of `htdemucs_ft`, `hdemucs_mmi` and `htdemucs`), splits them per drum
  with MDX23C DrumSep (kick, snare, toms, hihat, ride, crash) and writes the
  best distinct one-shots per drum, each cut from the isolated drum track and
  from the full drum stem.
- Hits found with spectral flux on each drum's own track, ranked by isolation
  from the other drums, complete decay and strength. Only full-strength hits
  that die away by 12 dB qualify; near-duplicates (same spectrum and level)
  are skipped. A drum that only holds leakage is reported as not in the track.
- `--drums` for an input that already is a drum stem, `--remove-vocals`,
  `--models`, `-n`, `-o`, `-f`, `--device` and `--json`.

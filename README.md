# drum-kit

Extracts the drums from a track as cleanly as possible, splits them per drum
(kick, snare, toms, hihat, ride, crash) and writes the best one-shots of each
drum: the cleanest, fullest hits, chosen so they differ from each other. It
keeps the drum stem and the one-shots; the separated track per drum is only an
intermediate step.

Unlike [stem-split](https://github.com/mdbtq/stem-split), it does not try to
keep every instrument: only the drums matter, so no compromise is made for the
rest of the mix.

## Requirements

- Python 3.10–3.14 (3.12 is the tested version; PyTorch has no wheels for
  3.15 yet). stem-split, Demucs, audio-separator and PyTorch are installed
  automatically.
- `ffmpeg` / `ffprobe` on PATH
- [uv](https://docs.astral.sh/uv/) for the install commands below (or pipx)
- Network access on first use of a model. Demucs weights are cached in
  `~/.cache/huggingface`; the DrumSep weights (and the RoFormer weights for
  `--remove-vocals`) come through audio-separator and are cached in
  `~/.cache/stem-split/audio-separator`, shared with stem-split.

## Install

```bash
uv tool install --python 3.12 git+https://github.com/mdbtq/drum-kit@v0.2.0
```

Or with pipx: `pipx install --python python3.12 git+https://github.com/mdbtq/drum-kit@v0.2.0`.
From a local checkout: `make build/install`.

## Usage

```bash
drum-kit track.wav                 # -> track_drumkit/
drum-kit track.wav -n 5            # 5 one-shots per drum instead of 3
drum-kit drums.wav --drums         # the input already is a drum stem
drum-kit a.wav b.wav -o kits/      # one subfolder per track
```

Output per file (a 5-minute metal track, about 2 minutes on an Apple M4 Pro):

```
track.wav: drums extracted -> track_drumkit
  kick     312 hits, best: 4:01.075 (+17 dB clean, 75 dB decay), 4:24.569 (+17 dB clean, 66 dB decay), 4:08.372 (+18 dB clean, 65 dB decay)
  snare    296 hits, best: 0:26.688 (+30 dB clean, 21 dB decay), 2:04.306 (+22 dB clean, 12 dB decay), 1:27.784 (+16 dB clean, 14 dB decay)
  toms     234 hits, best: 3:41.512 (+12 dB clean, 46 dB decay), 1:49.623 (+7 dB clean, 50 dB decay), 3:40.567 (+4 dB clean, 20 dB decay)
  hihat   1022 hits, best: 0:49.748 (+36 dB clean, 36 dB decay), 3:23.509 (+33 dB clean, 28 dB decay), 0:41.937 (+38 dB clean, 24 dB decay)
  ride     676 hits, not in this track: its hits stay 34 dB below the loudest drum (leakage only)
  crash    277 hits, best: 0:07.496 (-8 dB clean, 133 dB decay), 1:29.553 (-8 dB clean, 18 dB decay), 1:49.533 (-24 dB clean, 40 dB decay)
```

- `clean` – how much louder this drum is than all other drums together during
  the hit. Where other drums play along (the crash above always sounds with
  the kick), the separation leaves more of them behind, so a negative value
  means a one-shot with more artifacts.
- `decay` – how far the hit has died away at the end of the one-shot.
- A drum is reported as not in the track when its separated track holds only
  leakage from the others (see below); no one-shots are written for it.
- Demucs shifts the audio by a random offset, so a second run can pick
  slightly different hits.

## Output

```
track_drumkit/
  drums.wav         the extracted drum stem
  hits/kick_1.wav   one-shots, cut from the drum's own separated track
  hits/kick_2.wav
  ...
  report.json       every pick with its time, scores and length
```

A one-shot holds only that drum: it is cut from the drum's separated track,
not from the full drum stem, where the rest of the kit bleeds in at nearly
every hit. Running it again on the same track replaces the one-shots of the
earlier run.

## How the drums are extracted

The drum stem is the average of the drum outputs of three Demucs models
(`htdemucs_ft`, `hdemucs_mmi`, `htdemucs`); averaging cancels part of each
model's own leakage and artifacts. Every other stem is discarded.

On a 5-minute metal track, measured as the tonal (sustained) share of the
200–2000 Hz band, where guitars and vocals leak in (lower is cleaner):

| Drum stem | Tonal share 200–2000 Hz |
|---|---|
| `htdemucs_ft` | 4.7 % |
| `htdemucs_ft`, vocals removed first | 5.1 % |
| `hdemucs_mmi`, vocals removed first | 2.5 % |
| `htdemucs`, vocals removed first | 4.0 % |
| average of the three | 3.6 % |

This is an indication, not a benchmark: a lower share can also mean a model
removes part of the snare or cymbal body. Removing the vocals first did not
reduce the leakage there, so it is optional (`--remove-vocals`). `--models`
picks other Demucs models to average, e.g. `--models hdemucs_mmi`.

The split per drum uses MDX23C DrumSep by aufr33 and jarredou, through
[audio-separator](https://pypi.org/project/audio-separator/).

## How the best hits are chosen

1. Every hit is found on the drum's own track with spectral flux: a sudden
   rise of power across the spectrum, above the running median of the flux
   and at least 10 % of a strong attack of that drum. Flux does not depend on
   the level dropping between hits, so fast kicks over a sustained low end are
   found too. Within 150 ms after a hit, a new one needs 30 % of that hit's
   flux, so ripples in its decay do not count. The start is then placed on the
   level envelope, to the millisecond.
2. A hit ends at the next hit of the same drum, or at its maximum length
   (kick 0.6 s, snare 0.8 s, toms 1.2 s, hihat 0.4 s, ride 1.5 s, crash 3 s).
   The cut starts 3 ms before the attack, with a 1 ms fade in and a 10 ms fade
   out.
3. Only hits within 12 dB of the drum's loud hits, and that have died away by
   at least 12 dB at their end, are eligible. Each is scored on:
   - **isolation**: how much louder this drum is than all other drums together
     during the hit;
   - **decay**: how far it has died away when it ends, so a hit cut short by
     the next one ranks lower;
   - **strength**: how close its peak is to the drum's loudest hits.
4. The best hits are taken in order of score, skipping one that sounds the
   same as a hit already picked (spectrum of the first 100 ms at least 99 %
   alike and level within 1.5 dB). Triggered or sampled drums therefore do not
   give three copies of one sample; the picks differ in timbre or velocity.
   When fewer hits qualify, fewer one-shots are written.
5. A drum whose loud hits stay more than 30 dB below the loudest drum is
   reported as not in the track: its separated track only holds leakage.

Levels are not normalised, so the one-shots keep their relative dynamics.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-n, --count` | `3` | One-shots per drum |
| `--drums` | off | The input already is a drum stem: skip the extraction |
| `--remove-vocals` | off | Remove the vocals (Mel-Band RoFormer) before extracting the drums |
| `--models A,B,...` | `htdemucs_ft,hdemucs_mmi,htdemucs` | Demucs models averaged for the drum stem |
| `--device` | `auto` | Demucs: `mps`, `cuda` or `cpu`; `auto` picks the first available |
| `-o, --out-dir` | `<name>_drumkit` next to the input | Output folder (one subfolder per file when several are given) |
| `-f, --format` | `wav` | `wav`, `flac`, `aiff` |
| `--json FILE` | – | Also write all reports to one file |

## Architecture

```
cli ─► pipeline.DrumKit ─► separation.DrumExtractor (Demucs average, optional RoFormer)
                        ├► separation.DrumSep        (MDX23C DrumSep via audio-separator)
                        └► hits                      (find, score, pick, cut)
```

The model steps are injected into the pipeline, so the tests run it with
fakes; audio I/O and the Demucs and RoFormer wrappers come from stem-split.

## Notes

- Demucs and audio-separator are MIT-licensed. Check the licence of the model
  weights (DrumSep in particular) before using the one-shots commercially.
- The separation is the limit: where the drums are buried in a dense mix, the
  separated drum tracks carry artifacts. The ranking picks the moments where
  they are least.

## Tests

```bash
make test/all      # hit finding, ranking and the pipeline with fake models
```

## License

MIT — see [LICENSE](LICENSE).

# playlist_generator_ACTSM

Audio analysis pipeline for the ACTSM course in MTG-SMC. The current project recursively scans a folder of audio files, loads them with Essentia, prepares multiple resampled versions of each track, extracts a set of musical descriptors and embeddings, and exports the results as CSV.

## Project overview

The main entry point is `1-audio-analysis.py`. It:

- walks an input folder recursively and finds supported audio files
- loads each file with Essentia's `AudioLoader`
- converts audio to mono with `MonoMixer`
- prepares mono audio at `44.1 kHz`, `16 kHz`, and `48 kHz`
- extracts features through `feature-extraction.py`
- exports one CSV row per audio file

The current feature set includes:

- tempo in BPM with `RhythmExtractor2013`
- key estimation with `KeyExtractor` using `temperley`, `krumhansl`, and `edma`
- integrated loudness in LUFS with `LoudnessEBUR128`
- Discogs-Effnet embeddings
- Genre Discogs400 activations
- voice/instrumental classifier outputs
- danceability classifier outputs
- LAION-CLAP audio embeddings

## Repository structure

- `1-audio-analysis.py`: recursive audio processing, CLI, and CSV export
- `feature-extraction.py`: feature extraction logic and model loading helpers
- `requirements.txt`: Python dependencies
- `models/`: expected location for Essentia and CLAP model files

## Installation

### 1. Create and activate an environment

Recommended:

```bash
conda create -n actsm python=3.10 pip
conda activate actsm
```

Python `3.10` is the safest option for this stack because Essentia, PyTorch, and audio-model dependencies tend to be less fragile there than on newer versions.

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Current dependencies:

- `essentia-tensorflow`
- `laion-clap`
- `torch`
- `torchaudio`
- `torchvision`
- `tqdm`

### 3. Download the required model files

The code expects the following files either inside a local `models/` directory at the project root, or in a custom directory pointed to by `ESSENTIA_MODELS_DIR`.

Essentia models:

- `discogs-effnet-bs64-1.pb`
- `discogs-effnet-bs64-1.json`
- `genre_discogs400-discogs-effnet-1.pb`
- `genre_discogs400-discogs-effnet-1.json`
- `voice_instrumental-discogs-effnet-1.pb`
- `voice_instrumental-discogs-effnet-1.json`
- `danceability-discogs-effnet-1.pb`
- `danceability-discogs-effnet-1.json`

CLAP checkpoint:

- `music_speech_epoch_15_esc_89.25.pt`

Environment variables supported by the code:

```bash
export ESSENTIA_MODELS_DIR=/absolute/path/to/models
export CLAP_CHECKPOINT_PATH=/absolute/path/to/music_speech_epoch_15_esc_89.25.pt
```

If these variables are not set, the code will look in `./models/`.

## Running the analysis

### Print CSV to the console

```bash
python 1-audio-analysis.py /path/to/audio/folder
```

### Save CSV to a file

```bash
python 1-audio-analysis.py /path/to/audio/folder /path/to/output/results.csv
```

The script will:

- process all supported audio files recursively
- show a progress bar in the terminal
- print CSV to `stdout` if no output path is provided
- save the CSV if an output path is given
- resume from an existing output CSV by skipping tracks whose `path` is already present
- append each completed track immediately so interrupted runs can be resumed
- skip files that fail during analysis and continue with the rest of the collection

## Supported audio formats

The current scanner includes:

- `.aac`
- `.aif`
- `.aiff`
- `.flac`
- `.m4a`
- `.mp3`
- `.ogg`
- `.opus`
- `.wav`
- `.wma`

## Notes and caveats

- The model-based features require the TensorFlow-enabled Essentia build, which is why the project uses `essentia-tensorflow` instead of plain `essentia`.
- The CSV exporter serializes nested outputs such as dicts and embedding vectors as JSON strings inside CSV cells.
- Discogs-Effnet and classifier outputs are currently pooled across time with a mean operation to obtain one output per track.
- If a model file is missing, the script will raise a `FileNotFoundError` explaining where it looked.

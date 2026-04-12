from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import essentia
from essentia.standard import AudioLoader, MonoMixer, Resample
import laion_clap  # Imported now so the dependency is wired and ready for later use.
from tqdm import tqdm


essentia.log.warningActive = False


AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def _load_feature_extraction_module():
    module_path = Path(__file__).with_name("feature-extraction.py")
    module_spec = importlib.util.spec_from_file_location(
        "feature_extraction_stub",
        module_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load feature extraction module from {module_path}")

    feature_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(feature_module)
    return feature_module


feature_extraction = _load_feature_extraction_module()


@dataclass
class LoadedAudio:
    path: Path
    original_sample_rate: int
    number_channels: int
    audio_stereo: list[float]
    audio_mono: list[float]
    audio_44_1khz: list[float]
    audio_16khz: list[float]
    audio_48khz: list[float]


def iter_audio_files(root_folder: Path) -> Iterable[Path]:
    for path in sorted(root_folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def _resample_audio(audio: list[float], input_sample_rate: int, output_sample_rate: int) -> list[float]:
    if input_sample_rate == output_sample_rate:
        return audio

    resampler = Resample(
        inputSampleRate=input_sample_rate,
        outputSampleRate=output_sample_rate,
    )
    return resampler(audio)


def load_audio_versions(audio_path: Path) -> LoadedAudio:
    loader = AudioLoader(filename=str(audio_path))
    audio_stereo, original_sample_rate, number_channels, _, _, _ = loader()

    mono_mixer = MonoMixer()
    audio_mono = mono_mixer(audio_stereo, number_channels)

    audio_44_1khz = _resample_audio(audio_mono, original_sample_rate, 44100)
    audio_16khz = _resample_audio(audio_mono, original_sample_rate, 16000)
    audio_48khz = _resample_audio(audio_mono, original_sample_rate, 48000)

    return LoadedAudio(
        path=audio_path,
        original_sample_rate=original_sample_rate,
        number_channels=number_channels,
        audio_stereo=audio_stereo,
        audio_mono=audio_mono,
        audio_44_1khz=audio_44_1khz,
        audio_16khz=audio_16khz,
        audio_48khz=audio_48khz,
    )


def extract_features(loaded_audio: LoadedAudio) -> dict:
    features = {
        "path": str(loaded_audio.path),
        "original_sample_rate": loaded_audio.original_sample_rate,
    }

    features["tempo_bpm"] = feature_extraction.extract_tempo_bpm(loaded_audio)
    features["key"] = feature_extraction.extract_key_estimates(loaded_audio)
    features["loudness_lufs"] = feature_extraction.extract_loudness_lufs(loaded_audio)
    features["discogs_effnet_embeddings"] = feature_extraction.extract_discogs_effnet_embeddings(
        loaded_audio
    )
    features["music_styles"] = feature_extraction.extract_music_styles(loaded_audio)
    features["voice_instrumental"] = feature_extraction.extract_voice_instrumental(
        loaded_audio
    )
    features["danceability"] = feature_extraction.extract_danceability(loaded_audio)
    features["clap_embeddings"] = feature_extraction.extract_clap_embeddings(loaded_audio)

    return features


def analyze_folder(root_folder: Path) -> list[dict]:
    audio_files = list(iter_audio_files(root_folder))
    results = []

    for audio_path in tqdm(audio_files, desc="Analyzing audio", unit="file"):
        loaded_audio = load_audio_versions(audio_path)
        features = extract_features(loaded_audio)
        results.append(features)

    return results


def _csv_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def _serialize_csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def export_results_to_csv(results: list[dict], output_csv: Path | None) -> None:
    fieldnames = _csv_fieldnames(results)

    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in results:
                writer.writerow({key: _serialize_csv_value(value) for key, value in row.items()})
        return

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in results:
                writer.writerow({key: _serialize_csv_value(value) for key, value in row.items()})

    print(f"Saved CSV for {len(results)} audio file(s) to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively analyze audio files inside a folder."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder that contains the audio files to process recursively.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        help="Optional CSV output path. If omitted, the CSV is printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = args.input_folder.expanduser().resolve()

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    results = analyze_folder(input_folder)
    export_results_to_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

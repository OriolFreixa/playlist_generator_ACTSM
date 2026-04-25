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

OUTPUT_FIELDNAMES = [
    "path",
    "original_sample_rate",
    "tempo_bpm",
    "key",
    "loudness_lufs",
    "discogs_effnet_embeddings",
    "music_styles",
    "voice_instrumental",
    "danceability",
    "clap_embeddings",
]


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


def _serialize_csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _serialized_row(row: dict) -> dict[str, object]:
    return {
        key: _serialize_csv_value(row.get(key))
        for key in OUTPUT_FIELDNAMES
    }


def _load_completed_paths(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()

    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "path" not in (reader.fieldnames or []):
            raise ValueError(
                f"Cannot resume from {output_csv}: missing required 'path' column."
            )

        return {
            str(row["path"]).strip()
            for row in reader
            if row.get("path")
        }


def analyze_folder(root_folder: Path, output_csv: Path | None) -> None:
    audio_files = list(iter_audio_files(root_folder))
    completed_paths: set[str] = set()
    analyzed_count = 0
    skipped_existing_count = 0
    failed_count = 0

    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for audio_path in tqdm(audio_files, desc="Analyzing audio", unit="file"):
            try:
                loaded_audio = load_audio_versions(audio_path)
                features = extract_features(loaded_audio)
            except Exception as exc:
                failed_count += 1
                print(f"Skipping {audio_path}: {exc}", file=sys.stderr)
                continue

            writer.writerow(_serialized_row(features))
            analyzed_count += 1

        print(
            f"Finished analysis: wrote {analyzed_count} row(s), skipped {failed_count} failed file(s).",
            file=sys.stderr,
        )
        return

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_paths = _load_completed_paths(output_csv)
    skipped_existing_count = len(completed_paths)

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    try:
        with output_csv.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
            if write_header:
                writer.writeheader()

            for audio_path in tqdm(audio_files, desc="Analyzing audio", unit="file"):
                audio_path_str = str(audio_path)
                if audio_path_str in completed_paths:
                    continue

                try:
                    loaded_audio = load_audio_versions(audio_path)
                    features = extract_features(loaded_audio)
                except Exception as exc:
                    failed_count += 1
                    print(f"Skipping {audio_path}: {exc}", file=sys.stderr)
                    continue

                writer.writerow(_serialized_row(features))
                csv_file.flush()
                completed_paths.add(audio_path_str)
                analyzed_count += 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Partial progress was already written and can be resumed "
            f"by rerunning with the same output path: {output_csv}",
            file=sys.stderr,
        )
        raise

    print(
        "Finished analysis: "
        f"wrote {analyzed_count} new row(s), "
        f"reused {skipped_existing_count} existing row(s), "
        f"skipped {failed_count} failed file(s)."
    )


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

    analyze_folder(input_folder, args.output_csv)


if __name__ == "__main__":
    main()

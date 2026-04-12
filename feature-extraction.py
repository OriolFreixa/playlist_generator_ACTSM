from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from essentia.standard import KeyExtractor, LoudnessEBUR128, RhythmExtractor2013


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
ESSENTIA_MODELS_DIR_ENV = "ESSENTIA_MODELS_DIR"
CLAP_CHECKPOINT_ENV = "CLAP_CHECKPOINT_PATH"

DISCOGS_EFFNET_MODEL = "discogs-effnet-bs64-1.pb"
DISCOGS_EFFNET_METADATA = "discogs-effnet-bs64-1.json"
GENRE_DISCOGS400_MODEL = "genre_discogs400-discogs-effnet-1.pb"
GENRE_DISCOGS400_METADATA = "genre_discogs400-discogs-effnet-1.json"
VOICE_INSTRUMENTAL_MODEL = "voice_instrumental-discogs-effnet-1.pb"
VOICE_INSTRUMENTAL_METADATA = "voice_instrumental-discogs-effnet-1.json"
DANCEABILITY_MODEL = "danceability-discogs-effnet-1.pb"
DANCEABILITY_METADATA = "danceability-discogs-effnet-1.json"
CLAP_CHECKPOINT_NAME = "music_speech_epoch_15_esc_89.25.pt"


def extract_tempo_bpm(loaded_audio: Any) -> float | None:
    """Extract tempo in BPM with RhythmExtractor2013 from 44.1 kHz mono audio."""
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, _, _, _, _ = rhythm_extractor(loaded_audio.audio_44_1khz)
    return float(bpm)


def extract_key_estimates(loaded_audio: Any) -> dict[str, dict[str, str | float | None]]:
    key_estimates: dict[str, dict[str, str | float | None]] = {}

    for profile in ("temperley", "krumhansl", "edma"):
        key_extractor = _get_key_extractor(profile)
        key, scale, strength = key_extractor(loaded_audio.audio_44_1khz)
        key_estimates[profile] = {
            "key": key,
            "scale": scale,
            "strength": float(strength),
        }

    return key_estimates


def extract_loudness_lufs(loaded_audio: Any) -> float | None:
    stereo_audio = _ensure_stereo_audio(loaded_audio)
    loudness = LoudnessEBUR128(sampleRate=loaded_audio.original_sample_rate)
    _, _, integrated_loudness, _ = loudness(stereo_audio)
    return float(integrated_loudness)


def extract_discogs_effnet_embeddings(loaded_audio: Any) -> list[float] | None:
    patch_embeddings = _get_discogs_effnet_patch_embeddings(loaded_audio)
    pooled_embeddings = _pool_timewise(patch_embeddings)
    return _to_float_list(pooled_embeddings)


def extract_music_styles(loaded_audio: Any) -> dict[str, float]:
    classifier = _get_genre_discogs400_classifier()
    patch_embeddings = _get_discogs_effnet_patch_embeddings(loaded_audio)
    patch_predictions = classifier(patch_embeddings)
    pooled_predictions = _pool_timewise(patch_predictions)

    metadata = _load_model_metadata_optional(GENRE_DISCOGS400_METADATA)
    labels = _metadata_classes_or_fallback(metadata, len(pooled_predictions))
    return {
        label: float(score)
        for label, score in zip(labels, pooled_predictions, strict=False)
    }


def extract_voice_instrumental(loaded_audio: Any) -> dict[str, float | str] | None:
    classifier = _get_voice_instrumental_classifier()
    patch_embeddings = _get_discogs_effnet_patch_embeddings(loaded_audio)
    patch_predictions = classifier(patch_embeddings)
    pooled_predictions = _pool_timewise(patch_predictions)

    labels = _metadata_classes_or_fallback(
        _load_model_metadata_optional(VOICE_INSTRUMENTAL_METADATA),
        len(pooled_predictions),
    )
    activations = {
        label: float(score)
        for label, score in zip(labels, pooled_predictions, strict=False)
    }
    activations["predicted_label"] = labels[int(np.argmax(pooled_predictions))]
    return activations


def extract_danceability(loaded_audio: Any) -> dict[str, float | str] | None:
    classifier = _get_danceability_classifier()
    patch_embeddings = _get_discogs_effnet_patch_embeddings(loaded_audio)
    patch_predictions = classifier(patch_embeddings)
    pooled_predictions = _pool_timewise(patch_predictions)

    labels = _metadata_classes_or_fallback(
        _load_model_metadata_optional(DANCEABILITY_METADATA),
        len(pooled_predictions),
    )
    activations = {
        label: float(score)
        for label, score in zip(labels, pooled_predictions, strict=False)
    }
    activations["predicted_label"] = labels[int(np.argmax(pooled_predictions))]
    return activations


def extract_clap_embeddings(loaded_audio: Any) -> list[float] | None:
    clap_model = _get_clap_model()
    audio = np.asarray(loaded_audio.audio_48khz, dtype=np.float32).reshape(1, -1)
    embeddings = clap_model.get_audio_embedding_from_data(x=audio, use_tensor=False)
    pooled_embeddings = _pool_timewise(embeddings)
    return _to_float_list(pooled_embeddings)


def _ensure_stereo_audio(loaded_audio: Any):
    if loaded_audio.number_channels == 1:
        mono = np.asarray(loaded_audio.audio_mono, dtype=np.float32)
        return np.column_stack((mono, mono))

    return loaded_audio.audio_stereo


def _pool_timewise(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array
    return array.mean(axis=0)


def _to_float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float32).tolist()]


def _metadata_classes_or_fallback(metadata: dict[str, Any] | None, size: int) -> list[str]:
    if metadata and isinstance(metadata.get("classes"), list):
        return [str(label) for label in metadata["classes"]]
    return [f"class_{index:03d}" for index in range(size)]


def _load_model_metadata_optional(filename: str) -> dict[str, Any] | None:
    try:
        metadata_path = _resolve_model_path(filename)
    except FileNotFoundError:
        return None

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _resolve_model_path(filename: str) -> Path:
    env_models_dir = os.getenv(ESSENTIA_MODELS_DIR_ENV)
    candidate_paths = []

    if env_models_dir:
        candidate_paths.append(Path(env_models_dir) / filename)

    candidate_paths.append(DEFAULT_MODELS_DIR / filename)
    candidate_paths.append(PROJECT_ROOT / filename)

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    searched_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not find model file '{filename}'. Checked: {searched_paths}. "
        f"Place Essentia model files in '{DEFAULT_MODELS_DIR}' or set {ESSENTIA_MODELS_DIR_ENV}."
    )


def _resolve_clap_checkpoint_path() -> Path:
    checkpoint_path = os.getenv(CLAP_CHECKPOINT_ENV)
    if checkpoint_path:
        candidate = Path(checkpoint_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"{CLAP_CHECKPOINT_ENV} is set but does not point to a file: {candidate}"
        )

    candidate = DEFAULT_MODELS_DIR / CLAP_CHECKPOINT_NAME
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not find CLAP checkpoint '{CLAP_CHECKPOINT_NAME}'. "
        f"Place it in '{DEFAULT_MODELS_DIR}' or set {CLAP_CHECKPOINT_ENV}."
    )


@lru_cache(maxsize=3)
def _get_key_extractor(profile_type: str) -> KeyExtractor:
    return KeyExtractor(profileType=profile_type, sampleRate=44100)


@lru_cache(maxsize=1)
def _get_effnet_embedding_model():
    from essentia.standard import TensorflowPredictEffnetDiscogs

    model_path = _resolve_model_path(DISCOGS_EFFNET_MODEL)
    return TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1",
    )


@lru_cache(maxsize=1)
def _get_genre_discogs400_classifier():
    from essentia.standard import TensorflowPredict2D

    model_path = _resolve_model_path(GENRE_DISCOGS400_MODEL)
    return TensorflowPredict2D(
        graphFilename=str(model_path),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )


@lru_cache(maxsize=1)
def _get_voice_instrumental_classifier():
    from essentia.standard import TensorflowPredict2D

    model_path = _resolve_model_path(VOICE_INSTRUMENTAL_MODEL)
    return TensorflowPredict2D(
        graphFilename=str(model_path),
        output="model/Softmax",
    )


@lru_cache(maxsize=1)
def _get_danceability_classifier():
    from essentia.standard import TensorflowPredict2D

    model_path = _resolve_model_path(DANCEABILITY_MODEL)
    return TensorflowPredict2D(
        graphFilename=str(model_path),
        output="model/Softmax",
    )


@lru_cache(maxsize=1)
def _get_clap_model():
    import laion_clap

    checkpoint_path = _resolve_clap_checkpoint_path()
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(checkpoint_path))
    return model


def _get_discogs_effnet_patch_embeddings(loaded_audio: Any):
    cached = getattr(loaded_audio, "_discogs_effnet_patch_embeddings", None)
    if cached is not None:
        return cached

    model = _get_effnet_embedding_model()
    embeddings = model(loaded_audio.audio_16khz)
    setattr(loaded_audio, "_discogs_effnet_patch_embeddings", embeddings)
    return embeddings

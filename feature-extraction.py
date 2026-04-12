from __future__ import annotations

from typing import Any

from essentia.standard import RhythmExtractor2013


def extract_tempo_bpm(loaded_audio: Any) -> float | None:
    """Extract tempo in BPM with RhythmExtractor2013 from 44.1 kHz mono audio."""
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, _, _, _, _ = rhythm_extractor(loaded_audio.audio_44_1khz)
    return float(bpm)


def extract_key_estimates(loaded_audio: Any) -> dict[str, dict[str, str | float | None]]:
    """TODO: Extract key/scale with KeyExtractor using temperley, krumhansl, and edma profiles."""
    _ = loaded_audio
    return {
        "temperley": {
            "key": None,
            "scale": None,
            "strength": None,
        },
        "krumhansl": {
            "key": None,
            "scale": None,
            "strength": None,
        },
        "edma": {
            "key": None,
            "scale": None,
            "strength": None,
        },
    }


def extract_loudness_lufs(loaded_audio: Any) -> float | None:
    """TODO: Extract integrated loudness in LUFS with LoudnessEBUR128."""
    _ = loaded_audio
    return None


def extract_discogs_effnet_embeddings(loaded_audio: Any) -> list[float] | None:
    """TODO: Extract Discogs-Effnet embeddings for similarity and downstream classifiers."""
    _ = loaded_audio
    return None


def extract_music_styles(loaded_audio: Any) -> list[float] | None:
    """TODO: Run the Genre Discogs400 model on Discogs-Effnet embeddings."""
    _ = loaded_audio
    return None


def extract_voice_instrumental(loaded_audio: Any) -> dict[str, float] | None:
    """TODO: Run the voice/instrumental classifier on Discogs-Effnet embeddings."""
    _ = loaded_audio
    return None


def extract_danceability(loaded_audio: Any) -> float | dict[str, float] | None:
    """TODO: Extract danceability with the Essentia algorithm or Discogs-Effnet-based classifier."""
    _ = loaded_audio
    return None


def extract_clap_embeddings(loaded_audio: Any) -> list[float] | None:
    """TODO: Extract LAION-CLAP embeddings with music_speech_epoch_15_esc_89.25.pt."""
    _ = loaded_audio
    return None

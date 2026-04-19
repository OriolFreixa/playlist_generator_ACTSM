import json
import os
import random
from pathlib import Path

import numpy
import pandas
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "full.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
CLAP_CHECKPOINT_ENV = "CLAP_CHECKPOINT_PATH"
CLAP_CHECKPOINT_NAME = "music_speech_epoch_15_esc_89.25.pt"
PLAYLIST_PATH = Path("playlists/streamlit_text_query.m3u8")
KEY_PROFILE = "krumhansl"
SUGGESTED_PROMPTS = [
    "melancholic piano ballad with vocals",
    "energetic electronic dance track",
    "heavy distorted guitars and aggressive drums",
    "warm acoustic folk song",
    "cinematic orchestral soundtrack",
]


def parse_json_dict(value):
    if isinstance(value, dict):
        return value
    if pandas.isna(value):
        return {}
    return json.loads(value)


def parse_json_list(value):
    if isinstance(value, list):
        return value
    if pandas.isna(value):
        return []
    return json.loads(value)


def key_label_from_dict(key_dict):
    profile = key_dict.get(KEY_PROFILE, {})
    key = profile.get("key")
    scale = profile.get("scale")
    strength = profile.get("strength")
    label = " ".join(part for part in [key, scale] if part).strip()
    return label, strength


def format_track_label(path_value):
    path = Path(path_value)
    folder_bits = [part for part in [path.parent.name, path.parent.parent.name] if part]
    folder_label = "/".join(reversed(folder_bits))
    if folder_label:
        return f"{path.name} ({folder_label})"
    return path.name


def top_style_summary(style_dict, limit=3):
    if not style_dict:
        return ""
    top_styles = sorted(style_dict.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{label} ({score:.2f})" for label, score in top_styles)


def normalize_rows(values):
    norms = numpy.linalg.norm(values, axis=1, keepdims=True)
    norms = numpy.where(norms == 0.0, 1.0, norms)
    return values / norms


def normalize_vector(values):
    norm = numpy.linalg.norm(values)
    if norm == 0.0:
        return values
    return values / norm


def resolve_clap_checkpoint_path():
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


@st.cache_data
def load_similarity_dataset():
    analysis = pandas.read_csv(
        DATA_PATH,
        usecols=[
            "path",
            "tempo_bpm",
            "key",
            "loudness_lufs",
            "music_styles",
            "voice_instrumental",
            "danceability",
            "clap_embeddings",
        ],
    )

    analysis["key"] = analysis["key"].apply(parse_json_dict)
    analysis["music_styles"] = analysis["music_styles"].apply(parse_json_dict)
    analysis["voice_instrumental"] = analysis["voice_instrumental"].apply(parse_json_dict)
    analysis["danceability"] = analysis["danceability"].apply(parse_json_dict)
    analysis["clap_embeddings"] = analysis["clap_embeddings"].apply(parse_json_list)

    key_info = analysis["key"].apply(key_label_from_dict)
    metadata = pandas.DataFrame(
        {
            "path": analysis["path"],
            "track": analysis["path"].apply(format_track_label),
            "tempo_bpm": pandas.to_numeric(analysis["tempo_bpm"], errors="coerce"),
            "loudness_lufs": pandas.to_numeric(analysis["loudness_lufs"], errors="coerce"),
            "key_label": key_info.apply(lambda value: value[0]),
            "key_strength": key_info.apply(lambda value: value[1]),
            "voice_label": analysis["voice_instrumental"].apply(
                lambda value: value.get("predicted_label", "")
            ),
            "danceability_label": analysis["danceability"].apply(
                lambda value: value.get("predicted_label", "")
            ),
            "top_styles": analysis["music_styles"].apply(top_style_summary),
        }
    ).set_index("path")

    vectors = numpy.asarray(analysis["clap_embeddings"].tolist(), dtype=numpy.float32)
    normalized_vectors = normalize_rows(vectors)
    return metadata, normalized_vectors


@st.cache_resource
def load_clap_model():
    try:
        import laion_clap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `laion-clap` package is not installed in the current environment."
        ) from exc

    checkpoint_path = resolve_clap_checkpoint_path()
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(checkpoint_path))
    return model, checkpoint_path


def encode_text_query(prompt):
    model, checkpoint_path = load_clap_model()
    embedding = model.get_text_embedding([prompt], use_tensor=False)
    vector = numpy.asarray(embedding[0], dtype=numpy.float32)
    return normalize_vector(vector), checkpoint_path


def rank_tracks_from_query(metadata, audio_vectors, prompt, negative_prompt=None, negative_weight=0.35):
    positive_vector, checkpoint_path = encode_text_query(prompt)
    query_vector = positive_vector.copy()

    if negative_prompt:
        negative_vector, _ = encode_text_query(negative_prompt)
        query_vector = normalize_vector(query_vector - (negative_weight * negative_vector))

    scores = audio_vectors @ query_vector
    ranked = metadata.copy()
    ranked["similarity"] = scores
    return ranked.sort_values("similarity", ascending=False), query_vector, checkpoint_path


def format_results_table(result):
    table = result.copy()
    table.index.name = "path"
    table["tempo_bpm"] = table["tempo_bpm"].round(1)
    table["loudness_lufs"] = table["loudness_lufs"].round(1)
    table["similarity"] = table["similarity"].round(4)
    ordered_columns = [
        "track",
        "similarity",
        "tempo_bpm",
        "loudness_lufs",
        "key_label",
        "voice_label",
        "danceability_label",
        "top_styles",
    ]
    return table[ordered_columns].reset_index()


st.set_page_config(page_title="Freeform Text Query Playlists", layout="wide")
st.title("Playlists Based On Freeform Text Queries")
st.write(
    "Describe the kind of music you want in plain language and rank the collection "
    "through CLAP text-audio similarity."
)

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH}`.")
    st.stop()

metadata, audio_vectors = load_similarity_dataset()
st.write(f"Loaded analysis for {len(metadata)} tracks using `clap_embeddings`.")

st.write("## Query")
prompt = st.text_area(
    "Describe the target sound:",
    placeholder="dreamy ambient synths, slow tempo, spacious and cinematic",
    height=120,
)
st.caption("Suggested prompts: " + " | ".join(f"`{value}`" for value in SUGGESTED_PROMPTS))

use_negative_prompt = st.checkbox("Use a negative prompt", value=False)
negative_prompt = ""
negative_weight = 0.35
if use_negative_prompt:
    negative_prompt = st.text_input(
        "Avoid tracks like:",
        placeholder="spoken word, comedy, noisy live recordings",
    )
    negative_weight = st.slider(
        "Negative prompt weight:",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Higher values push the ranking farther away from the negative prompt.",
    )

similarity_floor = st.slider(
    "Minimum cosine similarity:",
    min_value=-1.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
)

st.write("## Post-process")
max_tracks = st.number_input("Maximum number of tracks for the playlist (0 for all):", value=25)
display_limit = st.number_input("Rows to show in the results table:", min_value=5, value=25)
preview_count = st.number_input("Audio previews to render:", min_value=1, max_value=20, value=10)
shuffle = st.checkbox("Random shuffle after ranking", value=False)

if st.button("RUN"):
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()

    if not prompt:
        st.warning("Write a freeform text query first.")
        st.stop()

    try:
        ranked, query_vector, checkpoint_path = rank_tracks_from_query(
            metadata,
            audio_vectors,
            prompt,
            negative_prompt if use_negative_prompt and negative_prompt else None,
            negative_weight,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    result = ranked.loc[ranked["similarity"] >= similarity_floor].copy()
    if result.empty:
        st.warning("No tracks matched the current similarity threshold.")
        st.stop()

    if shuffle:
        shuffled = result.copy()
        shuffled["_random"] = [random.random() for _ in range(len(shuffled))]
        result = shuffled.sort_values(["_random", "similarity"], ascending=[True, False]).drop(
            columns="_random"
        )

    export_result = result
    if max_tracks:
        export_result = result.head(int(max_tracks)).copy()

    PLAYLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAYLIST_PATH.write_text("\n".join(export_result.index.tolist()) + "\n")

    st.write("## Results")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Matched tracks", len(result))
    summary_columns[1].metric("Playlist size", len(export_result))
    summary_columns[2].metric("Best similarity", f"{result['similarity'].iloc[0]:.4f}")
    summary_columns[3].metric("Mean similarity", f"{result['similarity'].mean():.4f}")

    st.caption(
        f"Query encoded with CLAP checkpoint `{checkpoint_path}`. "
        f"Query vector norm after normalization: `{numpy.linalg.norm(query_vector):.4f}`."
    )
    st.write(f"Stored M3U playlist to `{PLAYLIST_PATH}`.")

    with st.expander("Applied query details", expanded=True):
        st.write(f"Positive prompt: `{prompt}`")
        if use_negative_prompt and negative_prompt:
            st.write(f"Negative prompt: `{negative_prompt}`")
            st.write(f"Negative weight: `{negative_weight:.2f}`")

    st.dataframe(
        format_results_table(result.head(int(display_limit))),
        use_container_width=True,
        hide_index=True,
    )

    st.write("## Audio previews")
    for track_path in export_result.index[: int(preview_count)]:
        st.write(
            f"{format_track_label(track_path)} "
            f"(similarity: {export_result.loc[track_path, 'similarity']:.4f})"
        )
        st.audio(track_path, format="audio/mp3", start_time=0)

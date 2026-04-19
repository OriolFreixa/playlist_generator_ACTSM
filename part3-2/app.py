import json
import random
from pathlib import Path

import numpy
import pandas
import streamlit as st


DATA_PATH = Path("../outputs/full.csv")
PLAYLIST_PATH = Path("playlists/streamlit_similarity.m3u8")
KEY_PROFILE = "krumhansl"
EMBEDDING_OPTIONS = {
    "LAION-CLAP (512 dims)": "clap_embeddings",
    "Discogs-Effnet (1280 dims)": "discogs_effnet_embeddings",
}


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
            "discogs_effnet_embeddings",
            "clap_embeddings",
        ],
    )

    analysis["key"] = analysis["key"].apply(parse_json_dict)
    analysis["music_styles"] = analysis["music_styles"].apply(parse_json_dict)
    analysis["voice_instrumental"] = analysis["voice_instrumental"].apply(parse_json_dict)
    analysis["danceability"] = analysis["danceability"].apply(parse_json_dict)
    analysis["discogs_effnet_embeddings"] = analysis["discogs_effnet_embeddings"].apply(parse_json_list)
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

    metadata["search_blob"] = (
        metadata.index.to_series()
        + " "
        + metadata["track"].fillna("")
        + " "
        + metadata["top_styles"].fillna("")
        + " "
        + metadata["key_label"].fillna("")
    ).str.lower()

    normalized_embeddings = {}
    for label, column in EMBEDDING_OPTIONS.items():
        vectors = numpy.asarray(analysis[column].tolist(), dtype=numpy.float32)
        norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
        norms = numpy.where(norms == 0.0, 1.0, norms)
        normalized_embeddings[label] = vectors / norms

    return metadata, normalized_embeddings


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


def similarity_ranking(metadata, normalized_matrix, seed_paths, exclude_seed_tracks):
    seed_index = metadata.index.get_indexer(seed_paths)
    seed_vectors = normalized_matrix[seed_index]
    centroid = seed_vectors.mean(axis=0)
    centroid_norm = numpy.linalg.norm(centroid)
    if centroid_norm == 0.0:
        raise ValueError("The selected seed tracks do not produce a usable embedding centroid.")
    centroid = centroid / centroid_norm

    scores = normalized_matrix @ centroid
    ranked = metadata.copy()
    ranked["similarity"] = scores

    if exclude_seed_tracks:
        ranked = ranked.drop(seed_paths, errors="ignore")

    return ranked.sort_values("similarity", ascending=False), centroid


st.set_page_config(page_title="Track Similarity Playlists", layout="wide")
st.title("Playlists Based on Track Similarity")
st.write(
    "Build playlists from one or more seed tracks using the embeddings exported in "
    f"`{DATA_PATH}`."
)

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH}`.")
    st.stop()

metadata, normalized_embeddings = load_similarity_dataset()
st.write(f"Loaded analysis for {len(metadata)} tracks.")

embedding_label = st.selectbox(
    "Embedding space:",
    list(EMBEDDING_OPTIONS),
    help=(
        "CLAP is usually broader and more semantic. Discogs-Effnet tends to focus more "
        "on audio/style similarity."
    ),
)

search_query = st.text_input(
    "Search tracks to use as seeds:",
    placeholder="Type an artist id, filename, style, or key...",
)
filtered_metadata = metadata
if search_query:
    filtered_metadata = metadata.loc[
        metadata["search_blob"].str.contains(search_query.lower(), regex=False, na=False)
    ]

st.caption(f"{len(filtered_metadata)} track options match the current search.")

seed_tracks = st.multiselect(
    "Seed tracks:",
    filtered_metadata.index.tolist(),
    format_func=format_track_label,
    help="Choose one or more tracks. If you select several, the app averages their embeddings.",
)

similarity_floor = st.slider(
    "Minimum cosine similarity:",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)
exclude_seed_tracks = st.checkbox("Exclude the seed tracks from the results", value=True)

st.write("## Post-process")
max_tracks = st.number_input("Maximum number of tracks for the playlist (0 for all):", value=25)
display_limit = st.number_input("Rows to show in the results table:", min_value=5, value=25)
preview_count = st.number_input("Audio previews to render:", min_value=1, max_value=20, value=10)
shuffle = st.checkbox("Random shuffle after ranking", value=False)

if st.button("RUN"):
    if not seed_tracks:
        st.warning("Select at least one seed track to generate a playlist.")
        st.stop()

    ranked, centroid = similarity_ranking(
        metadata,
        normalized_embeddings[embedding_label],
        seed_tracks,
        exclude_seed_tracks,
    )
    result = ranked.loc[ranked["similarity"] >= similarity_floor].copy()

    st.write("## Seed tracks")
    st.dataframe(
        format_results_table(metadata.loc[seed_tracks].assign(similarity=1.0)),
        use_container_width=True,
        hide_index=True,
    )

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
    summary_columns[3].metric(
        "Mean similarity",
        f"{result['similarity'].mean():.4f}",
    )

    st.caption(
        "The ranking uses cosine similarity to the mean embedding of the selected seed tracks. "
        f"The current centroid norm is `{numpy.linalg.norm(centroid):.4f}`."
    )
    st.write(f"Stored M3U playlist to `{PLAYLIST_PATH}`.")

    st.dataframe(
        format_results_table(result.head(int(display_limit))),
        use_container_width=True,
        hide_index=True,
    )

    st.write("## Audio previews")
    st.write("### Seed tracks")
    for track_path in seed_tracks[: int(preview_count)]:
        st.write(format_track_label(track_path))
        st.audio(track_path, format="audio/mp3", start_time=0)

    st.write("### Top matches")
    for track_path in export_result.index[: int(preview_count)]:
        st.write(
            f"{format_track_label(track_path)} "
            f"(similarity: {export_result.loc[track_path, 'similarity']:.4f})"
        )
        st.audio(track_path, format="audio/mp3", start_time=0)

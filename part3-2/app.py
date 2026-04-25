import json
from pathlib import Path

import numpy
import pandas
import streamlit as st


DATA_PATH = Path("../outputs/full.csv")
PLAYLIST_PATHS = {
    "Discogs-Effnet (1280 dims)": Path("playlists/streamlit_similarity_discogs.m3u8"),
    "LAION-CLAP (512 dims)": Path("playlists/streamlit_similarity_clap.m3u8"),
}
KEY_PROFILE = "krumhansl"
EMBEDDING_OPTIONS = {
    "LAION-CLAP (512 dims)": "clap_embeddings",
    "Discogs-Effnet (1280 dims)": "discogs_effnet_embeddings",
}
TOP_K = 10


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
    "Select one query track and compare the top 10 neighbors returned by "
    "Discogs-Effnet and LAION-CLAP side by side."
)

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH}`.")
    st.stop()

metadata, normalized_embeddings = load_similarity_dataset()
st.write(f"Loaded analysis for {len(metadata)} tracks.")

search_query = st.text_input(
    "Search the query track:",
    placeholder="Type an artist id, filename, style, or key...",
)
filtered_metadata = metadata
if search_query:
    filtered_metadata = metadata.loc[
        metadata["search_blob"].str.contains(search_query.lower(), regex=False, na=False)
    ]

st.caption(f"{len(filtered_metadata)} track options match the current search.")

seed_options = filtered_metadata.index.tolist()
seed_track = st.selectbox(
    "Query track:",
    seed_options,
    format_func=format_track_label,
    index=None,
    placeholder="Choose one track to compare both embedding spaces.",
)

similarity_floor = st.slider(
    "Minimum cosine similarity for both lists:",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)
exclude_seed_tracks = st.checkbox("Exclude the seed tracks from the results", value=True)

if st.button("RUN"):
    if not seed_track:
        st.warning("Select one query track to generate the comparison.")
        st.stop()

    st.write("## Query track")
    st.dataframe(
        format_results_table(metadata.loc[[seed_track]].assign(similarity=1.0)),
        use_container_width=True,
        hide_index=True,
    )
    st.audio(seed_track, format="audio/mp3", start_time=0)

    comparison_results = {}
    centroids = {}
    for embedding_label, normalized_matrix in normalized_embeddings.items():
        ranked, centroid = similarity_ranking(
            metadata,
            normalized_matrix,
            [seed_track],
            exclude_seed_tracks,
        )
        result = ranked.loc[ranked["similarity"] >= similarity_floor].head(TOP_K).copy()
        comparison_results[embedding_label] = result
        centroids[embedding_label] = centroid

    if all(result.empty for result in comparison_results.values()):
        st.warning("No tracks matched the current similarity threshold.")
        st.stop()

    for embedding_label, export_result in comparison_results.items():
        playlist_path = PLAYLIST_PATHS[embedding_label]
        playlist_path.parent.mkdir(parents=True, exist_ok=True)
        playlist_path.write_text("\n".join(export_result.index.tolist()) + "\n")

    st.write("## Top-10 comparison")
    left_column, right_column = st.columns(2)
    for column, embedding_label in zip(
        [left_column, right_column],
        ["Discogs-Effnet (1280 dims)", "LAION-CLAP (512 dims)"],
        strict=True,
    ):
        result = comparison_results[embedding_label]
        centroid = centroids[embedding_label]

        with column:
            st.write(f"### {embedding_label}")
            if result.empty:
                st.warning("No tracks matched the current threshold for this embedding.")
                continue

            summary_columns = st.columns(3)
            summary_columns[0].metric("Tracks shown", len(result))
            summary_columns[1].metric("Best similarity", f"{result['similarity'].iloc[0]:.4f}")
            summary_columns[2].metric("Mean similarity", f"{result['similarity'].mean():.4f}")

            st.caption(
                "Cosine similarity to the query-track embedding. "
                f"Centroid norm: `{numpy.linalg.norm(centroid):.4f}`."
            )
            st.write(f"Stored M3U playlist to `{PLAYLIST_PATHS[embedding_label]}`.")

            st.dataframe(
                format_results_table(result),
                use_container_width=True,
                hide_index=True,
            )

            st.write("#### Audio previews")
            for track_path in result.index:
                st.write(
                    f"{format_track_label(track_path)} "
                    f"(similarity: {result.loc[track_path, 'similarity']:.4f})"
                )
                st.audio(track_path, format="audio/mp3", start_time=0)

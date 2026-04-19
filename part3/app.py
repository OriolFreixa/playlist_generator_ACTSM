import json
import os.path
import random
import math
from pathlib import Path

import pandas
import streamlit as st


m3u_filepaths_file = "playlists/streamlit.m3u8"
FULL_ANALYSIS_PATH = Path("../outputs/full.csv")
ESSENTIA_ANALYSIS_PATH = Path("data/files_essentia_effnet-discogs.jsonl.pickle")
KEY_PROFILE = "krumhansl"


def parse_json_dict(value):
    if isinstance(value, dict):
        return value
    if pandas.isna(value):
        return {}
    return json.loads(value)


def resolve_preview_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)

    if "musAVdataset/audio_chunks" in raw_path:
        _, relative_path = raw_path.split("musAVdataset/audio_chunks/", maxsplit=1)
        preview_path = Path("audio") / relative_path
        if preview_path.exists():
            return str(preview_path)

    return raw_path


def format_track_label(path_value):
    path = Path(path_value)
    return f"{path.name} ({path.parent.name}/{path.parent.parent.name})"


def split_style_label(style_label):
    if "---" not in style_label:
        return style_label, style_label
    return style_label.split("---", maxsplit=1)


def build_style_taxonomy(style_columns):
    taxonomy = {}
    for style_label in style_columns:
        genre, subgenre = split_style_label(style_label)
        taxonomy.setdefault(genre, {})[subgenre] = style_label
    return taxonomy


def genre_activation_frame(dataframe, style_taxonomy, genres):
    activations = {}
    for genre in genres:
        genre_styles = list(style_taxonomy.get(genre, {}).values())
        if genre_styles:
            # Genre-only filtering uses the strongest matching subgenre activation.
            activations[genre] = dataframe[genre_styles].max(axis=1)
    return pandas.DataFrame(activations, index=dataframe.index)


def selected_subgenre_activation_frame(dataframe, style_taxonomy, genres, subgenres):
    activations = {}
    for genre in genres:
        matching_styles = [
            style_taxonomy[genre][subgenre]
            for subgenre in subgenres
            if subgenre in style_taxonomy.get(genre, {})
        ]
        if matching_styles:
            # Within one genre, selected subgenres behave as an OR filter.
            activations[genre] = dataframe[matching_styles].max(axis=1)
    return pandas.DataFrame(activations, index=dataframe.index)


def style_selection_state(dataframe, style_taxonomy, genres, subgenres):
    selected_style_labels = [
        style_taxonomy[genre][subgenre]
        for genre in genres
        for subgenre in subgenres
        if subgenre in style_taxonomy.get(genre, {})
    ]
    selected_genre_activations = genre_activation_frame(dataframe, style_taxonomy, genres)
    selected_subgenre_activations = selected_subgenre_activation_frame(
        dataframe,
        style_taxonomy,
        genres,
        subgenres,
    )
    return {
        "genres": genres,
        "subgenres": subgenres,
        "style_labels": selected_style_labels,
        "genre_activations": selected_genre_activations,
        "subgenre_activations": selected_subgenre_activations,
    }


def render_style_selector(
    dataframe,
    style_stats,
    style_taxonomy,
    widget_prefix,
    label_prefix,
    include_range=False,
):
    genre_options = sorted(style_taxonomy)
    selected_genres = st.multiselect(
        f"{label_prefix} genre:",
        genre_options,
        key=f"{widget_prefix}_genres",
    )
    available_subgenres = sorted(
        {
            subgenre
            for genre in selected_genres
            for subgenre in style_taxonomy.get(genre, {})
        }
    )
    selected_subgenres = st.multiselect(
        f"{label_prefix} subgenre:",
        available_subgenres,
        disabled=not selected_genres,
        help="Choose one or more genres first to narrow the subgenre list.",
        key=f"{widget_prefix}_subgenres",
    )

    selection = style_selection_state(
        dataframe,
        style_taxonomy,
        selected_genres,
        selected_subgenres,
    )

    if include_range:
        selection["range"] = None
        if selection["style_labels"]:
            with st.expander("Selected style statistics", expanded=True):
                st.write(selection["subgenre_activations"].describe())
            style_select_str = ", ".join(
                f"{genre} / {subgenre}"
                for genre, subgenre in (
                    split_style_label(style) for style in selection["style_labels"]
                )
            )
            selection["range"] = st.slider(
                f"Select tracks with any of `{style_select_str}` activations within range:",
                min_value=0.0,
                max_value=1.0,
                value=[0.5, 1.0],
                help="Within each selected genre, a track matches if any selected subgenre falls in the range.",
                key=f"{widget_prefix}_range",
            )
        elif selection["genres"]:
            with st.expander("Selected genre statistics", expanded=True):
                st.write(selection["genre_activations"].describe())
            genre_select_str = ", ".join(selection["genres"])
            selection["range"] = st.slider(
                f"Select tracks with `{genre_select_str}` genre activations within range:",
                min_value=0.0,
                max_value=1.0,
                value=[0.5, 1.0],
                help="Genre-only filtering uses the highest subgenre activation inside each selected genre.",
                key=f"{widget_prefix}_range",
            )

    return selection


def selected_style_rank_frame(selection):
    if selection["style_labels"]:
        return selection["subgenre_activations"], "subgenre"
    if selection["genres"]:
        return selection["genre_activations"], "genre"
    return pandas.DataFrame(index=[]), None


def numeric_series(dataframe, column_name):
    return pandas.to_numeric(dataframe[column_name], errors="coerce").dropna()


def render_numeric_range_slider(dataframe, column_name, label):
    values = numeric_series(dataframe, column_name)
    if values.empty:
        st.info(f"`{column_name}` is unavailable for the loaded analysis, so its range filter is hidden.")
        return None

    min_value = round(float(values.min()), 1)
    max_value = round(float(values.max()), 1)

    if not (math.isfinite(min_value) and math.isfinite(max_value)):
        st.info(f"`{column_name}` contains non-finite values, so its range filter is hidden.")
        return None

    if min_value > max_value:
        min_value, max_value = max_value, min_value

    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
    )


@st.cache_data
def load_audio_analysis():
    if FULL_ANALYSIS_PATH.exists():
        analysis = pandas.read_csv(
            FULL_ANALYSIS_PATH,
            usecols=[
                "path",
                "tempo_bpm",
                "key",
                "loudness_lufs",
                "music_styles",
                "voice_instrumental",
                "danceability",
            ],
        )

        analysis["path"] = analysis["path"].apply(resolve_preview_path)
        analysis["key"] = analysis["key"].apply(parse_json_dict)
        analysis["voice_instrumental"] = analysis["voice_instrumental"].apply(parse_json_dict)
        analysis["danceability"] = analysis["danceability"].apply(parse_json_dict)
        analysis["music_styles"] = analysis["music_styles"].apply(parse_json_dict)
        analysis = analysis.set_index("path")

        styles = pandas.DataFrame(analysis["music_styles"].tolist(), index=analysis.index).fillna(0.0)
        key_values = analysis["key"].apply(lambda value: value.get(KEY_PROFILE, {}))

        metadata = pandas.DataFrame(
            {
                "tempo_bpm": analysis["tempo_bpm"].astype(float).to_numpy(),
                "loudness_lufs": analysis["loudness_lufs"].astype(float).to_numpy(),
                "key": key_values.apply(lambda value: value.get("key")),
                "scale": key_values.apply(lambda value: value.get("scale")),
                "key_strength": key_values.apply(lambda value: value.get("strength")),
                "voice_label": analysis["voice_instrumental"].apply(
                    lambda value: value.get("predicted_label")
                ),
                "voice_score": analysis["voice_instrumental"].apply(
                    lambda value: float(value.get("voice", 0.0))
                ),
                "instrumental_score": analysis["voice_instrumental"].apply(
                    lambda value: float(value.get("instrumental", 0.0))
                ),
                "danceability_label": analysis["danceability"].apply(
                    lambda value: value.get("predicted_label")
                ),
                "danceable_score": analysis["danceability"].apply(
                    lambda value: float(value.get("danceable", 0.0))
                ),
                "not_danceable_score": analysis["danceability"].apply(
                    lambda value: float(value.get("not_danceable", 0.0))
                ),
            },
            index=analysis.index,
        )

        return metadata.join(styles, how="inner"), styles.columns.tolist(), str(FULL_ANALYSIS_PATH)

    styles = pandas.read_pickle(ESSENTIA_ANALYSIS_PATH)
    metadata = pandas.DataFrame(index=styles.index)
    return metadata.join(styles, how="right"), styles.columns.tolist(), str(ESSENTIA_ANALYSIS_PATH)


def format_results_table(result, style_columns):
    table = pandas.DataFrame(index=result.index)
    table.index.name = "path"
    table["track"] = [format_track_label(path) for path in result.index]
    table["tempo_bpm"] = result.get("tempo_bpm")
    table["loudness_lufs"] = result.get("loudness_lufs")

    if "key" in result.columns and "scale" in result.columns:
        table["key"] = result["key"].fillna("") + " " + result["scale"].fillna("")
        table["key"] = table["key"].str.strip()

    if "voice_label" in result.columns:
        table["voice_label"] = result["voice_label"]

    if "danceability_label" in result.columns:
        table["danceability_label"] = result["danceability_label"]

    available_styles = [style for style in style_columns if style in result.columns]
    if available_styles:
        top_styles = result[available_styles].apply(
            lambda row: ", ".join(
                f"{label} ({score:.2f})"
                for label, score in row.nlargest(3).items()
            ),
            axis=1,
        )
        table["top_styles"] = top_styles

    if "tempo_bpm" in table.columns:
        table["tempo_bpm"] = table["tempo_bpm"].round(1)
    if "loudness_lufs" in table.columns:
        table["loudness_lufs"] = table["loudness_lufs"].round(1)

    ordered_columns = [
        column
        for column in [
            "track",
            "tempo_bpm",
            "loudness_lufs",
            "key",
            "voice_label",
            "danceability_label",
            "top_styles",
        ]
        if column in table.columns
    ]
    table = table[ordered_columns]
    return table.reset_index()


st.write("# Audio analysis playlists example")
audio_analysis, audio_analysis_styles, analysis_source = load_audio_analysis()
st.write(f"Using analysis data from `{analysis_source}`.")
st.write("Loaded audio analysis for", len(audio_analysis), "tracks.")

style_stats = audio_analysis[audio_analysis_styles]
style_taxonomy = build_style_taxonomy(audio_analysis_styles)

st.write("## 🔍 Select")
st.write("### By style")
with st.expander("Style activation statistics"):
    st.write(style_stats.describe())

filter_style_selection = render_style_selector(
    audio_analysis,
    style_stats,
    style_taxonomy,
    "filter_style",
    "Filter by",
    include_range=True,
)

numeric_filter_columns = [column for column in ["tempo_bpm", "loudness_lufs"] if column in audio_analysis.columns]
if numeric_filter_columns:
    st.write("### By other analysis results")

tempo_range = None
if "tempo_bpm" in audio_analysis.columns:
    tempo_range = render_numeric_range_slider(audio_analysis, "tempo_bpm", "Tempo BPM range:")

loudness_range = None
if "loudness_lufs" in audio_analysis.columns:
    loudness_range = render_numeric_range_slider(
        audio_analysis,
        "loudness_lufs",
        "Loudness LUFS range:",
    )

voice_label = None
if "voice_label" in audio_analysis.columns:
    voice_label = st.selectbox("Voice/instrumental prediction:", ["all", "voice", "instrumental"])

danceability_label = None
if "danceability_label" in audio_analysis.columns:
    danceability_label = st.selectbox(
        "Danceability prediction:",
        ["all", "danceable", "not_danceable"],
    )

key_options = []
selected_keys = []
if {"key", "scale"}.issubset(audio_analysis.columns):
    key_options = sorted(
        {
            f"{key} {scale}".strip()
            for key, scale in audio_analysis[["key", "scale"]].dropna().itertuples(index=False, name=None)
        }
    )
    selected_keys = st.multiselect(f"Key ({KEY_PROFILE})", key_options)

st.write("## 🔝 Rank")
rank_style_selection = render_style_selector(
    audio_analysis,
    style_stats,
    style_taxonomy,
    "rank_style",
    "Rank by",
)

extra_rank_options = [
    option
    for option in [
        "tempo_bpm",
        "loudness_lufs",
        "voice_score",
        "instrumental_score",
        "danceable_score",
        "not_danceable_score",
        "key_strength",
    ]
    if option in audio_analysis.columns
]
numeric_rank = st.selectbox("Then sort by another result:", ["none"] + extra_rank_options)
numeric_rank_desc = st.checkbox("Sort descending for the extra result", value=True)

st.write("## 🔀 Post-process")
max_tracks = st.number_input("Maximum number of tracks (0 for all):", value=0)
display_limit = st.number_input("Rows to show in the results table:", min_value=5, value=25)
shuffle = st.checkbox("Random shuffle")

if st.button("RUN"):
    st.write("## 🔊 Results")
    result = audio_analysis.copy()

    if filter_style_selection["style_labels"] and filter_style_selection["range"]:
        for genre in filter_style_selection["subgenre_activations"].columns:
            subgenre_scores = filter_style_selection["subgenre_activations"].loc[result.index, genre]
            result = result.loc[
                (subgenre_scores >= filter_style_selection["range"][0])
                & (subgenre_scores <= filter_style_selection["range"][1])
            ]
        st.write("Applied subgenre filters.")
    elif filter_style_selection["genres"] and filter_style_selection["range"] is not None:
        for genre in filter_style_selection["genres"]:
            genre_scores = filter_style_selection["genre_activations"].loc[result.index, genre]
            result = result.loc[
                (genre_scores >= filter_style_selection["range"][0])
                & (genre_scores <= filter_style_selection["range"][1])
            ]
        st.write("Applied genre filters.")

    if tempo_range:
        result = result.loc[
            (result["tempo_bpm"] >= tempo_range[0]) & (result["tempo_bpm"] <= tempo_range[1])
        ]
        st.write(f"Applied tempo filter: {tempo_range[0]:.1f} to {tempo_range[1]:.1f} BPM.")

    if loudness_range:
        result = result.loc[
            (result["loudness_lufs"] >= loudness_range[0])
            & (result["loudness_lufs"] <= loudness_range[1])
        ]
        st.write(f"Applied loudness filter: {loudness_range[0]:.1f} to {loudness_range[1]:.1f} LUFS.")

    if voice_label and voice_label != "all":
        result = result.loc[result["voice_label"] == voice_label]
        st.write(f"Applied voice/instrumental filter: `{voice_label}`.")

    if danceability_label and danceability_label != "all":
        result = result.loc[result["danceability_label"] == danceability_label]
        st.write(f"Applied danceability filter: `{danceability_label}`.")

    if selected_keys:
        key_filter = (result["key"].fillna("") + " " + result["scale"].fillna("")).str.strip()
        result = result.loc[key_filter.isin(selected_keys)]
        st.write(
            "Applied key filter on the "
            f"`{KEY_PROFILE}` profile: {', '.join(selected_keys)}."
        )

    rank_frame, rank_mode = selected_style_rank_frame(rank_style_selection)
    if rank_mode:
        result = result.copy()
        rank_columns = list(rank_frame.columns)
        result["RANK"] = rank_frame.loc[result.index, rank_columns[0]]
        for rank_column in rank_columns[1:]:
            result["RANK"] *= rank_frame.loc[result.index, rank_column]
        sort_columns = ["RANK"]
        ascending = [False]
        if numeric_rank != "none":
            sort_columns.append(numeric_rank)
            ascending.append(not numeric_rank_desc)
        result = result.sort_values(sort_columns, ascending=ascending)
        st.write(f"Applied ranking by {rank_mode} activations.")
    elif numeric_rank != "none":
        result = result.sort_values(numeric_rank, ascending=not numeric_rank_desc)
        st.write(f"Sorted by `{numeric_rank}`.")

    if result.empty:
        st.warning("No tracks matched the current filters.")
    else:
        summary_columns = st.columns(4)
        summary_columns[0].metric("Matched tracks", len(result))
        if "tempo_bpm" in result.columns:
            summary_columns[1].metric("Median BPM", f"{result['tempo_bpm'].median():.1f}")
        if "loudness_lufs" in result.columns:
            summary_columns[2].metric("Median LUFS", f"{result['loudness_lufs'].median():.1f}")
        if "key_strength" in result.columns:
            summary_columns[3].metric(
                f"Mean {KEY_PROFILE} strength",
                f"{result['key_strength'].mean():.2f}",
            )

        results_table = format_results_table(result, audio_analysis_styles)
        if "RANK" in result.columns:
            results_table.insert(2, "rank_score", result["RANK"].round(4).values)
        st.dataframe(results_table.head(display_limit), use_container_width=True, hide_index=True)

        mp3s = list(result.index)
        if max_tracks:
            mp3s = mp3s[:max_tracks]
            st.write("Using top", len(mp3s), "tracks from the results.")

        if shuffle:
            random.shuffle(mp3s)
            st.write("Applied random shuffle.")

        with open(m3u_filepaths_file, "w") as f:
            mp3_paths = [os.path.join("..", mp3) for mp3 in mp3s]
            f.write("\n".join(mp3_paths))
            st.write(f"Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.")

        st.write("Audio previews for the first 10 results:")
        for mp3 in mp3s[:10]:
            st.audio(mp3, format="audio/mp3", start_time=0)

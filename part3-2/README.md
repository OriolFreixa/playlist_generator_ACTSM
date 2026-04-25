Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the similarity playlist app:

```bash
./run.sh
```

The app reads `../outputs/full.csv` and writes two comparison playlists to
`playlists/streamlit_similarity_discogs.m3u8` and
`playlists/streamlit_similarity_clap.m3u8`.

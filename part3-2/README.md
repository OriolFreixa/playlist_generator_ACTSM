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

The app reads `../outputs/full.csv` and writes generated playlists to
`playlists/streamlit_similarity.m3u8`.

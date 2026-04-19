Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the freeform text-query playlist app:

```bash
./run.sh
```

The app reads `../outputs/full.csv`, uses the CLAP checkpoint from `../models/`,
and writes generated playlists to `playlists/streamlit_text_query.m3u8`.

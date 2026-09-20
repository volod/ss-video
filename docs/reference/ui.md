# UI

The current UI is a single Streamlit app in [`ui/app.py`](/home/vola/src/selfsuvis/ui/app.py). It is intentionally simple and mirrors the API directly.

## Tabs

### Index Video

- Upload a video file
- Submit a remote URL
- Submit a local directory path for batch indexing
- Poll job status by job ID

### Image Query

- Upload a query image
- Choose `search_type`
- Choose `vector_space` (`clip` or `dino`)
- Toggle reranking

### Text Query

- Submit a text search string
- Choose `search_type`
- Toggle reranking

### Admin

- View queue and active-learning stats
- Inspect mission list returned by `/admin/missions`
- Open available 3DGS outputs in the embedded SuperSplat iframe

## API form

`GET /index/form` serves a minimal HTML form for direct API testing without Streamlit.

## Result rendering

Search results show score, video ID, timestamp, frame/tile paths when present, and an `mpv` command for local playback from `.data/videos`.

---
[← API](api.md) | [Helpers →](../guide/helpers.md)

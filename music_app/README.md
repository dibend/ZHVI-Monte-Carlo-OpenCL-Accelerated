# OpenCL Music Generator

This Gradio application generates simple synthetic music tracks using PyOpenCL for randomization.

## Features
- Specify how many tracks to produce (up to 5).
- Generate individual tracks or all tracks at once.
- Delete tracks on demand.
- Each track is saved as a FLAC file in `music_app/tracks/`.

Run the app with:
```bash
python music_app/app.py
```
This will launch a Gradio interface with `share=True` and `debug=True`.

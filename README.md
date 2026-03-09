# NSFW_Censoring

A Flask-based NSFW moderation tool for:
- **Video censorship** (detect NSFW segments and censor/trim video + censor NSFW audio words)
- **Subtitle censorship** (clean `.srt` using a generated NSFW report JSON)

The app provides a web UI for uploads and background processing, and also supports direct CLI usage of the backend pipeline.

---

## 1) What this repository contains

- `app.py` — Flask web app, file upload flow, status tracking, background worker queue, and download endpoints.
- `final_backend.py` — core NSFW detection/censoring pipeline for video + audio, report generation, and CLI arguments.
- `srt_handler.py` — censors subtitle text and removes subtitle lines overlapping trimmed segments.
- `keyword.txt` — default NSFW keyword list.
- `requirements.txt` — Python dependencies.

---

## 2) System requirements

## OS support

Recommended:
- Ubuntu 22.04/24.04 (or similar Linux distro)

Also possible:
- Windows (WSL2 strongly recommended for CUDA workflows)
- macOS (CPU-only path; no NVIDIA CUDA)

## Hardware

- **GPU path (recommended):** NVIDIA GPU with recent drivers
- **CPU path (fallback):** Works but much slower

## Software prerequisites

- Python 3.10+ (3.11 preferred)
- `ffmpeg` and `ffprobe` installed and available on `PATH`
- `git`
- Build tools (for some Python wheels): `build-essential`, `python3-dev` (Linux)

---

## 3) NVIDIA GPU setup (detailed)

> If your machine already has a working CUDA + PyTorch environment, you can skip to section 4.

## Step 3.1 — Install/verify NVIDIA driver

On Ubuntu:

```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

You should see your GPU, driver version, and no major errors.

## Step 3.2 — CUDA toolkit note

This project pins PyTorch CUDA wheels as:
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`

This means your system should be compatible with **CUDA 12.4 wheel builds**. In most cases, you do **not** need a full local CUDA toolkit if the driver is compatible, because pip wheels bundle required CUDA runtime libs.

## Step 3.3 — Quick GPU sanity test

After creating venv and installing dependencies (sections below), run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If `CUDA available: True`, your acceleration path is active.

---

## 4) Project setup (from clone to ready)

## Step 4.1 — Clone

```bash
git clone <YOUR_REPO_URL>
cd NSFW_Censoring
```

## Step 4.2 — Create and activate virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## Step 4.3 — Install OS packages

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg git build-essential python3-dev
```

Verify:

```bash
ffmpeg -version
ffprobe -version
python --version
```

## Step 4.4 — Install Python dependencies

Because `requirements.txt` includes PyTorch CUDA wheels with `+cu124`, install with the PyTorch wheel index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

If you want CPU-only install (no NVIDIA), use:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

> Note: CPU-only mode can be significantly slower.

## Step 4.5 — First-time model downloads

At first run, Hugging Face and Whisper models will download automatically (internet required).

Models used in pipeline:
- `Falconsai/nsfw_image_detection`
- `AdamCodd/vit-base-nsfw-detector`
- Whisper `medium`

---

## 5) Run the web app

Start server:

```bash
python app.py
```

Expected:
- Flask starts on `0.0.0.0:5005`
- Background worker thread starts

Open in browser:

- `http://localhost:5005`

### Web app workflow

1. Choose moderation type:
   - **Video**: upload video and pick methods/resolution/trim threshold
   - **SRT**: upload `.srt` plus censorship report `.json`
2. Submit upload.
3. Polling status/progress updates appear in UI.
4. Download output when complete.

Outputs are stored under:

- `uploads/<uuid>/status.json`
- `uploads/<uuid>/censored_output.mp4` (video flow)
- `uploads/<uuid>/censored_output.srt` (subtitle flow)

---

## 6) Run backend pipeline directly (CLI mode)

You can run the detection/censoring script without Flask:

```bash
python final_backend.py /path/to/input.mp4 \
  --output censored_output.mp4 \
  --video-method blur \
  --audio-method mute \
  --trim-threshold 5 \
  --width 1920 \
  --height 1080 \
  --nsfw-words-file keyword.txt \
  --word-padding 100 \
  --fade-duration 50
```

Supported key arguments:
- `--video-method`: `blur | pixelate | black`
- `--audio-method`: `mute | beep | tone`
- `--trim-threshold`: seconds; long NSFW segments may be trimmed
- `--width`, `--height`: output dimensions
- `--nsfw-words-file`: override word list file

---

## 7) Subtitle-only censorship flow

If you already have:
- original `.srt`
- generated `nsfw_detection_report.json`

Run programmatically:

```python
from srt_handler import censor_srt_file
censor_srt_file("input.srt", "censored_output.srt", "nsfw_detection_report.json")
```

Or use the web UI SRT mode.

---

## 8) Recommended validation checklist

After setup, run:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "import cv2, flask, transformers, whisper, librosa; print('imports ok')"
ffmpeg -version
python app.py
```

Then upload a short test video and verify:
- status progresses in UI
- output MP4 downloads
- `nsfw_detection_report.json` is generated in job folder

---

## 9) Troubleshooting

## A) `pip install` fails on torch `+cu124`

Use:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

If still failing, ensure pip is updated and Python version is supported.

## B) CUDA not detected (`torch.cuda.is_available() == False`)

- Check driver with `nvidia-smi`
- Ensure venv has CUDA wheel builds installed (`torch==...+cu124`)
- Reinstall torch packages inside the active venv

## C) `ffmpeg` not found

Install ffmpeg via OS package manager and verify command works from same shell/venv session.

## D) Slow processing

- Confirm GPU is active
- Use shorter videos for tests
- Lower output resolution for experimentation

## E) Whisper/model download issues

- Ensure outbound internet access on first run
- Retry once network is stable

---

## 10) Production notes

Current app characteristics:
- Uses Flask dev server (`debug=True`) by default
- Uses in-process background worker thread and filesystem status tracking
- `app.secret_key` is hardcoded and should be changed for production

For production hardening, consider:
- Gunicorn/Uvicorn + reverse proxy
- Task queue (Celery/RQ) + Redis
- Persistent storage and cleanup policies
- Environment-variable based secrets

---

## 11) Quick start (copy/paste)

```bash
# 1) clone
# git clone <YOUR_REPO_URL> && cd NSFW_Censoring

# 2) venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 3) system deps
sudo apt update
sudo apt install -y ffmpeg git build-essential python3-dev

# 4) python deps (NVIDIA CUDA wheels)
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# 5) run
python app.py
```

Open: `http://localhost:5005`

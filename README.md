# NSFW_Censoring

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)

> **AI-powered tool to detect and censor NSFW content in videos** using deep learning for both video frames and audio — supports smooth transitions, word-level censoring, and automatic segment trimming.

---

## 🚀 Features

- 🎞️ **Video Detection**: Frame-level NSFW detection using an ensemble of `Falconsai/nsfw_image_detection` and `AdamCodd/vit-base-nsfw-detector`
- 🎤 **Audio Detection**: Word-level NSFW detection using OpenAI’s `whisper` with accurate timestamping
- 🧠 **Smart Censoring**: Smooth transitions and confidence-based censor strength
- 🔇 **Audio Censoring Options**: `mute`, `beep`, or `tone`
- ✂️ **Segment Trimming**: Automatically trims long NSFW segments
- 📊 **Report Generation**: Detailed JSON report with timestamps and detection metadata
- ⚙️ **Fully Configurable**: Easily tweak censoring settings with CLI arguments
- 🧹 **Automatic Cleanup** of temporary files

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/nsfw-censor.git
cd nsfw-censor
```

### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
```

### 3. Install FFmpeg
FFmpeg is required for reading and writing video files.

On Ubuntu you can install it via:
```bash
sudo apt-get install ffmpeg
```
For Windows or macOS, download a build from the [FFmpeg website](https://ffmpeg.org/) and ensure `ffmpeg` is on your `PATH`.

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the script
```bash
python script.py input_video.mp4
```

# NSFW Censor — Video & Audio Moderation Tool 🛡️

> Detect and censor NSFW content in videos and audio using AI models with word-level precision and smooth video censoring.

---

## ⚙️ Prerequisites (Windows)

### ✅ Install Chocolatey

Open **PowerShell as Administrator** and run:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
````

### ✅ Install FFmpeg via Chocolatey

In your VS Code terminal:

```bash
choco install ffmpeg -y
```

---

## 🐍 Python Setup

### ✅ Install Python 3.11

Install Python 3.11 from [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

> ⚠️ Required for CUDA compatibility with PyTorch

### ✅ Create a Virtual Environment

```bash
python3.11 -m venv venv
```

### ✅ Activate the Virtual Environment

```bash
venv\Scripts\activate   # For Windows
```

---

## 🧠 CUDA Setup (if you have a GPU)

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📦 Install Other Dependencies

Make sure you're in the project directory with a `requirements.txt`, then run:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Script

### Basic Usage

```bash
python script.py input_video.mp4
```

### Advanced Usage with Arguments

```bash
python script.py input_video.mp4 \
  --output output_censored.mp4 \
  --video-method pixelate \
  --audio-method beep \
  --transition-frames 50 \
  --trim-threshold 10 \
  --width 1280 \
  --height 720 \
  --nsfw-words-file keyword.txt \
  --word-padding 150 \
  --fade-duration 100
```

---

## 🧾 Command Line Arguments

| Argument              | Type     | Default               | Description                                   |
| --------------------- | -------- | --------------------- | --------------------------------------------- |
| `video`               | str      | **Required**          | Input video file path                         |
| `--output`            | str      | `censored_output.mp4` | Output video file name                        |
| `--video-method`      | choice   | `blur`                | Options: `blur`, `pixelate`, `black`          |
| `--audio-method`      | choice   | `mute`                | Options: `mute`, `beep`, `tone`               |
| `--transition-frames` | int      | `30`                  | Smooth transition frames for fade-in/out      |
| `--trim-threshold`    | float    | `5`                   | Trim segments longer than this many seconds   |
| `--width`             | int      | `1920`                | Output video width                            |
| `--height`            | int      | `1080`                | Output video height                           |
| `--nsfw-words-file`   | str      | `keyword.txt`         | File with NSFW keywords (1 per line)          |
| `--word-padding`      | int (ms) | `100`                 | Padding in milliseconds around detected words |
| `--fade-duration`     | int (ms) | `50`                  | Fade in/out duration around NSFW words        |

---

## ✅ Example Folder Structure

```text
nsfw-censor/
├── 1920.py
├── requirements.txt
├── keyword.txt
├── input_video.mp4
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```

---

Let me know if you want this broken into separate files (`INSTALL.md`, `USAGE.md`, etc.), or if you want it added to a `docs/` folder for a more structured repo layout.
```

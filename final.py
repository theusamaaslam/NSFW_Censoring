import cv2
import os
import numpy as np
import torch
import subprocess
import shutil
import argparse
import tempfile
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
import librosa
import soundfile as sf
import whisper
import json
import glob
import re
from collections import defaultdict
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+

# ✅ CONFIGURATION
VIDEO_OUTPUT = "censored_output.mp4"
TEMP_DIR = os.path.abspath("temp_processing")
FRAME_FOLDER = os.path.abspath(os.path.join(TEMP_DIR, "frames"))
AUDIO_FOLDER = os.path.abspath(os.path.join(TEMP_DIR, "audio"))
NSFW_THRESHOLD = 0.5
AUDIO_NSFW_THRESHOLD = 0.6
BATCH_SIZE = 96
CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.4
AUDIO_CHUNK_DURATION = 5  # seconds
TRANSITION_FRAMES = 30  # Number of frames for smooth transitions
MIN_SEGMENT_LENGTH = 30  # Minimum frames to consider as a segment
SEGMENT_PADDING = 50  # Extra frames to add before/after segments for smoother transitions
AUDIO_CENSORING_METHOD = "mute"  # Options: "mute", "beep", "tone"
VIDEO_CENSORING_METHOD = "blur"  # Options: "blur", "pixelate", "black"

# Output quality parameters
OUTPUT_FRAME_RATE = 23  # Target frame rate for output video
OUTPUT_BITRATE = "5000k"  # Target bitrate for output video
OUTPUT_WIDTH = 1920  # Target width for output video
OUTPUT_HEIGHT = 1080  # Target height for output video

# New parameter for NSFW segment trimming
TRIM_THRESHOLD_SEC = 5  # Segments longer than this will be trimmed instead of censored

# Audio censoring parameters
WORD_PADDING_MS = 100  # Padding around detected NSFW words (milliseconds)
FADE_DURATION_MS = 50   # Duration of fade in/out for word-level censoring

# Default NSFW word list file
DEFAULT_KEYWORD_FILE = "keyword.txt"

# Default NSFW word list (used if keyword file is not found)
DEFAULT_NSFW_WORDS = [
    "explicit", "obscene", "vulgar", "profanity", "offensive",
    # Add more default words as needed
]

# Debug mode for detailed logging
DEBUG_MODE = True

# ✅ Setup folders
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ✅ Load NSFW Detection Models
print("🔍 Loading NSFW Detection Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1

# Primary image classifier
primary_image_classifier = pipeline(
    "image-classification", 
    "Falconsai/nsfw_image_detection", 
    device=device_id
)

# Secondary image classifier
secondary_image_classifier = pipeline(
    "image-classification", 
    "AdamCodd/vit-base-nsfw-detector", 
    device=device_id
)

# Load improved Whisper model for audio transcription
print("🎤 Loading Enhanced Audio Transcription Model...")
audio_model = whisper.load_model("medium")  # Better accuracy than base

# Global variables to store results
nsfw_segments = []  # Will store continuous segments instead of individual frames
nsfw_audio_segments = []  # Store segment-level audio detections
nsfw_word_timestamps = []  # Store precise word-level timestamps for NSFW words
trimmed_segments = []  # Store segments that were trimmed
detection_confidence = {}
video_metadata = {}  # Store video metadata for consistent output
nsfw_words = []  # Will be populated with user-provided or default NSFW words

# ✅ Debug logging function
def debug_log(message):
    """Log debug messages if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

# ✅ Function to get accurate video duration
def get_accurate_video_duration(video_path):
    """Get accurate video duration using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout.strip():
        return float(result.stdout.strip())
    return 0

# ✅ Function to verify audio-video sync
def verify_av_sync(video_path):
    """Verify audio-video sync in the output file"""
    print("🔍 Verifying audio-video synchronization...")
    
    # Get video stream duration
    video_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    video_result = subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    video_duration = float(video_result.stdout.strip()) if video_result.stdout.strip() else 0
    
    # Get audio stream duration
    audio_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    audio_result = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    audio_duration = float(audio_result.stdout.strip()) if audio_result.stdout.strip() else 0
    
    # Get container duration
    format_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    format_result = subprocess.run(format_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    format_duration = float(format_result.stdout.strip()) if format_result.stdout.strip() else 0
    
    print(f"📊 Video stream duration: {video_duration:.2f}s")
    print(f"📊 Audio stream duration: {audio_duration:.2f}s")
    print(f"📊 Container duration: {format_duration:.2f}s")
    
    # Check for significant differences
    if abs(video_duration - audio_duration) > 0.5:  # More than 0.5 seconds difference
        print(f"⚠️ Warning: Audio and video durations differ by {abs(video_duration - audio_duration):.2f} seconds")
        return False
    
    print("✅ Audio-video synchronization looks good")
    return True

def ensemble_image_classification(images):
    """Use multiple models and combine their predictions"""
    if not isinstance(images, list):
        images = [images]
        
    # Get predictions from primary model
    primary_results = primary_image_classifier(images)
    if not isinstance(primary_results[0], list):
        primary_results = [primary_results]
        
    # Get predictions from secondary model
    secondary_results = secondary_image_classifier(images)
    if not isinstance(secondary_results[0], list):
        secondary_results = [secondary_results]

    ensemble_scores = []
    for i in range(len(images)):
        # Extract NSFW scores from primary model
        primary_nsfw = next((r["score"] for r in primary_results[i] 
                            if "nsfw" in r["label"].lower() or "porn" in r["label"].lower()), 0.0)
        
        # Extract NSFW scores from secondary model
        secondary_nsfw = next((r["score"] for r in secondary_results[i] 
                              if "nsfw" in r["label"].lower()), 0.0)
        
        # Weighted ensemble
        ensemble_score = (0.10 * primary_nsfw) + (0.90 * secondary_nsfw)
        ensemble_scores.append(ensemble_score)
        
    return ensemble_scores

def load_nsfw_words(words_file=None):
    """Load NSFW words from a file or use defaults"""
    global nsfw_words

    # First check for the default keyword file in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file_path = os.path.join(script_dir, DEFAULT_KEYWORD_FILE)

    # If a specific file is provided via command line, use that instead
    file_to_use = words_file if words_file else default_file_path

    if os.path.exists(file_to_use):
        try:
            with open(file_to_use, 'r') as f:
                words = [line.strip().lower() for line in f if line.strip()]
            print(f"✅ Loaded {len(words)} NSFW words from {file_to_use}")
            nsfw_words = words
        except Exception as e:
            print(f"⚠️ Error loading NSFW words from file: {str(e)}")
            print(f"⚠️ Using default NSFW word list with {len(DEFAULT_NSFW_WORDS)} words")
            nsfw_words = DEFAULT_NSFW_WORDS
    else:
        print(f"⚠️ NSFW words file not found at {file_to_use}")
        print(f"⚠️ Using default NSFW word list with {len(DEFAULT_NSFW_WORDS)} words")
        nsfw_words = DEFAULT_NSFW_WORDS

    # Convert all words to lowercase for case-insensitive matching
    nsfw_words = [word.lower() for word in nsfw_words]
    return nsfw_words

def get_video_metadata(video_path):
    """Extract and store video metadata for consistent output"""
    global video_metadata

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return None

    # Extract metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Get codec information using ffprobe
    codec_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:
        result = subprocess.run(codec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        codec_info = result.stdout.strip().split('\n')
        codec_name = codec_info[0] if len(codec_info) > 0 else "h264"
        bit_rate = codec_info[1] if len(codec_info) > 1 else "5000k"
    except:
        codec_name = "h264"
        bit_rate = "5000k"

    video_metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration,
        "codec_name": codec_name,
        "bit_rate": bit_rate
    }

    cap.release()
    print(f"📊 Video metadata: {width}x{height}, {fps:.2f} fps, {duration:.2f}s, {codec_name} codec")
    return video_metadata

def extract_frames_consistently(video_path):
    """Extract frames with consistent naming and direct scaling to 1920x1080 resolution"""
    print("🎞 Extracting frames consistently at 1920x1080 resolution...")

    # Get video metadata
    metadata = get_video_metadata(video_path)
    if not metadata:
        return

    # Use ffmpeg for more reliable frame extraction with consistent timing and direct scaling to 1920x1080
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vsync", "0",  # Ensure frame timing is preserved
        "-q:v", "1",    # High quality
        "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",  # Direct scaling without preserving aspect ratio
        os.path.join(FRAME_FOLDER, "frame_%08d.jpg").replace("\\", "/")
    ]

    subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Count extracted frames
    frame_count = len(glob.glob(os.path.join(FRAME_FOLDER, "frame_*.jpg")))
    print(f"✅ Extracted {frame_count} frames consistently at 1920x1080 resolution.")

    return frame_count

def extract_audio_with_enhancement(video_path):
    """Extract audio with enhanced processing"""
    print("🔊 Extracting and enhancing audio...")
    temp_audio = os.path.abspath(os.path.join(TEMP_DIR, "temp_audio.wav"))
    enhanced_audio = os.path.abspath(os.path.join(TEMP_DIR, "enhanced_audio.wav"))

    # Extract audio using ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        temp_audio
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Load audio
    audio, sr = librosa.load(temp_audio, sr=16000)

    # Apply noise reduction (simple implementation)
    y_denoised = librosa.effects.preemphasis(audio)

    # Save enhanced audio
    sf.write(enhanced_audio, y_denoised, sr)

    # Split audio into overlapping chunks for better context
    duration = librosa.get_duration(y=y_denoised, sr=sr)
    chunk_samples = int(AUDIO_CHUNK_DURATION * sr)
    overlap_samples = int(1.0 * sr)  # 1 second overlap
    chunks = []

    for i in range(0, len(y_denoised) - overlap_samples, chunk_samples - overlap_samples):
        chunk = y_denoised[i:i + chunk_samples]
        if len(chunk) < sr:  # Skip chunks shorter than 1 second
            continue
            
        chunk_path = os.path.abspath(os.path.join(AUDIO_FOLDER, f"chunk_{i//chunk_samples}.wav"))
        sf.write(chunk_path, chunk, sr)
        start_time = (i / sr) * 1000  # Start time in ms
        end_time = ((i + len(chunk)) / sr) * 1000  # End time in ms
        chunks.append((chunk_path, start_time, end_time))

    print(f"✅ Extracted {len(chunks)} enhanced audio chunks with overlap.")
    return chunks, enhanced_audio

def process_audio_chunks_with_word_level_detection(audio_chunks):
    """Process audio chunks with improved word-level detection of NSFW words"""
    print("🎧 Analyzing audio content with improved word-level precision...")
    global nsfw_audio_segments
    global nsfw_word_timestamps

    # Ensure we have NSFW words to match
    if not nsfw_words:
        print("⚠️ No NSFW words provided for matching. Audio censoring will be skipped.")
        return

    print(f"🔍 Using {len(nsfw_words)} NSFW words for matching")

    # Store all transcriptions with timestamps
    all_transcripts = []

    # First pass: transcribe all chunks with word timestamps
    for chunk_path, chunk_start_time, chunk_end_time in tqdm(audio_chunks, desc="Transcribing Audio with Word Timestamps"):
        # Use Whisper to get word-level timestamps
        result = audio_model.transcribe(
            chunk_path, 
            temperature=0.0,  # More deterministic
            word_timestamps=True,  # Enable word-level timestamps
            fp16=False        # Better accuracy with fp32
        )
        
        # Process each segment with word timestamps
        for segment in result["segments"]:
            # Calculate absolute segment time (in ms)
            segment_start = segment["start"] * 1000 + chunk_start_time
            segment_end = segment["end"] * 1000 + chunk_start_time
            
            # Debug info
            segment_text = segment["text"].strip()
            debug_log(f"Segment {segment_start/1000:.2f}s-{segment_end/1000:.2f}s: {segment_text}")
            
            # Process each word in the segment
            for word_info in segment.get("words", []):
                word = word_info["word"].lower().strip()
                # Remove leading/trailing punctuation
                word = re.sub(r'^\W+|\W+$', '', word)
                
                if not word:
                    continue
                    
                # Calculate absolute timestamps for this word with high precision
                word_start = word_info["start"] * 1000 + chunk_start_time
                word_end = word_info["end"] * 1000 + chunk_start_time
                
                # Check if this word is in our NSFW list
                for nsfw_word in nsfw_words:
                    # Use word boundary matching to avoid partial matches
                    if word == nsfw_word or re.search(r'\b' + re.escape(nsfw_word) + r'\b', word):
                        print(f"❌ NSFW word '{word}' detected at {word_start/1000:.3f}s - {word_end/1000:.3f}s")
                        
                        # Add padding around the word for better censoring
                        censored_start = max(0, word_start - WORD_PADDING_MS)
                        censored_end = word_end + WORD_PADDING_MS
                        
                        # Store the word timestamp for precise censoring
                        nsfw_word_timestamps.append({
                            "word": word,
                            "matched_keyword": nsfw_word,
                            "start_time": censored_start,
                            "end_time": censored_end,
                            "duration": censored_end - censored_start,
                            "segment_text": segment_text,  # Store context for debugging
                            "chunk_start": chunk_start_time,  # Store chunk info for debugging
                            "chunk_end": chunk_end_time
                        })
                        break  # Stop checking other NSFW words once we find a match

    # Sort word timestamps by start time
    nsfw_word_timestamps.sort(key=lambda x: x["start_time"])

    # Merge overlapping word timestamps
    merged_word_timestamps = []
    if nsfw_word_timestamps:
        current = nsfw_word_timestamps[0]
        
        for next_word in nsfw_word_timestamps[1:]:
            # If this word overlaps with the current merged segment
            if next_word["start_time"] <= current["end_time"]:
                # Extend the current segment
                current["end_time"] = max(current["end_time"], next_word["end_time"])
                current["duration"] = current["end_time"] - current["start_time"]
                # Add the word to the list of words in this segment
                if "words" not in current:
                    current["words"] = [current["word"]]
                current["words"].append(next_word["word"])
            else:
                # Finalize the current segment and start a new one
                if "words" not in current:
                    current["words"] = [current["word"]]
                merged_word_timestamps.append(current)
                current = next_word
        
        # Add the last segment
        if "words" not in current:
            current["words"] = [current["word"]]
        merged_word_timestamps.append(current)

    # Update the global list with merged timestamps
    nsfw_word_timestamps = merged_word_timestamps

    # Also create segment-level detections for compatibility with the rest of the code
    for word_timestamp in nsfw_word_timestamps:
        nsfw_audio_segments.append((
            word_timestamp["start_time"],
            word_timestamp["end_time"],
            1.0  # Confidence score
        ))

    print(f"✅ Found {len(nsfw_word_timestamps)} precise NSFW word instances to censor.")
    
    # Debug: print all detected words with timestamps
    if DEBUG_MODE:
        for i, word in enumerate(nsfw_word_timestamps):
            debug_log(f"Word {i+1}: '{word.get('word', 'unknown')}' at {word['start_time']/1000:.3f}s - {word['end_time']/1000:.3f}s")
    
    return nsfw_word_timestamps

def process_audio_with_word_level_censoring(input_audio_path):
    """Process and censor NSFW audio with word-level precision and improved timing"""
    print("🔊 Processing audio with word-level censoring and improved timing...")

    # If no NSFW word timestamps, just return the original audio
    if not nsfw_word_timestamps:
        print("✅ No NSFW words detected in audio.")
        return input_audio_path

    # Load audio with original sample rate preserved
    audio, sr = librosa.load(input_audio_path, sr=None)
    
    # Store original audio format (mono/stereo)
    is_stereo = len(audio.shape) > 1
    if is_stereo:
        channels = audio.shape[1]
        # Convert to mono for processing
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    # Create a copy for censoring
    if is_stereo:
        censored_audio = audio.copy()
    else:
        censored_audio = audio_mono.copy()
    
    # Get audio duration in milliseconds
    audio_duration_ms = (len(audio_mono) / sr) * 1000
    
    # Debug information
    debug_log(f"Audio duration: {audio_duration_ms/1000:.2f} seconds")
    debug_log(f"Sample rate: {sr} Hz")
    debug_log(f"Audio format: {'Stereo' if is_stereo else 'Mono'}")
    
    # Apply censoring to each detected word with improved timing
    print(f"🔇 Censoring {len(nsfw_word_timestamps)} NSFW words using {AUDIO_CENSORING_METHOD} method...")
    
    # Sort timestamps by start time to process in order
    sorted_timestamps = sorted(nsfw_word_timestamps, key=lambda x: x["start_time"])
    
    for word_info in tqdm(sorted_timestamps, desc="Censoring NSFW Words"):
        word = word_info.get("word", "unknown")
        start_time_ms = word_info["start_time"]
        end_time_ms = word_info["end_time"]
        
        # Add extra padding for better coverage
        start_time_ms = max(0, start_time_ms - 50)  # 50ms extra padding
        end_time_ms = min(audio_duration_ms, end_time_ms + 50)  # 50ms extra padding
        
        # Convert timestamps to sample indices with high precision
        start_sample = int((start_time_ms / 1000) * sr)
        end_sample = int((end_time_ms / 1000) * sr)
        
        # Ensure we're within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_mono), end_sample)
        
        if end_sample <= start_sample:
            print(f"⚠️ Invalid time range for word '{word}': {start_time_ms/1000:.3f}s - {end_time_ms/1000:.3f}s")
            continue
        
        # Calculate fade in/out regions
        fade_samples = int((FADE_DURATION_MS / 1000) * sr)
        fade_in_end = min(start_sample + fade_samples, end_sample)
        fade_out_start = max(start_sample, end_sample - fade_samples)
        
        debug_log(f"Censoring '{word}' at {start_time_ms/1000:.3f}s - {end_time_ms/1000:.3f}s (samples: {start_sample}-{end_sample})")
        
        # Apply selected censoring method with improved timing
        if AUDIO_CENSORING_METHOD == "mute":
            if not is_stereo:
                # Apply fade in
                if fade_in_end > start_sample:
                    fade_in = np.linspace(1, 0, fade_in_end - start_sample)
                    censored_audio[start_sample:fade_in_end] *= fade_in
                
                # Apply fade out
                if fade_out_start < end_sample:
                    fade_out = np.linspace(0, 1, end_sample - fade_out_start)
                    censored_audio[fade_out_start:end_sample] *= fade_out
                
                # Mute the center portion
                censored_audio[fade_in_end:fade_out_start] = 0
            else:
                # For stereo, apply to both channels
                for channel in range(channels):
                    # Apply fade in
                    if fade_in_end > start_sample:
                        fade_in = np.linspace(1, 0, fade_in_end - start_sample)
                        censored_audio[start_sample:fade_in_end, channel] *= fade_in
                    
                    # Apply fade out
                    if fade_out_start < end_sample:
                        fade_out = np.linspace(0, 1, end_sample - fade_out_start)
                        censored_audio[fade_out_start:end_sample, channel] *= fade_out
                    
                    # Mute the center portion
                    censored_audio[fade_in_end:fade_out_start, channel] = 0
            
        elif AUDIO_CENSORING_METHOD == "beep":
            # Generate beep tone
            duration = (end_sample - start_sample) / sr
            t = np.linspace(0, duration, end_sample - start_sample, False)
            beep = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 0.5 amplitude, 1000Hz
            
            # Apply fade in/out to beep
            if fade_in_end > start_sample:
                fade_in_samples = fade_in_end - start_sample
                fade_in = np.linspace(0, 1, fade_in_samples)
                beep[:fade_in_samples] *= fade_in
            
            if fade_out_start < end_sample:
                fade_out_samples = end_sample - fade_out_start
                fade_out = np.linspace(1, 0, fade_out_samples)
                beep[-fade_out_samples:] *= fade_out
            
            # Apply beep to audio
            if not is_stereo:
                censored_audio[start_sample:end_sample] = beep
            else:
                for channel in range(channels):
                    censored_audio[start_sample:end_sample, channel] = beep
            
        elif AUDIO_CENSORING_METHOD == "tone":
            # Generate tone with fade in/out
            duration = (end_sample - start_sample) / sr
            t = np.linspace(0, duration, end_sample - start_sample, False)
            tone = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 0.3 amplitude, 1000Hz
            
            # Apply fade in/out
            fade_in_samples = int(duration * sr * 0.1)  # 10% fade in
            fade_out_samples = int(duration * sr * 0.1)  # 10% fade out
            
            if fade_in_samples > 0:
                fade_in = np.linspace(0, 1, fade_in_samples)
                tone[:fade_in_samples] *= fade_in
            
            if fade_out_samples > 0:
                fade_out = np.linspace(1, 0, fade_out_samples)
                tone[-fade_out_samples:] *= fade_out
            
            # Apply tone to audio
            if not is_stereo:
                censored_audio[start_sample:end_sample] = tone
            else:
                for channel in range(channels):
                    censored_audio[start_sample:end_sample, channel] = tone

    # Save censored audio
    censored_audio_path = os.path.abspath(os.path.join(TEMP_DIR, "censored_audio.wav"))
    sf.write(censored_audio_path, censored_audio, sr)

    print(f"✅ Censored audio saved with improved word-level precision.")
    return censored_audio_path

def detect_nsfw_frames():
    """Detect NSFW frames and group them into continuous segments for smooth censoring"""
    print("🔎 Scanning frames with ensemble detection...")
    global nsfw_segments

    # Get all frame files in order
    frame_files = sorted(glob.glob(os.path.join(FRAME_FOLDER, "frame_*.jpg")), 
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    # Store frame scores for all frames
    frame_scores = {}

    # Process in batches for efficiency
    for i in tqdm(range(0, len(frame_files), BATCH_SIZE), desc="Analyzing Frame Batches"):
        batch_files = frame_files[i:i+BATCH_SIZE]
        
        # Get ensemble scores
        images = [Image.open(p).convert("RGB") for p in batch_files]
        scores = ensemble_image_classification(images)
        
        # Store scores for each frame
        for j, file in enumerate(batch_files):
            frame_num = int(os.path.basename(file).split('_')[1].split('.')[0])
            frame_scores[frame_num] = scores[j]

    # Apply temporal smoothing to all scores
    smoothed_scores = {}
    frame_nums = sorted(frame_scores.keys())

    # Convert to list for smoothing
    score_list = [frame_scores[num] for num in frame_nums]
    smoothed_list = temporal_smoothing(score_list, window_size=7)  # Larger window for smoother results

    # Convert back to dictionary
    for i, num in enumerate(frame_nums):
        smoothed_scores[num] = smoothed_list[i]

    # Identify frames above threshold
    nsfw_frames = []
    for frame_num, score in smoothed_scores.items():
        if score > LOW_CONFIDENCE_THRESHOLD:
            nsfw_frames.append((frame_num, score))

    # Group frames into continuous segments
    if nsfw_frames:
        nsfw_frames.sort(key=lambda x: x[0])  # Sort by frame number
        
        current_segment = {
            "start_frame": nsfw_frames[0][0],
            "end_frame": nsfw_frames[0][0],
            "frames": [nsfw_frames[0][0]],
            "scores": [nsfw_frames[0][1]],
            "max_score": nsfw_frames[0][1]
        }
        
        for frame_num, score in nsfw_frames[1:]:
            # If this frame is consecutive or close to the previous one
            if frame_num - current_segment["end_frame"] <= 120:  # Allow small gaps (3 frames)
                current_segment["end_frame"] = frame_num
                current_segment["frames"].append(frame_num)
                current_segment["scores"].append(score)
                current_segment["max_score"] = max(current_segment["max_score"], score)
            else:
                # Only add segments that are long enough
                if len(current_segment["frames"]) >= MIN_SEGMENT_LENGTH:
                    # Add padding frames for smoother transitions
                    start_with_padding = max(1, current_segment["start_frame"] - SEGMENT_PADDING)
                    end_with_padding = min(max(frame_nums), current_segment["end_frame"] + SEGMENT_PADDING)
                    
                    nsfw_segments.append({
                        "start_frame": start_with_padding,
                        "end_frame": end_with_padding,
                        "original_start": current_segment["start_frame"],
                        "original_end": current_segment["end_frame"],
                        "avg_score": sum(current_segment["scores"]) / len(current_segment["scores"]),
                        "max_score": current_segment["max_score"]
                    })
                
                # Start a new segment
                current_segment = {
                    "start_frame": frame_num,
                    "end_frame": frame_num,
                    "frames": [frame_num],
                    "scores": [score],
                    "max_score": score
                }
        
        # Add the last segment if it's long enough
        if len(current_segment["frames"]) >= MIN_SEGMENT_LENGTH:
            # Add padding frames for smoother transitions
            start_with_padding = max(1, current_segment["start_frame"] - SEGMENT_PADDING)
            end_with_padding = min(max(frame_nums), current_segment["end_frame"] + SEGMENT_PADDING)
            
            nsfw_segments.append({
                "start_frame": start_with_padding,
                "end_frame": end_with_padding,
                "original_start": current_segment["start_frame"],
                "original_end": current_segment["end_frame"],
                "avg_score": sum(current_segment["scores"]) / len(current_segment["scores"]),
                "max_score": current_segment["max_score"]
            })

    print(f"✅ Detected {len(nsfw_segments)} NSFW video segments.")

    # Convert frame numbers to timestamps for the report
    for segment in nsfw_segments:
        segment["start_time"] = frame_to_time(segment["start_frame"])
        segment["end_time"] = frame_to_time(segment["end_frame"])

    return nsfw_segments

def identify_segments_to_trim():
    """Identify NSFW segments longer than threshold that should be trimmed instead of censored"""
    global nsfw_segments
    global trimmed_segments

    segments_to_trim = []
    segments_to_censor = []

    # Convert threshold to milliseconds
    TRIM_THRESHOLD_MS = TRIM_THRESHOLD_SEC * 1000

    print(f"🔍 Identifying NSFW segments to trim (longer than {TRIM_THRESHOLD_SEC} seconds)...")

    for segment in nsfw_segments:
        segment_duration_ms = segment["end_time"] - segment["start_time"]
        
        if segment_duration_ms > TRIM_THRESHOLD_MS:
            print(f"✂️ Will trim segment at {segment['start_time']/1000:.2f}s - {segment['end_time']/1000:.2f}s (Duration: {segment_duration_ms/1000:.2f}s)")
            segments_to_trim.append(segment)
            # Store in global trimmed_segments for reporting
            trimmed_segments.append(segment)
        else:
            segments_to_censor.append(segment)

    # Update the global nsfw_segments to only include segments that will be censored
    nsfw_segments = segments_to_censor

    print(f"✅ Identified {len(segments_to_trim)} segments to trim and {len(segments_to_censor)} segments to censor")
    return segments_to_trim

def apply_blur_to_all_segments():
    """Apply blur to all detected NSFW segments."""
    print(f"🖌 Applying blur to {len(nsfw_segments)} video segments...")
    for segment in tqdm(nsfw_segments, desc="Applying Blur"):
        apply_blur_to_segment(segment)
    print("✅ Blur applied to all segments.")

def apply_blur_to_segment(segment):
    """Apply blur to a specific segment."""
    start_frame = segment["start_frame"]
    end_frame = segment["end_frame"]

    for frame_num in range(start_frame, end_frame + 1):
        frame_path = os.path.join(FRAME_FOLDER, f"frame_{frame_num:08d}.jpg")
        if not os.path.exists(frame_path):
            continue

        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Apply blur
        blurred_img = apply_censoring_with_strength(img, "blur", 1.0, segment["max_score"])

        # Save the blurred frame
        cv2.imwrite(frame_path, blurred_img)

def apply_censoring_with_strength(img, method, strength, confidence_score):
    """Apply censoring with variable strength for smooth transitions"""
    h, w = img.shape[:2]

    if method == "blur":
        # Adaptive blur strength based on confidence and transition strength
        base_blur = int(50 + (confidence_score * 50))
        blur_strength = int(base_blur * strength)
        blur_strength = max(1, blur_strength)
        # Ensure blur_strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        if strength < 1.0:
            # For transitions, blend between original and blurred
            blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 30)
            return cv2.addWeighted(img, 1 - strength, blurred, strength, 0)
        else:
            # Full strength blur
            return cv2.GaussianBlur(img, (blur_strength, blur_strength), 50)
        
    elif method == "pixelate":
        # Adaptive pixelation based on confidence and transition strength
        base_block = int(10 + (confidence_score * 20))
        block_size = max(1, int(base_block * strength))
        
        if strength < 1.0:
            # For transitions, blend between original and pixelated
            small = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            return cv2.addWeighted(img, 1 - strength, pixelated, strength, 0)
        else:
            # Full strength pixelation
            small = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
    elif method == "black":
        # Fade to black based on transition strength
        black_img = np.zeros_like(img)
        return cv2.addWeighted(img, 1 - strength, black_img, strength, 0)

    else:
        # Default to blur if unknown method
        blur_strength = int(75 * strength)
        if blur_strength % 2 == 0:
            blur_strength += 1
        blur_strength = max(1, blur_strength)
        
        if strength < 1.0:
            blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
            return cv2.addWeighted(img, 1 - strength, blurred, strength, 0)
        else:
            return cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

def reconstruct_video_with_censored_frames():
    """Reconstruct video from frames with direct scaling to 1920x1080 resolution."""
    print("🎬 Reconstructing video with censored frames at 1920x1080 resolution...")

    # Create temporary video without audio
    temp_video = os.path.abspath(os.path.join(TEMP_DIR, "censored_frames_video.mp4"))

    # Get original video specs
    fps = video_metadata.get("fps", OUTPUT_FRAME_RATE)  # Use target frame rate if metadata is missing
    codec = video_metadata.get("codec_name", "h264")
    bitrate = video_metadata.get("bit_rate", OUTPUT_BITRATE)  # Use target bitrate if metadata is missing

    # Use ffmpeg to create video from frames with controlled frame rate and bitrate
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(FRAME_FOLDER, "frame_%08d.jpg").replace("\\", "/"),
        "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",  # Direct scaling without preserving aspect ratio
        "-c:v", codec if codec in ["h264", "libx264", "hevc", "libx265"] else "libx264",
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-preset", "medium",  # Balance between speed and quality
        "-profile:v", "high",
        "-level", "4.1",
        temp_video
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"✅ Reconstructed video saved with censored frames at 1920x1080 resolution.")
    return temp_video

def combine_video_audio_with_sync(video_path, audio_path):
    """Combine video and audio with full re-encoding to ensure perfect sync."""
    print("🎞 Combining censored video and censored audio with perfect sync...")

    # Create temporary file for the synchronized output
    synced_output = os.path.abspath(os.path.join(TEMP_DIR, "synced_output.mp4"))

    # Use ffmpeg to combine video and audio with precise sync
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",  # Take video from first input
        "-map", "1:a:0",  # Take audio from second input
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",       # End when shortest input ends
        "-async", "1",     # Audio sync method
        synced_output
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(synced_output) and os.path.getsize(synced_output) > 0:
        print("✅ Successfully combined censored video and censored audio with sync.")
        verify_av_sync(synced_output)
        return synced_output
    else:
        print("❌ Failed to combine video and audio.")
        return None

def create_trimmed_video(input_video, segments_to_trim):
    """Create a new video by trimming out NSFW segments with clean cuts."""
    if not segments_to_trim:
        print("✅ No segments to trim, using input video")
        return input_video

    print("✂️ Creating final trimmed video by removing NSFW segments...")

    # Get video duration
    duration_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total_duration_sec = float(result.stdout.strip())
    total_duration_ms = total_duration_sec * 1000

    # Sort segments by start time
    segments_to_trim.sort(key=lambda x: x["start_time"])

    # Create a list of safe segments to keep
    safe_segments = []
    last_end_time = 0

    for segment in segments_to_trim:
        if segment["start_time"] > last_end_time:
            # There's a safe segment before this NSFW segment
            safe_segments.append((last_end_time, segment["start_time"]))
        last_end_time = segment["end_time"]

    # Add the final safe segment if needed
    if last_end_time < total_duration_ms:
        safe_segments.append((last_end_time, total_duration_ms))

    # Check if we have any safe segments to keep
    if not safe_segments:
        print("⚠️ No safe segments found to keep. The entire video contains NSFW content.")
        return input_video

    # Create a temporary file for the final output
    trimmed_video = os.path.abspath(VIDEO_OUTPUT)

    # Create a temporary file for the list of segments
    segments_file = os.path.join(TEMP_DIR, "segments_list.txt")
    with open(segments_file, "w") as f:
        for start_ms, end_ms in safe_segments:
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            duration_sec = end_sec - start_sec

            # Skip very short segments
            if duration_sec < 0.5:
                continue

            # Write segment info
            f.write(f"file '{input_video}'\n")
            f.write(f"inpoint {start_sec}\n")
            f.write(f"outpoint {end_sec}\n")

    # Use ffmpeg to create the trimmed video
    trim_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", segments_file,
        "-c", "copy",  # Use copy to avoid re-encoding
        trimmed_video
    ]

    # Run the command
    process = subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful and the file exists
    if process.returncode == 0 and os.path.exists(trimmed_video) and os.path.getsize(trimmed_video) > 0:
        print(f"✅ Successfully created final trimmed video: {trimmed_video}")
        return trimmed_video
    else:
        print(f"⚠️ Trimming failed with return code {process.returncode}")
        print(f"Error: {process.stderr}")
        
        # Try with re-encoding instead of copy
        print("🔄 Trying with re-encoding instead of copy...")
        reencoded_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", segments_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            trimmed_video
        ]
        
        reencoded_process = subprocess.run(reencoded_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if reencoded_process.returncode == 0 and os.path.exists(trimmed_video) and os.path.getsize(trimmed_video) > 0:
            print(f"✅ Successfully created final trimmed video with re-encoding: {trimmed_video}")
            return trimmed_video
        else:
            print(f"❌ All trimming approaches failed. Using input video.")
            return input_video

def frame_to_time(frame_num):
    """Convert frame number to timestamp in milliseconds with high precision"""
    if video_metadata and video_metadata.get("fps"):
        # Use high precision calculation
        return int((frame_num / video_metadata["fps"]) * 1000)
    return frame_num * 33  # Assume 30fps if metadata not available

def time_to_frame(time_ms):
    """Convert timestamp in milliseconds to frame number with high precision"""
    if video_metadata and video_metadata.get("fps"):
        # Use high precision calculation
        return int((time_ms / 1000) * video_metadata["fps"])
    return int(time_ms / 33)  # Assume 30fps if metadata not available

def temporal_smoothing(scores, window_size=5):
    """Apply temporal smoothing to reduce false positives"""
    if len(scores) <= 1:
        return scores
        
    smoothed = scores.copy()

    # Apply sliding window
    for i in range(len(scores)):
        start = max(0, i - window_size // 2)
        end = min(len(scores), i + window_size // 2 + 1)
        window = scores[start:end]
        
        # Calculate weighted average (center frame has more weight)
        center_weight = 0.4
        side_weight = (1.0 - center_weight) / (end - start - 1) if end - start > 1 else 0
        
        weighted_sum = center_weight * scores[i]
        for j in range(start, end):
            if j != i:
                weighted_sum += side_weight * scores[j]
                
        smoothed[i] = weighted_sum
        
    return smoothed

def verify_output_specs(output_video):
    """Verify that output video has correct specifications and check for A/V sync"""
    print("🔍 Verifying output video specifications...")
    
    # Get output video specs
    output_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        output_video
    ]

    output_result = subprocess.run(output_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output_specs = output_result.stdout.strip().split(',')

    # Check if output resolution is as expected
    if len(output_specs) >= 2:
        width = output_specs[0]
        height = output_specs[1]
        if width == str(OUTPUT_WIDTH) and height == str(OUTPUT_HEIGHT):
            print(f"✅ Output video has the requested {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} resolution.")
        else:
            print(f"⚠️ Warning: Output video resolution is {width}x{height}, not {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.")
    else:
        print("⚠️ Warning: Could not verify output video resolution.")
    
    # Check for A/V sync issues
    verify_av_sync(output_video)
    
    # Get output duration
    output_duration = get_accurate_video_duration(output_video)
    print(f"📊 Output duration: {output_duration:.2f}s")

def generate_report():
    """Generate a detailed report of detections including trimmed segments and word-level censoring"""
    report = {
        "video_segments": [],
        "audio_segments": [],
        "nsfw_words": [],  # New section for word-level detections
        "trimmed_segments": [],
        "statistics": {
            "total_video_segments": len(nsfw_segments),
            "total_audio_segments": len(nsfw_audio_segments),
            "total_nsfw_words": len(nsfw_word_timestamps),
            "total_trimmed_segments": len(trimmed_segments),
            "total_censored_duration_ms": 0,
            "total_trimmed_duration_ms": 0,
            "total_censored_words_duration_ms": 0
        }
    }

    # Process video segments (censored)
    for segment in nsfw_segments:
        report["video_segments"].append({
            "start_time_ms": segment["start_time"],
            "end_time_ms": segment["end_time"],
            "duration_ms": segment["end_time"] - segment["start_time"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "confidence_score": segment["max_score"],
            "time_readable": f"{segment['start_time']/1000:.2f}s - {segment['end_time']/1000:.2f}s"
        })
        report["statistics"]["total_censored_duration_ms"] += segment["end_time"] - segment["start_time"]

    # Process trimmed segments
    for segment in trimmed_segments:
        report["trimmed_segments"].append({
            "start_time_ms": segment["start_time"],
            "end_time_ms": segment["end_time"],
            "duration_ms": segment["end_time"] - segment["start_time"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "confidence_score": segment["max_score"],
            "time_readable": f"{segment['start_time']/1000:.2f}s - {segment['end_time']/1000:.2f}s"
        })
        report["statistics"]["total_trimmed_duration_ms"] += segment["end_time"] - segment["start_time"]

    # Process word-level detections
    for word_info in nsfw_word_timestamps:
        report["nsfw_words"].append({
            "word": word_info.get("word", ""),
            "words": word_info.get("words", [word_info.get("word", "")]),
            "matched_keyword": word_info.get("matched_keyword", ""),
            "start_time_ms": word_info["start_time"],
            "end_time_ms": word_info["end_time"],
            "duration_ms": word_info["duration"],
            "time_readable": f"{word_info['start_time']/1000:.2f}s - {word_info['end_time']/1000:.2f}s"
        })
        report["statistics"]["total_censored_words_duration_ms"] += word_info["duration"]

    # Process audio segments (for backward compatibility)
    for start, end, score in nsfw_audio_segments:
        report["audio_segments"].append({
            "start_time_ms": start,
            "end_time_ms": end,
            "duration_ms": end - start,
            "confidence_score": score,
            "time_readable": f"{start/1000:.2f}s - {end/1000:.2f}s"
        })

    # Calculate total censored and trimmed duration in seconds
    report["statistics"]["total_censored_duration_sec"] = report["statistics"]["total_censored_duration_ms"] / 1000
    report["statistics"]["total_trimmed_duration_sec"] = report["statistics"]["total_trimmed_duration_ms"] / 1000
    report["statistics"]["total_censored_words_duration_sec"] = report["statistics"]["total_censored_words_duration_ms"] / 1000

    # Save report
    report_path = os.path.abspath("nsfw_detection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"✅ Generated detailed detection report: {report_path}")
    return report

def cleanup():
    """Clean up temporary files"""
    print("🧹 Cleaning up temporary files...")

    # Remove temporary directory and all its contents
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print("✅ Cleanup complete")

def process_video(video_path):
    """Main function to process video with the new sequence."""
    print(f"🎬 Processing video: {video_path}")
    print("=" * 50)

    try:
        # Step 1: Extract and prepare
        # Load NSFW words from default file
        load_nsfw_words()
        
        # Extract frames consistently at 1920x1080 resolution
        extract_frames_consistently(video_path)
        
        # Extract and process audio
        audio_chunks, enhanced_audio = extract_audio_with_enhancement(video_path)
        
        # Step 2: Detect NSFW content
        # Detect NSFW words in audio
        process_audio_chunks_with_word_level_detection(audio_chunks)
        
        # Detect NSFW frames in video
        detect_nsfw_frames()
        
        # Identify segments to trim (longer than threshold)
        segments_to_trim = identify_segments_to_trim()
        
        # Step 3: Censor audio first
        print("🔊 STEP 1: Censoring audio by replacing NSFW words...")
        censored_audio = process_audio_with_word_level_censoring(enhanced_audio)
        
        # Step 4: Apply blur to video segments
        print("🖌 STEP 2: Applying blur to NSFW video segments...")
        if nsfw_segments:
            apply_blur_to_all_segments()
        else:
            print("✅ No NSFW video segments to blur.")
        
        # Step 5: Reconstruct video with censored frames
        censored_video = reconstruct_video_with_censored_frames()
        
        # Step 6: Synchronize censored audio with censored video
        print("🔄 STEP 3: Synchronizing censored audio with censored video...")
        synced_output = combine_video_audio_with_sync(censored_video, censored_audio)
        
        # Step 7: Trim video to remove NSFW segments
        if segments_to_trim:
            print("✂️ STEP 4: Trimming video to remove NSFW segments...")
            final_output = create_trimmed_video(synced_output, segments_to_trim)
        else:
            print("✅ No segments to trim, using synchronized output as final video.")
            # Rename synced output to final output
            final_output = os.path.abspath(VIDEO_OUTPUT)
            shutil.copy2(synced_output, final_output)
        
        # Verify final output
        verify_output_specs(final_output)
        
        # Generate detailed report
        report = generate_report()
        
        # Clean up temporary files
        cleanup()
        
        print("\n📊 Processing Statistics:")
        print(f"   NSFW words censored: {len(nsfw_word_timestamps)}")
        print(f"   Video segments blurred: {len(nsfw_segments)}")
        print(f"   Video segments trimmed: {len(trimmed_segments)}")
        print(f"   Total censored audio duration: {report['statistics']['total_censored_words_duration_sec']:.2f} seconds")
        print(f"   Total censored video duration: {report['statistics']['total_censored_duration_sec']:.2f} seconds")
        print(f"   Total trimmed duration: {report['statistics']['total_trimmed_duration_sec']:.2f} seconds")
        print(f"\n✅ Processing complete! Final output: {final_output}")
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NSFW Content Detection and Censoring with Improved Sequence")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output", help="Output filename", default=VIDEO_OUTPUT)
    parser.add_argument("--video-method", choices=["blur", "pixelate", "black"], default=VIDEO_CENSORING_METHOD,
                        help="Video censoring method")
    parser.add_argument("--audio-method", choices=["mute", "beep", "tone"], default=AUDIO_CENSORING_METHOD,
                        help="Audio censoring method")
    parser.add_argument("--transition-frames", type=int, default=TRANSITION_FRAMES,
                        help="Number of frames for smooth transitions")
    parser.add_argument("--trim-threshold", type=float, default=TRIM_THRESHOLD_SEC,
                        help="Threshold in seconds for trimming NSFW segments instead of censoring")
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH,
                        help="Output video width in pixels")
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT,
                        help="Output video height in pixels")
    parser.add_argument("--nsfw-words-file", help="Path to file containing NSFW words, one per line")
    parser.add_argument("--word-padding", type=int, default=WORD_PADDING_MS,
                        help="Padding around detected NSFW words in milliseconds")
    parser.add_argument("--fade-duration", type=int, default=FADE_DURATION_MS,
                        help="Duration of fade in/out for word-level censoring in milliseconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed logging")

    args = parser.parse_args()

    # Update global variables from command line arguments
    VIDEO_OUTPUT = args.output
    VIDEO_CENSORING_METHOD = args.video_method
    AUDIO_CENSORING_METHOD = args.audio_method
    TRANSITION_FRAMES = args.transition_frames
    TRIM_THRESHOLD_SEC = args.trim_threshold
    OUTPUT_WIDTH = args.width
    OUTPUT_HEIGHT = args.height
    WORD_PADDING_MS = args.word_padding
    FADE_DURATION_MS = args.fade_duration
    DEBUG_MODE = args.debug

    # Process the video
    process_video(args.video)
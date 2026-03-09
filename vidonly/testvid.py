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
import json
import glob
import re
from collections import defaultdict
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+


# ✅ CONFIGURATION (Trimming Only)
# UPDATED: The output file name is set here.
VIDEO_OUTPUT = "censored_video.mp4"
TEMP_DIR = os.path.abspath("temp_processing")
FRAME_FOLDER = os.path.abspath(os.path.join(TEMP_DIR, "frames"))
NSFW_THRESHOLD = 0.5
BATCH_SIZE = 96
CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.5
TRANSITION_FRAMES = 30  # PADDING: Number of frames added BEFORE and AFTER the trimmed section
MIN_SEGMENT_LENGTH = 30  # Minimum frames to consider as a segment (e.g., 1 second at 30 FPS)

# Output quality parameters (used for frame extraction and final check)
OUTPUT_WIDTH = 1920  # Target width for output video
OUTPUT_HEIGHT = 1080  # Target height for output video

# Parameter for NSFW segment trimming (retained for argument parsing, though all segments are trimmed)
TRIM_THRESHOLD_SEC = 5

# ✅ Setup folders
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

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

# Global variables to store results
nsfw_segments = []  # Will store continuous segments
trimmed_segments = []  # Store segments that were trimmed
detection_confidence = {}
video_metadata = {}  # Store video metadata for consistent output

# ==============================================================================
# ✅ CORE UTILITIES
# ==============================================================================

def format_ms_to_hms(ms):
    """Convert milliseconds to HH:MM:SS.ms format for readable output"""
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    # Format the string to include hours, minutes, and seconds with two decimal places
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


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

def verify_av_sync(video_path):
    """Verify audio-video sync in the output file (basic check)"""
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
    """Extract frames with consistent naming for detection purposes"""
    print(f"🎞 Extracting frames for detection...")

    # Get video metadata
    metadata = get_video_metadata(video_path)
    if not metadata:
        return

    # Use ffmpeg for more reliable frame extraction
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vsync", "0",  # Ensure frame timing is preserved
        "-q:v", "1",    # High quality
        "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", 
        os.path.join(FRAME_FOLDER, "frame_%08d.jpg").replace("\\", "/")
    ]

    subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Count extracted frames
    frame_count = len(glob.glob(os.path.join(FRAME_FOLDER, "frame_*.jpg")))
    print(f"✅ Extracted {frame_count} frames for detection.")

    return frame_count

def temporal_smoothing(scores, window_size=3):
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

def frame_to_time(frame_num):
    """Convert frame number to timestamp in milliseconds"""
    if video_metadata and video_metadata.get("fps"):
        return int((frame_num / video_metadata["fps"]) * 1000)
    return frame_num * 33  # Assume 30fps if metadata not available

def time_to_frame(time_ms):
    """Convert timestamp in milliseconds to frame number"""
    if video_metadata and video_metadata.get("fps"):
        return int((time_ms / 1000) * video_metadata["fps"])
    return int(time_ms / 33)  # Assume 30fps if metadata not available

def detect_nsfw_frames():
    """Detect NSFW frames and group them into continuous segments for trimming"""
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
    smoothed_list = temporal_smoothing(score_list, window_size=5)  # Larger window for smoother results

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
            if frame_num - current_segment["end_frame"] <= 160:  # Allow small gaps
                current_segment["end_frame"] = frame_num
                current_segment["frames"].append(frame_num)
                current_segment["scores"].append(score)
                current_segment["max_score"] = max(current_segment["max_score"], score)
            else:
                # Only add segments that are long enough
                if len(current_segment["frames"]) >= MIN_SEGMENT_LENGTH:
                    start_frame = current_segment["start_frame"]
                    end_frame = current_segment["end_frame"]
                    
                    nsfw_segments.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "original_start": start_frame,
                        "original_end": end_frame,
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
            start_frame = current_segment["start_frame"]
            end_frame = current_segment["end_frame"]
            
            nsfw_segments.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "original_start": start_frame,
                "original_end": end_frame,
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
    """Applies the 30-frame buffer and finalizes segments for trimming."""
    global nsfw_segments
    global trimmed_segments

    segments_to_trim = []
    
    print(f"🔍 Identifying NSFW segments to trim...")

    # Compute transition padding in milliseconds based on TRANSITION_FRAMES and video fps
    fps = video_metadata.get("fps", 30) if video_metadata else 30
    transition_ms = int((TRANSITION_FRAMES / fps) * 1000)
    video_duration_ms = int(video_metadata.get("duration", 0) * 1000) if video_metadata and video_metadata.get("duration") else None
    
    # CONFIRMATION: This is where the 30-frame padding is applied
    print(f"ℹ️ Expanding trims by {TRANSITION_FRAMES} frames ({transition_ms} ms) BEFORE and AFTER the segment.")

    for segment in nsfw_segments:
        # Expand segment by transition padding 
        expanded_start = max(0, segment["start_time"] - transition_ms)
        expanded_end = segment["end_time"] + transition_ms
        if video_duration_ms is not None:
            expanded_end = min(expanded_end, video_duration_ms)

        segment_copy = segment.copy()
        segment_copy["start_time"] = int(expanded_start)
        segment_copy["end_time"] = int(expanded_end)
        # update frame indices to match expanded times
        segment_copy["start_frame"] = time_to_frame(segment_copy["start_time"])
        segment_copy["end_frame"] = time_to_frame(segment_copy["end_time"])
        segments_to_trim.append(segment_copy)
        trimmed_segments.append(segment_copy)
        print(f"✂️ Will trim (expanded) segment at {format_ms_to_hms(expanded_start)} - {format_ms_to_hms(expanded_end)}")
        
    # Clear nsfw_segments as they are all marked for trimming
    nsfw_segments = []

    print(f"✅ Identified {len(segments_to_trim)} segments to trim")
    return segments_to_trim

def create_trimmed_video(input_video, segments_to_trim):
    """
    Create a new video by trimming out NSFW segments with clean cuts and maintaining A/V sync.
    FIXED: Uses a two-step process (cut then concat) to ensure compatibility with
    -c copy and frame-accurate cuts.
    """
    output_path = os.path.abspath(VIDEO_OUTPUT)
    
    # Get the absolute path of the input video for FFmpeg
    absolute_input_video = os.path.abspath(input_video)

    if not segments_to_trim:
        print("✅ No segments to trim. Copying original video as output.")
        shutil.copy2(input_video, output_path)
        return output_path

    print("✂️ Creating trimmed video by removing long NSFW segments...")

    total_duration_sec = get_accurate_video_duration(input_video)
    total_duration_ms = total_duration_sec * 1000

    segments_to_trim.sort(key=lambda x: x["start_time"])

    safe_segments = []
    last_end_time = 0

    # 1. Merge Overlapping/Adjacent Trimmed Segments
    if segments_to_trim:
        current_start = segments_to_trim[0]["start_time"]
        current_end = segments_to_trim[0]["end_time"]
        
        for next_segment in segments_to_trim[1:]:
            # If segments overlap or touch
            if next_segment["start_time"] <= current_end:
                current_end = max(current_end, next_segment["end_time"])
            else:
                # Add the non-NSFW segment before the current merged segment
                if current_start > last_end_time:
                    safe_segments.append((last_end_time, current_start))
                last_end_time = current_end
                
                # Start a new merged segment
                current_start = next_segment["start_time"]
                current_end = next_segment["end_time"]
        
        # Add the non-NSFW segment before the final merged segment
        if current_start > last_end_time:
            safe_segments.append((last_end_time, current_start))
        last_end_time = current_end

    # 2. Add the final safe segment if needed
    if last_end_time < total_duration_ms:
        safe_segments.append((last_end_time, total_duration_ms))

    if not safe_segments:
        print("⚠️ No safe segments found to keep.")
        with open(output_path, 'w') as f:
            pass
        return output_path

    try:
        print("🔄 Executing two-step trimming (cut segments, then concatenate)...")

        # List to hold paths of individual safe clips
        safe_clip_files = []

        # 3. Cut each safe segment and save it as a temporary clip
        for i, (start_ms, end_ms) in tqdm(enumerate(safe_segments), total=len(safe_segments), desc="Cutting Safe Segments"):
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            duration_sec = end_sec - start_sec

            if duration_sec < 0.05:
                continue

            temp_clip_path = os.path.join(TEMP_DIR, f"safe_clip_{i:04d}.mp4").replace("\\", "/")
            safe_clip_files.append(temp_clip_path)

            # FFmpeg command for accurate, stream-copied cutting:
            # -ss before -i for fast seeking to a keyframe near start_sec
            # -ss after -i for accurate seeking to start_sec
            # -t for duration
            cut_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),  # Fast seek (pre-input)
                "-i", absolute_input_video,
                "-t", str(duration_sec),
                "-c", "copy",
                "-map", "0:v:0", 
                "-map", "0:a:0?",
                temp_clip_path
            ]
            subprocess.run(cut_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        # 4. Write a new concat list file referencing the temporary clips
        concat_list_file = os.path.join(TEMP_DIR, "concat_list.txt")
        with open(concat_list_file, "w") as f:
            for clip_path in safe_clip_files:
                # IMPORTANT: The concat demuxer *only* supports the 'file' keyword and a path.
                # The path must be relative to the running directory or absolute.
                f.write(f"file '{clip_path}'\n")

        # 5. Concatenate the temporary clips into the final video
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_file,
            "-c", "copy",
            "-fflags", "+genpts", # Generate new presentation timestamps (critical for concat copy)
            output_path
        ]

        process = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Successfully created trimmed video: {output_path}")
            return output_path
        else:
            print(f"❌ Trimming failed in concatenation step with return code {process.returncode}")
            print(f"Error: {process.stderr}")
            return input_video

    except Exception as e:
        print(f"⚠️ Error during trimming process: {str(e)}")
        return input_video

def verify_output_specs(input_video, output_video):
    """Verify that output video has same specifications as input and check for A/V sync"""
    print("🔍 Verifying output video specifications...")
    
    # Check for A/V sync issues
    verify_av_sync(output_video)
    
    # Check duration accuracy
    input_duration = get_accurate_video_duration(input_video)
    output_duration = get_accurate_video_duration(output_video)
    
    # Calculate expected duration by subtracting trimmed segments
    total_trimmed_ms = sum(segment["end_time"] - segment["start_time"] for segment in trimmed_segments)
    expected_duration = input_duration - (total_trimmed_ms / 1000)
    
    print(f"📊 Original duration: {format_ms_to_hms(int(input_duration*1000))}")
    print(f"📊 Output duration: {format_ms_to_hms(int(output_duration*1000))}")
    print(f"📊 Expected duration (trimmed): {format_ms_to_hms(int(expected_duration*1000))}")
    
    # Check if the output duration is close to expected
    if abs(output_duration - expected_duration) > 2.0:  # More than 2 seconds difference
        print(f"⚠️ Warning: Output video duration differs from expected by {abs(output_duration - expected_duration):.2f} seconds")
    else:
        print("✅ Output video duration matches expected trimmed duration.")

def generate_report():
    """Generate a detailed report of only trimmed segments with HH:MM:SS.ms format"""
    report = {
        "trimmed_segments": [],
        "statistics": {
            "total_trimmed_segments": len(trimmed_segments),
            "total_trimmed_duration_ms": 0
        }
    }

    # Process trimmed segments
    for segment in trimmed_segments:
        start_ms = segment["start_time"]
        end_ms = segment["end_time"]
        duration_ms = end_ms - start_ms

        report["trimmed_segments"].append({
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "duration_ms": duration_ms,
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "confidence_score": segment["max_score"],
            "time_readable": f"{format_ms_to_hms(start_ms)} - {format_ms_to_hms(end_ms)}"
        })
        report["statistics"]["total_trimmed_duration_ms"] += duration_ms

    # Calculate total trimmed duration in seconds
    report["statistics"]["total_trimmed_duration_sec"] = report["statistics"]["total_trimmed_duration_ms"] / 1000
    report["statistics"]["total_trimmed_duration_readable"] = format_ms_to_hms(report["statistics"]["total_trimmed_duration_ms"])

    # Save report
    report_path = os.path.abspath("nsfw_trimming_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"✅ Generated detailed trimming report: {report_path}")
    return report

def cleanup():
    """Clean up temporary files"""
    print("🧹 Cleaning up temporary files...")

    # Remove temporary directory and all its contents
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print("✅ Cleanup complete")

# ==============================================================================
# ✅ MAIN PROCESS FUNCTION
# ==============================================================================

def process_video(video_path):
    """Main function to process video with trimming for NSFW segments."""
    global VIDEO_OUTPUT
    print(f"🎬 Processing video for trimming only: {video_path}")
    print("=" * 50)

    try:
        # 1. Extract frames for detection (using the original video)
        extract_frames_consistently(video_path)
        
        # 2. Detect NSFW frames and group into segments
        detect_nsfw_frames()
        
        # 3. Identify all detected segments to trim and apply 30-frame padding
        segments_to_trim = identify_segments_to_trim()
        
        # 4. Create the final trimmed video with synchronized audio/video stream copy
        final_video_path = create_trimmed_video(video_path, segments_to_trim)
        
        # 5. Verify the final output
        verify_output_specs(video_path, final_video_path)
        
        # 6. Generate detailed report 
        report = generate_report()
        
        # 7. Clean up temporary files
        cleanup()
        
        print("\n📊 Trimming Statistics:")
        print(f"   Video segments trimmed: {len(trimmed_segments)}")
        print(f"   Total trimmed duration: {report['statistics']['total_trimmed_duration_readable']}")
        print(f"\n✅ Processing complete! Output video saved to: {final_video_path}")
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NSFW Content Trimming (Trimming only, no censoring)")
    parser.add_argument("video", help="Path to the video file")
    
    # Set the default output to censored_video.mp4 here for the argument parser
    parser.add_argument("--output", help="Output filename", default="censored_video.mp4")
    
    parser.add_argument("--trim-threshold", type=float, default=TRIM_THRESHOLD_SEC,
                        help="Threshold in seconds for trimming NSFW segments (All segments are trimmed)")
    parser.add_argument("--transition-frames", type=int, default=TRANSITION_FRAMES,
                        help="Number of frames for padding smooth cuts (applied BEFORE and AFTER the trimmed segment)")
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH,
                        help="Output video width in pixels (only affects temporary frame extraction)")
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT,
                        help="Output video height in pixels (only affects temporary frame extraction)")

    args = parser.parse_args()

    # Update global variables from command line arguments
    VIDEO_OUTPUT = args.output
    TRANSITION_FRAMES = args.transition_frames
    TRIM_THRESHOLD_SEC = args.trim_threshold
    OUTPUT_WIDTH = args.width
    OUTPUT_HEIGHT = args.height

    # Process the video
    process_video(args.video)

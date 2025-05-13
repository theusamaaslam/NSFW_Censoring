import os
import uuid
import json
import time
import shutil
import subprocess
import re
import traceback
import sys
from datetime import datetime
from threading import Thread
from queue import Queue
from flask import Flask, request, session, jsonify, send_file, render_template_string, redirect, url_for, Response
from srt_handler import censor_srt_file


app = Flask(__name__)
app.secret_key = '123456'

# Configuration
UPLOAD_FOLDER = os.path.abspath('uploads')  # Use absolute path
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 5000 * 1024 * 1024  # 5000MB (5GB)


# Global queue for tasks
task_queue = Queue()

def worker():
    while True:
        try:
            file_path, video_dir, video_id, selected_params = task_queue.get()
            print(f"[Worker] Processing job: {video_id}")
            process_video(file_path, video_dir, video_id, selected_params)
            
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            task_queue.task_done()



# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML template for the upload page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Video Moderation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    :root {
      --primary: #1e3a8a;
      --primary-soft: #c7d2fe;
      --accent: #14b8a6;
      --background: #f9fafb;
      --text-dark: #1e293b;
      --text-light: #64748b;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--background);
      color: var(--text-dark);
      padding: 60px 20px;
    }
    .wrapper { max-width: 700px; margin: auto; }
    .header {
      text-align: center;
      margin-bottom: 50px;
    }
    .header h1 {
      font-size: 36px;
      font-weight: 700;
      background: linear-gradient(to right, var(--primary), var(--accent));
      -webkit-background-clip: text;
      color: transparent;
      margin-bottom: 15px;
      padding: 10px 0;
    }
    .header p {
      color: var(--text-light);
      font-size: 16px;
      font-weight: 500;
    }
    .card {
      background: #ffffff;
      border-radius: 18px;
      padding: 40px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.1);
      transition: all 0.3s ease;
    }
    .form-control {
      margin-bottom: 25px;
    }
    .form-row {
      display: flex;
      gap: 20px;
      margin-bottom: 20px;
    }
    .form-row .form-control {
      flex: 1;
      margin-bottom: 0;
    }
    select {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--primary-soft);
      font-size: 16px;
    }
    .file-box {
      border: 2px dashed var(--primary-soft);
      border-radius: 12px;
      padding: 30px;
      text-align: center;
      background-color: #f1f5f9;
      transition: 0.3s;
      margin-bottom: 20px;
    }
    .file-box:hover {
      background-color: var(--primary-soft);
      border-color: var(--primary);
    }
    .file-box p {
      font-size: 15px;
      color: var(--text-light);
      margin-bottom: 20px;
    }
    .file-box label {
      display: inline-block;
      padding: 12px 25px;
      background: var(--primary);
      color: white;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 500;
      font-size: 16px;
      transition: background 0.3s;
    }
    .file-box label:hover {
      background: #1e2a8a;
    }
    input[type="file"] { display: none; }
    .file-name {
      margin-top: 10px;
      font-size: 14px;
      color: var(--text-dark);
      font-style: italic;
    }
    button {
      margin-top: 20px;
      padding: 16px 32px;
      width: 100%;
      background: linear-gradient(to right, var(--primary), var(--accent));
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 18px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 8px 24px rgba(0,188,212,0.15);
    }
    button:hover { opacity: 0.95; transform: translateY(-2px); }
    .status {
      margin-top: 35px;
      padding: 20px;
      border-radius: 14px;
      font-weight: 600;
      text-align: center;
    }
    .processing { background: #fffae5; color: #b45309; }
    .success { background: #d1fae5; color: #065f46; }
    .error { background: #fee2e2; color: #991b1b; }
    .progress-container {
      background: #e2e8f0;
      border-radius: 12px;
      height: 18px;
      margin-top: 20px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(to right, var(--primary), var(--accent));
      width: {{ progress }}%;
      transition: width 0.5s ease-in-out;
    }
    .download-btn {
      display: inline-block;
      margin-top: 30px;
      background: var(--accent);
      color: white;
      padding: 14px 28px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 600;
      transition: background 0.3s ease;
    }
    .download-btn:hover { background: #0f766e; }
    .log-container {
      margin-top: 15px;
      max-height: 300px;
      overflow-y: auto;
      background-color: #f8f9fa;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 10px;
      font-family: monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-all;
      text-align: left;
    }
    .log-line { margin: 2px 0; line-height: 1.4; }
    .log-info { color: #0c5460; }
    .log-warning { color: #856404; }
    .log-error { color: #721c24; }
    .log-success { color: #155724; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>Video Moderation</h1>
      <p>Upload your content to detect and censor sensitive material.</p>
    </div>

    <div class="card">
      
      <div id="full_form">
        <form method="POST" action="/upload" enctype="multipart/form-data" oninput="updateFilename()">
          <div class="form-control">
            <label for="censor_type" style="font-weight: bold;">Select Moderation Type</label>
            <br/><br/>
            <select name="censor_type" id="censor_type" required onchange="toggleFields()">
              <option value="">-- Choose --</option>
              <option value="video">Video</option>
              <option value="srt">SRT Subtitle</option>
            </select>
          </div>

          <div class="file-box" id="video-box" style="display:none;">
            <p>Select a video file (MP4, MOV, AVI, MKV)</p>
            <label for="video">Choose Video</label>
            <input type="file" name="video" id="video" accept="video/*">
            <div class="file-name" id="video-name"></div>
          </div>

          <div class="file-box" id="srt-box" style="display:none;">
            <p>Select a .srt subtitle file</p>
            <label for="srt">Choose SRT</label>
            <input type="file" name="srt" id="srt" accept=".srt" required>
            <div class="file-name" id="srt-name"></div>
          </div>

          <div class="file-box" id="json-box" style="display:none;">
            <p>Select a censorship .json file</p>
            <label for="json">Choose JSON</label>
            <input type="file" name="json" id="json" accept=".json" required>
            <div class="file-name" id="json-name"></div>
          </div>


          <div id="params-box" class="hidden">
            <div class="form-row">
              <div class="form-control">
                <label>Video Method</label>
                <select name="video-method" required>
                  {% for val in ["blur", "pixelate", "black"] %}
                    <option value="{{ val }}" {% if val == 'blur' %}selected{% endif %}>{{ val }}</option>
                  {% endfor %}
                </select>
              </div>
              <div class="form-control">
                <label>Audio Method</label>
                <select name="audio-method" required>
                  {% for val in ["mute", "beep", "tone"] %}
                    <option value="{{ val }}" {% if val == 'mute' %}selected{% endif %}>{{ val }}</option>
                  {% endfor %}
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-control">
                <label>Video Width (in pixels)</label>
                <select name="width" required>
                  {% for val in [640,854,1280,1920] %}
                    <option value="{{ val }}" {% if val == 1920 %}selected{% endif %}>{{ val }}</option>
                  {% endfor %}
                </select>
              </div>
              <div class="form-control">
                <label>Video Height (in pixels)</label>
                <select name="height" required>
                  {% for val in [360,480,720,1080] %}
                    <option value="{{ val }}" {% if val == 1080 %}selected{% endif %}>{{ val }}</option>
                  {% endfor %}
                </select>
              </div>
            </div>

            <div class="form-row">
              <div class="form-control">
                <label>Trimming Threshold (in sec)</label>
                <select name="trim-threshold" required>
                  {% for val in [1, 2, 3, 5, 7, 10, 15, 20] %}
                    <option value="{{ val }}" {% if val == 5 %}selected{% endif %}>{{ val }}</option>
                  {% endfor %}
                </select>
              </div>
             
            </div>
           
            
          </div>
          <input type="hidden" id="moderation_status" value="{{ status }}">

          <button type="submit" {% if form_disabled %}disabled{% endif %}>Upload & Start Moderation</button>

        </form>
      </div>
      <div id="loading-spinner" style="display:none; text-align:center; margin-top:10px;">
        <img src="https://i.gifer.com/ZZ5H.gif" alt="Loading..." width="60">
        <p>Uploading & Initializing...</p>
      </div>

      {% if status %}
        <div class="status {{ status_class }}" id="processing-status">
          {{ status_message }}

          {% if video_id %}
              {% if progress == 0 %}
                <div class="warning-message">⚠️ Video is already in processing...</div>
              {% endif %}
            <div class="progress-container">
              <div class="progress-bar" id="progress-bar" style="width: {{ progress }}%"></div>
            </div>
            <div id="progress-percentage">{{ progress }}%</div>
            <div class="log-container" id="processing-log">
              {% for log in logs %}
                <div class="log-line {{ log.type }}">{{ log.message }}</div>
              {% endfor %}
            </div>
          {% endif %}

          {% if download_url %}
            <p>✅ Moderation Completed!</p>
            <a href="{{ download_url }}" class="download-btn"> ⬇  Download  Video</a>
          {% endif %}

          {% if subtitle_url %}
              <a href="{{ subtitle_url }}" class="download-btn"> ⬇  Download  Subtitles</a>
          {% endif %}

        </div>
      {% endif %}
    </div>
  </div>

  <script>
                    function toggleFields() {
              const type = document.getElementById('censor_type').value;
              
              const videoInput = document.getElementById('video');
              const srtInput = document.getElementById('srt');
              const jsonInput = document.getElementById('json');

              document.getElementById('video-box').style.display = (type === 'video') ? 'block' : 'none';
              document.getElementById('srt-box').style.display = (type === 'srt') ? 'block' : 'none';
              document.getElementById('json-box').style.display = (type === 'srt') ? 'block' : 'none';

              document.getElementById('params-box').classList.toggle('hidden', type !== 'video');

              // Add/remove required attributes
              if (type === 'video') {
                videoInput.required = true;
                srtInput.required = false;
                jsonInput.required = false;
              } else {
                videoInput.required = false;
                srtInput.required = true;
                jsonInput.required = true;
              }
            }

        window.addEventListener('DOMContentLoaded', () => {
          const status = document.getElementById("moderation_status").value;
          if (status === "completed") {
            document.getElementById("full_form").style.display = "none";
          }
        });


    function updateFilename() {
        const video = document.getElementById('video');
        const srt = document.getElementById('srt');
        const json = document.getElementById('json');

        if (video) document.getElementById('video-name').textContent = video.files[0]?.name || '';
        if (srt) document.getElementById('srt-name').textContent = srt.files[0]?.name || '';
        if (json) document.getElementById('json-name').textContent = json.files[0]?.name || '';
      }

     document.querySelector("form").addEventListener("submit", function (e) {
          const type = document.getElementById("censor_type").value;
          const video = document.getElementById("video");

          // Validate video field manually
          if (type === "video" && (!video || !video.files.length)) {
            e.preventDefault();
            alert("⚠ Please select a video file before submitting.");
            return;
          }

          document.getElementById("loading-spinner").style.display = "block";
          document.getElementById("full_form").style.display = "none";

          const submitButton = document.querySelector("button[type='submit']");
          if (submitButton.disabled) {
            e.preventDefault();
            alert("⚠ A video is already being processed. Please wait for it to finish.");
          }
        });



    {% if status in ['queued', 'processing'] %}
    document.getElementById("full_form").style.display = "none";
    const videoId = "{{ video_id }}";
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const statusDiv = document.querySelector('#processing-status');
    const logContainer = document.getElementById('processing-log');
    let lastLogCount = {{ logs|length }};
    function checkStatus() {
      fetch(`/status/${videoId}`)
        .then(response => response.json())
        .then(data => {
          if (data.status === 'completed') {
            statusDiv.textContent = 'Processing complete! Your video is ready for download.';
            statusDiv.className = 'status success';
            progressBar.style.width = '100%';
            progressPercentage.textContent = '100%';
            window.location.href = `/?id=${videoId}`;
          } else if (data.status === 'processing') {
            progressBar.style.width = `${data.progress}%`;
            progressPercentage.textContent = `${data.progress}%`;
            if (data.logs && data.logs.length > lastLogCount) {
              logContainer.innerHTML = '';
              for (let i = 0; i < data.logs.length; i++) {
                const log = data.logs[i];
                const logLine = document.createElement('div');
                logLine.className = `log-line ${log.type}`;
                logLine.textContent = log.message;
                logContainer.appendChild(logLine);
              }
              lastLogCount = data.logs.length;
              logContainer.scrollTop = logContainer.scrollHeight;
            }
            setTimeout(checkStatus, 2000);
          } else if (data.status === 'error') {
            statusDiv.textContent = `Error: ${data.message}`;
            statusDiv.className = 'status error';
          }
        })
        .catch(error => {
          console.error('Error checking status:', error);
          setTimeout(checkStatus, 5000);
        });
    }
    if (videoId) {
      setTimeout(checkStatus, 2000);
    }
    {% endif %}
  </script>
</body>
</html>
'''

def safe_update_status(video_id, status, progress=0, message='', logs=None):
    """Update the status file for a video processing job with error handling"""
    try:
        # Ensure the directory exists
        video_dir = os.path.join(UPLOAD_FOLDER, video_id)
      
        os.makedirs(video_dir, exist_ok=True)
        
        status_path = os.path.join(video_dir, 'status.json')
        print(f"DEBUG - video_dir: {video_dir}, status_path: {status_path}")
        status_data = {
            'status': status,
            'progress': progress,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': message
        }
        
        if logs is not None:
            status_data['logs'] = logs
            
        with open(status_path, 'w') as f:
            json.dump(status_data, f)
            
        return True
    except Exception as e:
        print(f"Error updating status: {str(e)}")
        print(traceback.format_exc())
        return False

def process_video(video_path, output_dir, video_id, selected_params):
    """Process the video using the NSFW detection script"""
    logs = [{'type': 'log-info', 'message': '🔍 Starting video processing...'}]
    print(f"DEBUG - video_id: {video_id}, output_dir: {output_dir}")

    try:
        # Update status to processing
        safe_update_status(video_id, 'processing', progress=5, logs=logs)
        
        # Path to the Python script - use absolute path
        script_path = os.path.abspath('final_backend.py')
        
        # Ensure the script exists
        if not os.path.exists(script_path):
            logs.append({'type': 'log-error', 'message': f"❌ Censoring script not found: {script_path}"})
            safe_update_status(video_id, 'error', message=f"Censoring script not found: {script_path}", logs=logs)
            return False
        
        # Create a temporary directory for processing
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use the original video path directly - don't copy or change directories
        input_video_path = os.path.abspath(video_path)
        
        # Log the paths for debugging
        logs.append({'type': 'log-info', 'message': f"📂 Input video path: {input_video_path}"})
        logs.append({'type': 'log-info', 'message': f"📂 Script path: {script_path}"})
        logs.append({'type': 'log-info', 'message': f"📂 Output directory: {os.path.abspath(output_dir)}"})
        safe_update_status(video_id, 'processing', progress=5, logs=logs)
        
        # Store original directory
        original_dir = os.getcwd()
        
        # Run the script with real-time output capture - pass the ABSOLUTE path to the input video
        # cmd = ["python", script_path, input_video_path]
        cmd = [sys.executable, script_path, input_video_path] + [f"--{k}={v}" for k, v in selected_params.items()]
        cmd_str = " ".join(cmd)
        logs.append({'type': 'log-info', 'message': f"🚀 Running command: {cmd_str}"})
        safe_update_status(video_id, 'processing', progress=5, logs=logs)
        
        # Set the working directory to the output directory so any files created
        # by the script will be in the right place
        os.chdir(output_dir)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,encoding='utf-8',  # 👈 force proper decoding
            errors='replace'   # 👈 avoids crashing on bad characters
        )
        
        # Process output in real-time
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
                
            # Print the line to server console for debugging
            print(f"Script output: {line}")
                
            # Determine log type
            log_type = 'log-info'
            if '❌' in line:
                log_type = 'log-warning'
            elif 'Error' in line or 'error' in line:
                log_type = 'log-error'
            elif '✅' in line:
                log_type = 'log-success'
                
            # Add to logs
            logs.append({'type': log_type, 'message': line})
            
            # Extract progress information
            progress = 5  # Default progress
            
            # Check for loading model
            if 'Loading NSFW Model' in line:
                progress = 10
                
            # Check for frame extraction
            elif 'frames extracted' in line:
                progress = 15
                
            # Check for processing progress
            elif 'Processing:' in line:
                # Extract percentage from the line
                match = re.search(r'Processing:\s+(\d+)%', line)
                if match:
                    # Scale the processing percentage to be between 15-90%
                    processing_percent = int(match.group(1))
                    progress = 15 + int(processing_percent * 0.75)  # Scale to 15-90%
            
            # Check for reconstruction
            elif 'Reconstructing censored video' in line:
                progress = 90
                
            # Check for writing video progress
            elif 'Writing Video:' in line:
                match = re.search(r'Writing Video:\s+(\d+)%', line)
                if match:
                    writing_percent = int(match.group(1))
                    progress = 90 + int(writing_percent * 0.05)  # Scale to 90-95%
                    
            # Check for merging audio
            elif 'Merging and muting audio' in line:
                progress = 95
                
            # Check for completion
            elif 'Final video with audio censorship saved' in line or 'Removed frames' in line:
                progress = 100
                
            # Update status with new progress and logs
            safe_update_status(video_id, 'processing', progress=progress, logs=logs)
            
        # Wait for process to complete
        process.wait()
        
        # Check if the output file was created - look for it in the current directory
        # or in the output directory (depending on how the script works)
        output_file_paths = [
            os.path.join(output_dir, "censored_output.mp4"),  # Output directory
            "censored_output.mp4",  # Current directory (which should be output_dir)
            os.path.join(os.path.dirname(input_video_path), "censored_output.mp4")  # Same directory as input
        ]
        
        output_file_found = False
        for output_path in output_file_paths:
            if os.path.exists(output_path):
                # If the file is not already in the output directory, copy it there
                target_path = os.path.join(output_dir, "censored_output.mp4")
                if os.path.abspath(output_path) != os.path.abspath(target_path):
                    try:
                        shutil.copy(output_path, target_path)
                        logs.append({'type': 'log-info', 'message': f'📂 Copied output file from {output_path} to {target_path}'})
                    except shutil.SameFileError:
                        # If they're the same file, no need to copy
                        logs.append({'type': 'log-info', 'message': f'📂 Output file already in correct location: {output_path}'})
                else:
                    logs.append({'type': 'log-info', 'message': f'📂 Output file already in correct location: {output_path}'})
                
                output_file_found = True
                logs.append({'type': 'log-success', 'message': f'✅ Found output file at: {output_path}'})
                break
        
        if output_file_found:
            # Add final log entry
            logs.append({'type': 'log-success', 'message': '✅ Processing complete! Video saved and ready for download.'})
            safe_update_status(video_id, 'completed', progress=100, logs=logs)
            return True
        else:
            logs.append({'type': 'log-error', 'message': '❌ Processing failed to generate output file'})
            logs.append({'type': 'log-info', 'message': f'📂 Checked paths: {", ".join(output_file_paths)}'})
            safe_update_status(video_id, 'error', message="Processing failed to generate output file", logs=logs)
            return False
            
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error processing video: {error_message}")
        print(error_traceback)
        
        logs.append({'type': 'log-error', 'message': f'❌ Error: {error_message}'})
        logs.append({'type': 'log-error', 'message': f'Traceback: {error_traceback}'})
        
        safe_update_status(video_id, 'error', message=error_message, logs=logs)
        return False
    finally:
        # Change back to the original directory
        try:
            os.chdir(original_dir)
        except:
            pass
        
        # Clean up the temp directory and any frames directories
        try:
            # Check for frames directories in multiple locations
            frames_dirs = [
                os.path.join(original_dir, "frames"),
                os.path.join(output_dir, "frames"),
                os.path.join(os.path.dirname(video_path), "frames")
            ]
            
            for frames_dir in frames_dirs:
                if os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir)
                    logs.append({'type': 'log-info', 'message': f'🧹 Cleaned up frames directory: {frames_dir}'})
                
            # Clean up the temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logs.append({'type': 'log-info', 'message': '🧹 Cleaned up temporary files'})
                
            # Final check to ensure the output file exists and status is updated correctly
            final_output_path = os.path.join(output_dir, "censored_output.mp4")
            if os.path.exists(final_output_path):
                # Make sure the status is set to completed
                safe_update_status(video_id, 'completed', progress=100, logs=logs)
            else:
                # If the output file doesn't exist, make sure the status is set to error
                safe_update_status(video_id, 'error', message="Output file not found after processing", logs=logs)
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            print(traceback.format_exc())
            try:
                logs.append({'type': 'log-warning', 'message': f'⚠️ Cleanup warning: {str(e)}'})
                safe_update_status(video_id, 'error' if 'error_message' in locals() else 'completed', logs=logs)
            except:
                pass



# Set maximum content length for uploads
# app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@app.route('/')

# def index():
#     # Detect if user came via a link with ?id=
#     came_from_link = bool(request.args.get('id'))

#     # Manage session & video_id
#     video_id = request.args.get('id')
#     if video_id:
#         session['video_id'] = video_id
#     else:
#         video_id = None

#     # Init values
#     status = None
#     status_class = ''
#     status_message = ''
#     progress = 0
#     download_url = None
#     subtitle_url = None
#     logs = []
#     form_disabled = False
#     warning_message = ''

#     # Check status if video_id exists
#     if video_id:
#         video_dir = os.path.join(UPLOAD_FOLDER, video_id)  # 👈 define it here FIRST
#         status_path = os.path.join(UPLOAD_FOLDER, video_id, 'status.json')
#         # print(f"DEBUG - video_dir: {video_dir}, status_path: {status_path}")

#         if os.path.exists(status_path):
#             try:
#                 with open(status_path, 'r') as f:
#                     status_data = json.load(f)
#                     # print(f"DEBUG - status_data: {status_data}")
#                     status = status_data['status']
#                     # print(f"DEBUG - status: {status}")
#                     progress = status_data.get('progress', 0)
#                     # print(f"DEBUG - progress: {progress}")
#                     logs = status_data.get('logs', [])
                    

#                     if status in ['processing', 'queued']:
#                         status_class = 'processing'
#                         status_message = f'Processing your video... {progress}% complete'
#                         # warning_message = f'⚠ A video is already being processed (ID: {video_id}). Please wait.'
#                         form_disabled = True

#                     elif status == 'completed':
#                         if came_from_link:
#                             status_class = 'success'
#                             status_message = '✅ Processing complete! Your video is ready for download.'
#                             download_url = f'/download/{video_id}'

#                             # 👇 Check for subtitle file and generate subtitle URL
#                             subtitle_path = os.path.join(video_dir, 'output.srt')
#                             if os.path.exists(subtitle_path):
#                                 subtitle_url = f'/download/{video_id}/censored_output.srt'
#                         else:
#                             # Don't show completed message on homepage
#                             status = None

#                     elif status == 'error':
#                         status_class = 'error'
#                         status_message = f'❌ Error: {status_data.get("message", "Unknown error")}'

#             except Exception as e:
#                 print(f"Error reading status file: {str(e)}")
#                 status = 'error'
#                 status_class = 'error'
#                 status_message = f'❌ Error reading status file: {str(e)}'

#     return render_template_string(
#         HTML_TEMPLATE,
#         status=status,
#         status_class=status_class,
#         status_message=status_message,
#         progress=progress,
#         download_url=download_url,
#         subtitle_url=subtitle_url,
#         video_id=video_id,
#         logs=logs,
#         form_disabled=form_disabled,
#         warning_message=warning_message
#     )
@app.route('/')
def index():
    # Detect if user came via a link with ?id=
    came_from_link = bool(request.args.get('id'))

    # Manage session & video_id
    video_id = request.args.get('id')
    if video_id:
        session['video_id'] = video_id
    else:
        video_id = None

    # Init values
    status = None
    status_class = ''
    status_message = ''
    progress = 0
    download_url = None
    subtitle_url = None
    logs = []
    form_disabled = False
    warning_message = ''

    # Check status if video_id exists
    if video_id:
        video_dir = os.path.join(UPLOAD_FOLDER, video_id)
        status_path = os.path.join(video_dir, 'status.json')

        if os.path.exists(status_path):
            try:
                with open(status_path, 'r') as f:
                    status_data = json.load(f)
                    status = status_data['status']
                    progress = status_data.get('progress', 0)
                    logs = status_data.get('logs', [])

                    if status in ['processing', 'queued']:
                        status_class = 'processing'
                        status_message = f'Processing your video... {progress}% complete'
                        form_disabled = True

                    elif status == 'completed':
                        if came_from_link:
                            status_class = 'success'
                            status_message = '✅ Processing complete! Your file is ready for download.'
                            subtitle_path = os.path.join(video_dir, 'censored_output.srt')
                            videoo_path = os.path.join(video_dir, 'censored_output.mp4')
                            if os.path.exists(subtitle_path):
                                subtitle_url = f'/download/{video_id}/censored_output.srt'
                            elif os.path.exists(videoo_path):
                                download_url = f'/download/{video_id}'                            
                            else:
                                status_message += " (⚠ No file found)"
                        else:
                            status = None  # hide on homepage

                    elif status == 'error':
                        status_class = 'error'
                        status_message = f'❌ Error: {status_data.get("message", "Unknown error")}'

            except Exception as e:
                print(f"Error reading status file: {str(e)}")
                status = 'error'
                status_class = 'error'
                status_message = f'❌ Error reading status file: {str(e)}'

    return render_template_string(
        HTML_TEMPLATE,
        status=status,
        status_class=status_class,
        status_message=status_message,
        progress=progress,
        download_url=download_url,
        subtitle_url=subtitle_url,
        video_id=video_id,
        logs=logs,
        form_disabled=form_disabled,
        warning_message=warning_message
    )

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():

    """Handle file upload"""
    censor_type = request.form.get('censor_type')

    if censor_type == 'video':
        if 'video' not in request.files:
            return jsonify({'error': 'No video part'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        try:
            # Create a unique ID for this upload
            video_id = str(uuid.uuid4())
            session['video_id'] = video_id
            video_dir = os.path.join(UPLOAD_FOLDER, video_id)
            os.makedirs(video_dir, exist_ok=True)

            # Save the uploaded file
            filename = os.path.basename(file.filename)
            file_path = os.path.join(video_dir, filename)
            file.save(file_path)

            # ✅ Extract user-selected parameters from form
            selected_params = {
                'video-method': request.form.get('video-method', 'blur'),
                'audio-method': request.form.get('audio-method', 'mute'),
                'width': int(request.form.get('width', 1920)),
                'height': int(request.form.get('height', 1080)),
                # 'LOW_CONFIDENCE_THRESHOLD': float(request.form.get('LOW_CONFIDENCE_THRESHOLD', 0.4)),
                'trim-threshold': int(request.form.get('trim-threshold', 5)),
                # 'OUTPUT_FRAME_RATE': int(request.form.get('OUTPUT_FRAME_RATE', 23)),
                # 'OUTPUT_BITRATE': request.form.get('OUTPUT_BITRATE', '5000k'),
            }

            # ✅ Create status.json with parameters
            status_data = {
                'status': 'queued',
                'progress': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_filename': filename,
                'logs': [],
                'parameters': selected_params
            }
            with open(os.path.join(video_dir, 'status.json'), 'w') as f:
                json.dump(status_data, f)

            # ✅ Enqueue job for background processing
            task_queue.put((file_path, video_dir, video_id, selected_params))
            print(f"[Upload] Queued job: {video_id}, Queue size now: {task_queue.qsize()}")


            return redirect(url_for('index', id=video_id))
        except Exception as e:
            print(f"Error in upload: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    elif censor_type == 'srt':
        srt_file = request.files.get('srt')
        json_file = request.files.get('json')  # Add support for JSON input

        if not srt_file or srt_file.filename == '':
            return jsonify({'error': 'Missing .srt file'}), 400
        if not json_file or json_file.filename == '':
            return jsonify({'error': 'Missing JSON censorship file'}), 400

        try:
            srt_id = str(uuid.uuid4())
            srt_dir = os.path.join(UPLOAD_FOLDER, srt_id)
            os.makedirs(srt_dir, exist_ok=True)

            # Save input files
            input_srt_path = os.path.join(srt_dir, srt_file.filename)
            input_json_path = os.path.join(srt_dir, json_file.filename)
            output_srt_path = os.path.join(srt_dir, 'censored_output.srt')

            srt_file.save(input_srt_path)
            json_file.save(input_json_path)

            # Call updated SRT handler with JSON intervals
            censor_srt_file(input_srt_path, output_srt_path, input_json_path)

            # Save status
            with open(os.path.join(srt_dir, 'status.json'), 'w') as f:
                json.dump({
                    'status': 'completed',
                    'progress': 100,
                    'message': '✅ SRT file trimmed based on censorship segments',
                    'logs': [{'type': 'log-success', 'message': 'SRT trimmed successfully'}]
                }, f)

            return redirect(url_for('index', id=srt_id))

        except Exception as e:
            with open(os.path.join(srt_dir, 'status.json'), 'w') as f:
                json.dump({
                    'status': 'error',
                    'progress': 0,
                    'message': str(e),
                    'logs': [{'type': 'log-error', 'message': str(e)}]
                }, f)
            return redirect(url_for('index', id=srt_id))


    return jsonify({'error': 'Invalid censor type selected'}), 400

# def upload_file():
#     """Handle file upload"""
#     censor_type = request.form.get('censor_type')

#     if censor_type == 'video':
#         if 'video' not in request.files:
#             return jsonify({'error': 'No video part'}), 400
        
#         file = request.files['video']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'File type not allowed'}), 400
        
#         try:
#             # Create a unique ID for this upload
#             video_id = str(uuid.uuid4())
#             session['video_id'] = video_id
#             video_dir = os.path.join(UPLOAD_FOLDER, video_id)
#             os.makedirs(video_dir, exist_ok=True)
            
#             # Save the uploaded file
#             filename = os.path.basename(file.filename)
#             file_path = os.path.join(video_dir, filename)
#             file.save(file_path)
            
#             # Create initial status file
#             with open(os.path.join(video_dir, 'status.json'), 'w') as f:
#                 json.dump({
#                     'status': 'queued',
#                     'progress': 0,
#                     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                     'original_filename': filename,
#                     'logs': []
#                 }, f)
            
#             # Start processing in a background thread
#             task_queue.put((file_path, video_dir, video_id))  # Enqueue job
            
#             return redirect(url_for('index', id=video_id))  # 👈 Classic form redirect
#         except Exception as e:
#             print(f"Error in upload: {str(e)}")
#             print(traceback.format_exc())
#             return jsonify({'error': str(e)}), 500

#     elif censor_type == 'srt':
#         file = request.files.get('srt')
#         if not file or file.filename == '':
#             return jsonify({'error': 'Invalid .srt file'}), 400

#         try:
#             srt_id = str(uuid.uuid4())
#             srt_dir = os.path.join(UPLOAD_FOLDER, srt_id)
#             os.makedirs(srt_dir, exist_ok=True)

#             input_path = os.path.join(srt_dir, file.filename)
#             output_path = os.path.join(srt_dir, 'censored_output.srt')
#             file.save(input_path)

#             # Call the SRT censor function
#             censor_srt_file(input_path, output_path)

#             with open(os.path.join(srt_dir, 'status.json'), 'w') as f:
#                 json.dump({
#                     'status': 'completed',
#                     'progress': 100,
#                     'message': '✅ SRT file censored',
#                     'logs': [{'type': 'log-success', 'message': 'SRT processed successfully'}]
#                 }, f)

#             return redirect(url_for('index', id=srt_id))
#         except Exception as e:
#             with open(os.path.join(srt_dir, 'status.json'), 'w') as f:
#                 json.dump({
#                     'status': 'error',
#                     'progress': 0,
#                     'message': str(e),
#                     'logs': [{'type': 'log-error', 'message': str(e)}]
#                 }, f)
#             return redirect(url_for('index', id=srt_id))

#     return jsonify({'error': 'Invalid censor type selected'}), 400

@app.route('/status/<video_id>')
def check_status(video_id):
    """Check the status of a video processing job"""
    status_path = os.path.join(UPLOAD_FOLDER, video_id, 'status.json')
    if not os.path.exists(status_path):
        return jsonify({'error': 'Video not found', 'status': 'error'}), 404
    
    try:
        with open(status_path, 'r') as f:
            status_data = json.load(f)
        
        return jsonify(status_data)
    except Exception as e:
        print(f"Error reading status file: {str(e)}")
        return jsonify({'error': f'Error reading status file: {str(e)}', 'status': 'error'}), 500

@app.route('/download/<video_id>')
def download_file(video_id):
    video_dir = os.path.join(UPLOAD_FOLDER, video_id)
    output_path = os.path.join(video_dir, 'censored_output.mp4')
    status_path = os.path.join(video_dir, 'status.json')

    if not os.path.exists(status_path):
        return jsonify({'error': 'Video not found'}), 404

    try:
        with open(status_path, 'r') as f:
            status_data = json.load(f)

        if status_data['status'] != 'completed':
            return jsonify({'error': 'Video processing not complete'}), 400

        # Retry for 2 seconds max to ensure the file exists
        for _ in range(4):
            if os.path.exists(output_path):
                break
            time.sleep(0.5)

        if not os.path.exists(output_path):
            return jsonify({'error': 'Processed video not found'}), 404

        # Ensure the correct file is served
        original_filename = status_data.get('original_filename', 'video.mp4')
        download_name = f'censored_{original_filename}'

        return send_file(
            output_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='video/mp4'
        )

    except Exception as e:
        print("❌ Download error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/download/<video_id>/censored_output.srt')
def download_srt(video_id):
    path = os.path.join(UPLOAD_FOLDER, video_id, 'censored_output.srt')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Censored .srt file not found", 404

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


if __name__ == '__main__':
    # Start one background worker thread
    Thread(target=worker, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5005)

    

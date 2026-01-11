"""
-------------------------------------------------------------------------
SETUP INSTRUCTIONS (Windows/Conda)
-------------------------------------------------------------------------
This script was tested on python 3.10+ using Miniconda.

1. Install Miniconda (if you don't have it):
   https://docs.conda.io/en/latest/miniconda.html

2. Create and activate a new conda environment:
   conda create -n spleeter_env python=3.10 -y

3. Activate the environment:
   conda activate spleeter_env

4. Install & verify FFmpeg:
   conda install -c conda-forge ffmpeg=4.4.* -y
   & "$env:CONDA_PREFIX\Library\bin\ffmpeg.exe" -version
   & "$env:CONDA_PREFIX\Library\bin\ffprobe.exe" -version

   You MUST see version information printed.
   If nothing prints, Spleeter WILL NOT WORK.  

5. Install Spleeter:
    pip install spleeter

You may now run this script in the "spleeter" environment via:
   python Spleeter_AI.py "path_to_your_audio_file"
   or hardcode a path in the script and run:
   python Spleeter_AI.py
-------------------------------------------------------------------------
"""

import os
import sys
import subprocess

def separate_audio(input_file):
    # --- SETUP ---
    input_path = os.path.abspath(input_file)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
    output_root = os.path.join(os.path.dirname(input_path), "Music_Spleeter_Output")
    
    # Spleeter creates a folder named after the filename (e.g., "Song.mp3" -> "Song")
    filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
    final_output_dir = os.path.join(output_root, filename_no_ext)

    print(f"--- PROCESSING: {filename_no_ext} ---")

    # --- RUN SPLEETER ---
    # We use 4stems: vocals, drums, bass, other
    cmd = ["spleeter", "separate", "-p", "spleeter:4stems", "-o", output_root, input_path]
    
    print("1. Running separation...")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Spleeter failed or is not installed.")
        return

    # --- CREATE 'REST.WAV' (Mix Bass + Other) ---
    bass_path = os.path.join(final_output_dir, "bass.wav")
    other_path = os.path.join(final_output_dir, "other.wav")
    rest_path = os.path.join(final_output_dir, "rest.wav")

    if os.path.exists(bass_path) and os.path.exists(other_path):
        print("2. Mixing Bass + Other into 'rest.wav'...")
        
        # FFmpeg command: Mix 2 inputs, maintain volume
        mix_cmd = [
            "ffmpeg", "-y", "-v", "error", # -v error suppresses logs
            "-i", bass_path, 
            "-i", other_path,
            "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=0,volume=2",
            "-ac", "2", 
            rest_path
        ]
        
        try:
            subprocess.run(mix_cmd, check=True)
            # Cleanup only if mix succeeded
            os.remove(bass_path)
            os.remove(other_path)
            print("Created rest.wav and cleaned up.")
        except subprocess.CalledProcessError:
            print("FFmpeg mixing failed. Keeping original files.")
    
    print(f"\nDONE! Output location:\n{final_output_dir}")

if __name__ == "__main__":
    # Handle arguments or drag-and-drop
    if len(sys.argv) > 1:
        separate_audio(sys.argv[1])
    else:
        # Hardcoded default
        default_file = "music_files/Supertramp - Child Of Vision.mp3"
        if os.path.exists(default_file):
            separate_audio(default_file)
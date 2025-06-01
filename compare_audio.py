import argparse
import os
import platform
import subprocess
import shutil
import numpy as np
import scipy.io.wavfile as wavfile
import json

# Default frequencies for analysis, aligned with AudioWatermarker
DEFAULT_CARRIER_FREQ = 19000
DEFAULT_SECONDARY_FREQ = 19500

def analyze_audio_frequencies(file_path, carrier_freq=DEFAULT_CARRIER_FREQ, secondary_freq=DEFAULT_SECONDARY_FREQ):
    """
    Analyzes an audio file to find energy around specified frequencies.
    Adapted from AudioWatermarker.extract_watermark.
    """
    try:
        sample_rate, audio_data = wavfile.read(file_path)

        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483647.0
            elif audio_data.dtype == np.uint8:
                audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
            else: # Default or unknown
                audio_float = audio_data.astype(np.float32) / 32767.0 # Assuming int16 scaling as a fallback
        else:
            audio_float = audio_data.copy()

        if len(audio_float.shape) > 1 and audio_float.shape[1] > 1: # Stereo
            audio_channel = audio_float[:, 0] # Analyze first channel
        else: # Mono
            audio_channel = audio_float

        if len(audio_channel) == 0:
            return {"error": "Empty audio channel data"}

        fft_data = np.fft.rfft(audio_channel)
        freqs = np.fft.rfftfreq(len(audio_channel), 1/sample_rate)

        carrier_mask = np.abs(freqs - carrier_freq) < 100  # 100 Hz tolerance
        secondary_mask = np.abs(freqs - secondary_freq) < 100 # 100 Hz tolerance

        carrier_energy = np.abs(fft_data[carrier_mask]).sum() if np.any(carrier_mask) else 0.0
        secondary_energy = np.abs(fft_data[secondary_mask]).sum() if np.any(secondary_mask) else 0.0

        total_energy = np.sum(np.abs(fft_data))

        return {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "duration_seconds": len(audio_channel) / sample_rate,
            "carrier_frequency_target": carrier_freq,
            "secondary_frequency_target": secondary_freq,
            "carrier_energy": float(carrier_energy),
            "secondary_energy": float(secondary_energy),
            "total_energy": float(total_energy),
            "carrier_ratio": float(carrier_energy / total_energy) if total_energy > 0 else 0.0,
            "secondary_ratio": float(secondary_energy / total_energy) if total_energy > 0 else 0.0,
        }

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Error analyzing audio {file_path}: {e}"}

def get_player_command(audio_file_path):
    """Determines the appropriate command-line audio player for the OS."""
    system = platform.system()
    player_exe_name = None # Will store 'aplay', 'afplay', 'powershell'

    if system == "Linux":
        player_exe_name = "aplay"
    elif system == "Darwin": # macOS
        player_exe_name = "afplay"
    elif system == "Windows":
        player_exe_name = "powershell"
    else:
        print(f"Unsupported OS: {system}. No default player available.")
        return None, None # Return None for command and player name

    player_path = shutil.which(player_exe_name)

    if not player_path:
        print(f"Audio player '{player_exe_name}' not found. Please install it to enable playback.")
        if system == "Linux":
            print("Try: sudo apt-get install alsa-utils")
        elif system == "Darwin":
            print("afplay should be pre-installed. If not, there might be an issue with your OS.")
        elif system == "Windows":
            print("PowerShell should be available. Ensure it's in your PATH and Media.SoundPlayer is accessible.")
        return None, player_exe_name # Return None for command, but provide name for error message

    # Construct cmd using the found player_path
    if system == "Windows":
        # PowerShell command structure is different, uses os.path.abspath for the audio file
        ps_command = f"(New-Object Media.SoundPlayer '{os.path.abspath(audio_file_path)}').PlaySync()"
        cmd = [player_path, "-c", ps_command]
    else:
        # For Linux/macOS, it's just [player_path, audio_file_path]
        # audio_file_path here is relative or absolute as passed in
        cmd = [player_path, audio_file_path]

    return cmd, player_exe_name # Return the command (using full path for player) and the original name for messages

def play_audio(audio_file_path):
    """Plays the audio file using the OS-specific command."""
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found: {audio_file_path}")
        return

    command, player_name = get_player_command(audio_file_path)
    if not command:
        return

    print(f"Attempting to play {os.path.basename(audio_file_path)} using {player_name}...")
    try:
        subprocess.run(command, check=True, capture_output=True) # Using capture_output to hide player's stdout/stderr unless error
        print(f"Finished playing {os.path.basename(audio_file_path)}.")
    except subprocess.CalledProcessError as e:
        print(f"Error playing audio with {player_name}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout.decode(errors='ignore')}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode(errors='ignore')}")
    except FileNotFoundError: # Should be caught by shutil.which, but as a fallback
        print(f"Player '{player_name}' not found. Please ensure it is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during playback: {e}")

def main_logic(original_audio_path, watermarked_audio_path):
    """
    Core logic for comparing audio files, including analysis and playback menu.
    Designed to be callable from tests.
    """
    print("Analyzing original audio...")
    original_analysis = analyze_audio_frequencies(original_audio_path)
    print(json.dumps(original_analysis, indent=2))

    print("\nAnalyzing watermarked audio...")
    watermarked_analysis = analyze_audio_frequencies(watermarked_audio_path)
    print(json.dumps(watermarked_analysis, indent=2))

    if "error" in original_analysis or "error" in watermarked_analysis:
        print("\nExiting due to errors in audio analysis.")
        return 1 # Indicate error

    while True:
        print("\nMenu:")
        print(f"1. Play original audio ({os.path.basename(original_audio_path)})")
        print(f"2. Play watermarked audio ({os.path.basename(watermarked_audio_path)})")
        print("3. Exit")

        try:
            choice = input("Enter your choice (1-3): ")
        except EOFError: # Can happen if stdin is closed, e.g. in some test environments
            print("EOF received, exiting menu.")
            break

        if choice == '1':
            play_audio(original_audio_path)
        elif choice == '2':
            play_audio(watermarked_audio_path)
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")
    return 0 # Indicate success

def cli_entry_point():
    """
    Parses command line arguments and calls the main logic.
    This is the entry point for the script when run from the command line.
    """
    parser = argparse.ArgumentParser(description="Compare original and watermarked audio files.")
    parser.add_argument("original_audio_path", help="Path to the original audio file.")
    parser.add_argument("watermarked_audio_path", help="Path to the watermarked audio file.")

    args = parser.parse_args()

    return main_logic(args.original_audio_path, args.watermarked_audio_path)

if __name__ == "__main__":
    # Run the CLI entry point and exit with its status code (0 for success, 1 for error)
    # Note: main_logic now returns 0 or 1. If it raises an unhandled exception,
    # the script will exit with a non-zero status anyway.
    # For explicit error returns from main_logic (e.g. analysis failure):
    import sys
    sys.exit(cli_entry_point())

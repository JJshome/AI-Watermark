"""
AI Watermark - Example Usage

This script demonstrates how to use the AI Watermark tools programmatically.
It provides examples for both audio and video watermarking.

Based on the patent by Ucaretron Inc. for methods to indicate AI-generated content.
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import cv2
import subprocess

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_watermark import AudioWatermarker
from video_watermark import VideoWatermarker
from watermark_extractor import WatermarkExtractor


def create_sample_audio(output_file, duration=5.0, sample_rate=44100, waveform='sine', frequency=440.0):
    """
    Create a sample audio file for demonstration.
    
    Args:
        output_file: Path to save the audio file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        waveform: Type of waveform ('sine', 'square', 'sawtooth', 'noise')
        frequency: Frequency of the waveform (Hz), ignored for 'noise'
    """
    print(f"Creating sample audio file: {output_file} (waveform: {waveform}, freq: {frequency if waveform != 'noise' else 'N/A'}Hz)")
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if waveform == 'sine':
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    elif waveform == 'square':
        signal = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == 'sawtooth':
        signal = 0.5 * scipy.signal.sawtooth(2 * np.pi * frequency * t)
    elif waveform == 'noise':
        signal = 0.5 * np.random.uniform(-1, 1, size=t.shape)
    else:
        raise ValueError(f"Unsupported waveform: {waveform}. Choose 'sine', 'square', 'sawtooth', or 'noise'.")
    
    # Add a fade in/out
    fade_samples = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    # Convert to int16
    audio_data = (signal * 32767).astype(np.int16)
    
    # Save the audio file
    wavfile.write(output_file, sample_rate, audio_data)
    print(f"Sample audio saved to: {output_file}")
    
    return output_file


def create_sample_video(output_file, duration=5.0, fps=30, width=640, height=480):
    """
    Create a sample video file for demonstration.
    
    Args:
        output_file: Path to save the video file
        duration: Duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    print(f"Creating sample video file: {output_file}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Total frames to generate
    total_frames = int(duration * fps)
    
    # Create the frames
    for i in range(total_frames):
        # Create a frame with a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with a gradient
        for y in range(height):
            for x in range(width):
                # Create a moving gradient
                r = int(128 + 127 * np.sin(x * 0.01 + i * 0.1))
                g = int(128 + 127 * np.sin(y * 0.01 + i * 0.05))
                b = int(128 + 127 * np.sin((x + y) * 0.01 + i * 0.15))
                frame[y, x] = [b, g, r]
        
        # Add a moving circle
        radius = 50
        center_x = int(width/2 + width/4 * np.sin(i * 0.05))
        center_y = int(height/2 + height/4 * np.cos(i * 0.05))
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), -1)
        
        # Add frame text
        cv2.putText(frame, f"Frame {i}/{total_frames}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
    
    # Release resources
    out.release()
    print(f"Sample video saved to: {output_file}")
    
    return output_file


def audio_watermark_example():
    """Demonstrate audio watermarking functionality."""
    print("\n=== Audio Watermarking Example ===\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample audio file
    sample_audio = os.path.join(output_dir, "sample_audio.wav")
    # Generate a 3-second sine wave at 440Hz
    create_sample_audio(sample_audio, waveform='sine', frequency=440.0, duration=3.0, sample_rate=44100)
    
    # Create an AudioWatermarker instance
    watermarker = AudioWatermarker(
        carrier_frequency=19000,  # Hz
        amplitude=0.05           # Relative amplitude
    )
    
    # Add a watermark
    watermarked_audio = os.path.join(output_dir, "watermarked_audio.wav")
    watermarker.add_watermark(
        sample_audio,
        watermarked_audio,
        confidence=0.8,      # 80% confidence
        ai_generated=True    # Mark as AI-generated
    )
    
    # Extract and verify the watermark
    extractor = WatermarkExtractor()
    results = extractor.extract_watermark(
        watermarked_audio,
        output_dir
    )
    
    print("\nWatermark Extraction Results:")
    print(f"Watermark Detected: {'Yes' if results['watermark_detected'] else 'No'}")
    
    if results['watermark_detected']:
        analysis = results['analysis']
        print(f"Carrier Energy: {analysis.get('carrier_energy', 0)}")
        print(f"Secondary Energy: {analysis.get('secondary_energy', 0)}")
        print(f"Likely AI Generated: {'Yes' if analysis.get('likely_ai_generated', False) else 'No'}")

    # After watermarking, call the comparison script
    print("\nLaunching audio comparison tool...")
    # Assuming compare_audio.py is in the parent directory of 'examples'
    compare_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compare_audio.py")

    if os.path.exists(compare_script_path):
        try:
            # Use sys.executable to ensure the script is run with the same Python interpreter
            subprocess.run([sys.executable, compare_script_path, sample_audio, watermarked_audio], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running compare_audio.py: {e}")
        except FileNotFoundError:
            print(f"Error: Python interpreter '{sys.executable}' not found.") # Should not happen
    else:
        print(f"Error: compare_audio.py not found at expected path: {compare_script_path}")
        print("Please ensure 'compare_audio.py' is in the project's root directory.")


def video_watermark_example():
    """Demonstrate video watermarking functionality."""
    print("\n=== Video Watermarking Example ===\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample video file
    sample_video = os.path.join(output_dir, "sample_video.mp4")
    create_sample_video(sample_video, duration=3.0)  # Shorter duration for quicker processing
    
    # Create a VideoWatermarker instance
    watermarker = VideoWatermarker(
        method='histogram',  # Use histogram method
        strength=5           # Medium strength
    )
    
    # Add a watermark
    watermarked_video = os.path.join(output_dir, "watermarked_video.mp4")
    watermarker.add_watermark(
        sample_video,
        watermarked_video,
        confidence=0.9,      # 90% confidence
        ai_generated=True    # Mark as AI-generated
    )
    
    # Extract and verify the watermark
    extractor = WatermarkExtractor()
    results = extractor.extract_watermark(
        watermarked_video,
        output_dir
    )
    
    print("\nWatermark Extraction Results:")
    print(f"Watermark Detected: {'Yes' if results['watermark_detected'] else 'No'}")
    
    if results['watermark_detected']:
        analysis = results['analysis']
        print(f"Confidence: {analysis.get('confidence', 0) * 100:.1f}%")
        print(f"Likely AI Generated: {'Yes' if analysis.get('likely_ai_generated', False) else 'No'}")
        
        method_scores = analysis.get("method_scores", {})
        print("\nDetection by Method:")
        for method, score in method_scores.items():
            print(f"  - {method}: {score * 100:.1f}%")


if __name__ == "__main__":
    print("AI Watermark - Example Usage")
    print("============================")
    print("This script demonstrates how to use the AI Watermark tools programmatically.")
    print("It will create sample audio and video files, add watermarks, and then extract them.")
    
    try:
        audio_watermark_example()
        video_watermark_example()
        
        print("\nExample completed successfully!")
        print("Check the 'examples/output' directory for the generated files and analysis.")
    except Exception as e:
        print(f"\nError during example execution: {e}")

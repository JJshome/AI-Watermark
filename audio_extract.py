#!/usr/bin/env python3
"""
Audio Watermark Extraction Tool

This script provides specialized functionality for extracting and analyzing
watermarks from audio files. It uses the AudioWatermarker class for extraction
and provides additional visualization features.

Based on the patent by Ucaretron Inc. for methods to indicate AI-generated content.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal

# Add parent directory to path for imports if run as script
if __name__ == "__main__":
    # Add the parent directory to the path if running as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_watermark import AudioWatermarker


def analyze_audio_frequencies(audio_file, output_dir=None, freq_range=(18000, 20000)):
    """
    Analyze the high frequency components of an audio file for watermarks.
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save analysis results
        freq_range: Frequency range to analyze (min_hz, max_hz)
        
    Returns:
        dict: Analysis results
    """
    try:
        # Read the audio file
        sample_rate, audio_data = wavfile.read(audio_file)
        
        # Convert to float for processing if needed
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483647.0
            elif audio_data.dtype == np.uint8:
                audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
            else:
                audio_float = audio_data.astype(np.float32) / 32767.0
        else:
            audio_float = audio_data.copy()
        
        # Extract first channel if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_channel = audio_float[:, 0]
        else:
            audio_channel = audio_float
        
        # Calculate audio duration
        duration = len(audio_channel) / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        
        # Create a time array
        t = np.arange(0, duration, 1/sample_rate)
        
        # Compute the spectrogram
        f, t_spec, Sxx = signal.spectrogram(
            audio_channel, 
            fs=sample_rate,
            nperseg=4096,
            noverlap=3072,
            nfft=8192,
            scaling='spectrum'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Focus on the high frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        high_freqs = f[freq_mask]
        high_freq_spectrogram = Sxx_db[freq_mask, :]
        
        # Calculate energy in specified frequency ranges
        energy_19000 = np.sum(Sxx[(f >= 18900) & (f <= 19100), :])
        energy_19500 = np.sum(Sxx[(f >= 19400) & (f <= 19600), :])
        
        # Check if frequencies around 19kHz have abnormal energy
        background_energy = np.mean(Sxx[(f >= 17000) & (f <= 21000), :])
        carrier_ratio = energy_19000 / (background_energy + 1e-10)
        secondary_ratio = energy_19500 / (background_energy + 1e-10)
        
        # Determine if watermark is likely present
        watermark_likely = carrier_ratio > 2.0 or secondary_ratio > 2.0
        
        # Prepare results
        results = {
            "file": audio_file,
            "duration": float(duration),
            "sample_rate": int(sample_rate),
            "analysis": {
                "energy_19000Hz": float(energy_19000),
                "energy_19500Hz": float(energy_19500),
                "background_energy": float(background_energy),
                "carrier_ratio": float(carrier_ratio),
                "secondary_ratio": float(secondary_ratio),
                "watermark_likely": watermark_likely
            }
        }
        
        # Create visualizations if output directory provided
        if output_dir and os.path.exists(output_dir):
            # Create a figure for visualization
            plt.figure(figsize=(12, 10))
            
            # Plot the original waveform
            plt.subplot(3, 1, 1)
            plt.plot(t[:min(len(t), 1000000)], audio_channel[:min(len(t), 1000000)])
            plt.title('Audio Waveform (first part)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Plot full spectrogram
            plt.subplot(3, 1, 2)
            plt.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='viridis')
            plt.title('Full Spectrogram')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.ylim(0, sample_rate / 2)  # Limit to Nyquist frequency
            
            # Plot high frequency spectrogram
            plt.subplot(3, 1, 3)
            plt.pcolormesh(t_spec, high_freqs, high_freq_spectrogram, shading='gouraud', cmap='viridis')
            plt.title('High Frequency Spectrogram (Potential Watermark Region)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.ylim(freq_range)
            
            # Mark the carrier frequency
            plt.axhline(y=19000, color='r', linestyle='--', alpha=0.7)
            plt.text(t_spec[-1] * 0.8, 19000, '19kHz (Carrier)', color='r')
            
            # Mark the secondary frequency
            plt.axhline(y=19500, color='g', linestyle='--', alpha=0.7)
            plt.text(t_spec[-1] * 0.8, 19500, '19.5kHz (Secondary)', color='g')
            
            # Add watermark status note
            if watermark_likely:
                status_text = "WATERMARK DETECTED: Abnormal energy detected in carrier frequencies"
                color = 'red'
            else:
                status_text = "NO WATERMARK DETECTED: Normal energy distribution"
                color = 'green'
            
            plt.figtext(0.5, 0.01, status_text, wrap=True, horizontalalignment='center', fontsize=12, 
                       bbox=dict(facecolor=color, alpha=0.2))
            
            # Save the figure
            base_name = os.path.basename(audio_file)
            file_name, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{file_name}_frequency_analysis.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Analysis visualization saved to: {output_path}")
            
            # Save results as JSON
            json_path = os.path.join(output_dir, f"{file_name}_frequency_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Analysis results saved to: {json_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during frequency analysis: {e}")
        return None


def main():
    """Command line interface for audio watermark extraction and analysis."""
    parser = argparse.ArgumentParser(description='Audio Watermark Extraction Tool')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output-dir', default='./output', 
                      help='Directory to save analysis results')
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='Generate visualizations (default: True)')
    parser.add_argument('--method', choices=['standard', 'frequency'], default='standard',
                      help='Extraction method (standard=use AudioWatermarker, frequency=detailed analysis)')
    parser.add_argument('--min-freq', type=int, default=18000,
                      help='Minimum frequency to analyze (Hz)')
    parser.add_argument('--max-freq', type=int, default=20000,
                      help='Maximum frequency to analyze (Hz)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Extract base filename
    base_name = os.path.basename(args.input)
    file_name, _ = os.path.splitext(base_name)
    
    # Choose extraction method
    if args.method == 'standard':
        # Use the AudioWatermarker for extraction
        watermarker = AudioWatermarker()
        
        output_file = None
        if args.visualize:
            output_file = os.path.join(args.output_dir, f"{file_name}_filtered.wav")
        
        results = watermarker.extract_watermark(args.input, output_file)
        
        if results:
            print("\nExtraction Results:")
            print(f"Watermark Detected: {'Yes' if results.get('watermark_present', False) else 'No'}")
            
            if results.get('watermark_present', False):
                print(f"Carrier Energy: {results.get('carrier_energy', 0)}")
                print(f"Secondary Energy: {results.get('secondary_energy', 0)}")
                print(f"Likely AI Generated: {'Yes' if results.get('likely_ai_generated', False) else 'No'}")
            
            # Save results to JSON
            json_path = os.path.join(args.output_dir, f"{file_name}_watermark_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {json_path}")
            if output_file and os.path.exists(output_file):
                print(f"Filtered audio saved to: {output_file}")
    
    else:  # frequency analysis
        # Perform detailed frequency analysis
        results = analyze_audio_frequencies(
            args.input, 
            args.output_dir,
            freq_range=(args.min_freq, args.max_freq)
        )
        
        if results:
            analysis = results['analysis']
            print("\nFrequency Analysis Results:")
            print(f"Duration: {results['duration']:.2f} seconds")
            print(f"Sample Rate: {results['sample_rate']} Hz")
            print(f"Energy at 19kHz: {analysis['energy_19000Hz']:.8f}")
            print(f"Energy at 19.5kHz: {analysis['energy_19500Hz']:.8f}")
            print(f"Background Energy: {analysis['background_energy']:.8f}")
            print(f"Carrier Ratio: {analysis['carrier_ratio']:.2f}")
            print(f"Secondary Ratio: {analysis['secondary_ratio']:.2f}")
            print(f"Watermark Likely Present: {'Yes' if analysis['watermark_likely'] else 'No'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

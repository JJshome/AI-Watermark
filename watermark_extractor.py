"""
Watermark Extractor Utility

This module provides a unified interface for extracting and analyzing watermarks
from both audio and video files. It combines the functionality of audio_watermark.py
and video_watermark.py extraction methods.

Based on the patent by Ucaretron Inc. for methods to indicate AI-generated content.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from audio_watermark import AudioWatermarker
from video_watermark import VideoWatermarker
import scipy.io.wavfile as wavfile
import cv2


class WatermarkExtractor:
    """Unified watermark extraction utility for audio and video files."""
    
    def __init__(self):
        """Initialize the watermark extractor."""
        self.audio_watermarker = AudioWatermarker()
        self.video_watermarker = VideoWatermarker()
    
    def detect_file_type(self, file_path):
        """
        Detect file type based on extension.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            str: File type ('audio', 'video', or 'unknown')
        """
        _, ext = os.path.splitext(file_path.lower())
        
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        
        if ext in audio_extensions:
            return 'audio'
        elif ext in video_extensions:
            return 'video'
        else:
            return 'unknown'
    
    def extract_watermark(self, input_file, output_dir=None, visualize=True):
        """
        Extract watermark from media file and optionally create visualizations.
        
        Args:
            input_file: Path to the watermarked media file
            output_dir: Directory to save analysis results and visualizations
            visualize: Whether to generate visualizations
            
        Returns:
            dict: Analysis results
        """
        file_type = self.detect_file_type(input_file)
        
        if file_type == 'unknown':
            print(f"Error: Unknown file type for {input_file}")
            return None
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Base filename for outputs
        base_name = os.path.basename(input_file)
        file_name, _ = os.path.splitext(base_name)
        
        # Results data structure
        results = {
            "file": input_file,
            "file_type": file_type,
            "watermark_detected": False,
            "analysis": None
        }
        
        if file_type == 'audio':
            # Extract audio watermark
            output_file = None
            if output_dir and visualize:
                output_file = os.path.join(output_dir, f"{file_name}_filtered.wav")
            
            analysis = self.audio_watermarker.extract_watermark(input_file, output_file)
            
            if analysis:
                results["watermark_detected"] = analysis.get("watermark_present", False)
                results["analysis"] = analysis
                
                # Create visualization if requested
                if visualize and output_dir:
                    self._visualize_audio_analysis(input_file, analysis, 
                                                output_dir, file_name)
        
        elif file_type == 'video':
            # Extract video watermark
            output_file = None
            if output_dir and visualize:
                output_file = os.path.join(output_dir, f"{file_name}_analysis.png")
            
            analysis = self.video_watermarker.extract_watermark(input_file, output_file)
            
            if analysis:
                results["watermark_detected"] = analysis.get("watermark_detected", False)
                results["analysis"] = analysis
                
                # Create visualization if requested
                if visualize and output_dir:
                    self._visualize_video_analysis(input_file, analysis, 
                                                output_dir, file_name)
        
        # Save analysis results as JSON
        if output_dir:
            json_path = os.path.join(output_dir, f"{file_name}_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Analysis results saved to: {json_path}")
        
        return results
    
    def _visualize_audio_analysis(self, input_file, analysis, output_dir, file_name):
        """
        Create visualizations for audio watermark analysis.
        
        Args:
            input_file: Path to the audio file
            analysis: Analysis results
            output_dir: Directory to save visualizations
            file_name: Base name for visualization files
        """
        try:
            # Read the audio file
            sample_rate, audio_data = wavfile.read(input_file)
            
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
            
            # Create a spectrogram visualization
            plt.figure(figsize=(10, 6))
            
            # Waveform subplot
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(len(audio_channel)) / sample_rate, audio_channel)
            plt.title('Audio Waveform')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Spectrogram subplot
            plt.subplot(2, 1, 2)
            
            # Compute spectrogram
            segment_length = 1024
            overlap = segment_length // 2
            
            # Highlight carrier frequency areas if watermark detected
            if analysis.get("watermark_present", False):
                # Mark the carrier frequency in the spectrogram
                carrier_freq = analysis.get("carrier_frequency", 19000)
                secondary_freq = analysis.get("secondary_frequency", 19500)
                
                # Draw frequency markers
                plt.axhline(y=carrier_freq, color='r', linestyle='--', alpha=0.5)
                plt.axhline(y=secondary_freq, color='g', linestyle='--', alpha=0.5)
                plt.text(0, carrier_freq, f"{carrier_freq} Hz", color='r')
                plt.text(0, secondary_freq, f"{secondary_freq} Hz", color='g')
            
            # Plot spectrogram
            plt.specgram(audio_channel, NFFT=segment_length, Fs=sample_rate, 
                        noverlap=overlap, cmap='viridis')
            plt.title('Audio Spectrogram' + 
                     (" (Watermark Detected)" if analysis.get("watermark_present", False) else ""))
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Intensity (dB)')
            
            # Add analysis details
            watermark_status = "✓ Watermark Detected" if analysis.get("watermark_present", False) else "✗ No Watermark"
            plt.figtext(0.5, 0.01, f"File: {os.path.basename(input_file)} | {watermark_status}", 
                      ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            # Save figure
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"{file_name}_spectrogram.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            
            print(f"Audio analysis visualization saved: {out_path}")
            
        except Exception as e:
            print(f"Error creating audio visualization: {e}")
    
    def _visualize_video_analysis(self, input_file, analysis, output_dir, file_name):
        """
        Create visualizations for video watermark analysis.
        
        Args:
            input_file: Path to the video file
            analysis: Analysis results
            output_dir: Directory to save visualizations
            file_name: Base name for visualization files
        """
        try:
            # Create a summary visualization
            plt.figure(figsize=(10, 6))
            
            # Plot method scores
            method_scores = analysis.get("method_scores", {})
            methods = list(method_scores.keys())
            scores = [method_scores.get(method, 0) for method in methods]
            
            # Create bar chart
            plt.subplot(1, 2, 1)
            bars = plt.bar(methods, scores)
            plt.title('Watermark Detection by Method')
            plt.xlabel('Detection Method')
            plt.ylabel('Detection Score')
            plt.ylim(0, 1.0)
            
            # Color bars based on score
            for bar, score in zip(bars, scores):
                if score > 0.3:
                    bar.set_color('green')
                elif score > 0.1:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
            
            # Add threshold line
            plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
            plt.text(0, 0.11, "Detection Threshold", color='r')
            
            # Add pie chart for overall result
            plt.subplot(1, 2, 2)
            watermark_detected = analysis.get("watermark_detected", False)
            likely_ai = analysis.get("likely_ai_generated", False)
            
            if watermark_detected:
                if likely_ai:
                    labels = ['AI Generated\n(High Confidence)', 'AI Generated\n(Low Confidence)']
                    sizes = [analysis.get("confidence", 0.5), 1 - analysis.get("confidence", 0.5)]
                    colors = ['red', 'orange']
                else:
                    labels = ['AI Generated\n(Low Confidence)', 'Undetermined']
                    sizes = [analysis.get("confidence", 0.2), 1 - analysis.get("confidence", 0.2)]
                    colors = ['orange', 'gray']
            else:
                labels = ['No Watermark\nDetected', '']
                sizes = [1, 0]
                colors = ['blue', 'white']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Watermark Analysis Result')
            
            # Add summary text
            summary = "AI Watermark Analysis Summary\n"
            summary += f"File: {os.path.basename(input_file)}\n"
            summary += f"Watermark Detected: {'Yes' if watermark_detected else 'No'}\n"
            if watermark_detected:
                summary += f"Confidence: {analysis.get('confidence', 0) * 100:.1f}%\n"
                summary += f"Likely AI Generated: {'Yes' if likely_ai else 'No'}\n"
            
            plt.figtext(0.5, 0.01, summary, ha='center', fontsize=10, 
                      bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            # Save figure
            plt.tight_layout(rect=[0, 0.1, 1, 0.9])
            out_path = os.path.join(output_dir, f"{file_name}_analysis_summary.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            
            print(f"Video analysis visualization saved: {out_path}")
            
        except Exception as e:
            print(f"Error creating video visualization: {e}")


def main():
    """Command line interface for watermark extraction."""
    parser = argparse.ArgumentParser(description='Watermark Extraction Tool')
    parser.add_argument('--input', required=True, help='Input media file')
    parser.add_argument('--output-dir', default='./output', 
                      help='Directory to save analysis results')
    parser.add_argument('--no-visualize', action='store_true', 
                      help='Disable visualization generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    extractor = WatermarkExtractor()
    results = extractor.extract_watermark(
        args.input, 
        args.output_dir, 
        not args.no_visualize
    )
    
    if results:
        print("\nAnalysis Summary:")
        print(f"File: {os.path.basename(args.input)}")
        print(f"Type: {results['file_type']}")
        print(f"Watermark Detected: {'Yes' if results['watermark_detected'] else 'No'}")
        
        if results['watermark_detected']:
            analysis = results['analysis']
            if results['file_type'] == 'audio':
                print(f"Carrier Energy: {analysis.get('carrier_energy', 0)}")
                print(f"Secondary Energy: {analysis.get('secondary_energy', 0)}")
                print(f"Likely AI Generated: {'Yes' if analysis.get('likely_ai_generated', False) else 'No'}")
            else:  # video
                print(f"Confidence: {analysis.get('confidence', 0) * 100:.1f}%")
                print(f"Likely AI Generated: {'Yes' if analysis.get('likely_ai_generated', False) else 'No'}")
                
                method_scores = analysis.get("method_scores", {})
                print("\nDetection by Method:")
                for method, score in method_scores.items():
                    print(f"  - {method}: {score * 100:.1f}%")


if __name__ == "__main__":
    main()

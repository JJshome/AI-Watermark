#!/usr/bin/env python3
"""
Video Watermark Extraction Tool

This script provides specialized functionality for extracting and analyzing
watermarks from video files. It uses the VideoWatermarker class for extraction
and provides additional visualization features.

Based on the patent by Ucaretron Inc. for methods to indicate AI-generated content.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime

# Add parent directory to path for imports if run as script
if __name__ == "__main__":
    # Add the parent directory to the path if running as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_watermark import VideoWatermarker


def analyze_video_frames(video_file, output_dir, sample_interval=30, max_frames=100):
    """
    Analyze video frames for potential watermarks.
    
    Args:
        video_file: Path to the video file
        output_dir: Directory to save analysis results
        sample_interval: Number of frames to skip between samples
        max_frames: Maximum number of frames to analyze
        
    Returns:
        dict: Analysis results
    """
    try:
        # Open the video file
        video = cv2.VideoCapture(video_file)
        
        if not video.isOpened():
            print(f"Error: Could not open video file {video_file}")
            return None
        
        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video: {width}x{height}, {fps} fps, {frame_count} frames, {duration:.2f} seconds")
        
        # Initialize analysis containers
        frames_analyzed = 0
        histogram_anomalies = 0
        histogram_distances = []
        edge_anomalies = 0
        corner_anomalies = 0
        
        # Initialize arrays for histogram data
        all_histograms = []
        
        # Create a montage of sample frames
        montage_frames = []
        montage_indices = []
        
        # Process frames
        frame_index = 0
        while frames_analyzed < max_frames and frame_index < frame_count:
            # Set the frame position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            
            if not ret:
                break
            
            # Convert to HSV for better analysis
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 1. Histogram analysis
            hist_h = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            
            # Store histogram for later comparison
            all_histograms.append((hist_h, hist_s))
            
            # 2. Check for unusual patterns or anomalies
            
            # 2.1 Check corner regions for anomalies
            # Bottom right corner
            corner_size = 32
            br_corner = frame[-corner_size:, -corner_size:]
            
            # Check corner for unusual patterns (simplified)
            b, g, r = cv2.split(br_corner)
            bg_ratio = np.mean(b) / (np.mean(g) + 0.001)
            
            if bg_ratio > 1.05:  # Threshold
                corner_anomalies += 1
            
            # 2.2 Check for unusual edges (potential pixel-level watermarks)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Unusual edge density can indicate pixel manipulation
            if edge_density > 0.1 and edge_density < 0.15:  # Suspicious range
                edge_anomalies += 1
            
            # Add to montage if it's a key frame
            if frames_analyzed % int(max_frames / 16) == 0 and len(montage_frames) < 16:
                # Resize for the montage
                montage_frame = cv2.resize(frame, (320, 180))
                montage_frames.append(montage_frame)
                montage_indices.append(frame_index)
            
            frames_analyzed += 1
            frame_index += sample_interval
        
        # Compare histograms to detect anomalies
        if len(all_histograms) > 1:
            for i in range(len(all_histograms) - 1):
                h1, s1 = all_histograms[i]
                h2, s2 = all_histograms[i + 1]
                
                # Compare consecutive frame histograms
                h_dist = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                s_dist = cv2.compareHist(s1, s2, cv2.HISTCMP_CORREL)
                
                # Store distances
                histogram_distances.append((h_dist, s_dist))
                
                # Check for histogram anomalies
                # High correlation values (>0.99) might indicate artificial consistency
                # Low values (<0.7) between consecutive frames might indicate manipulation
                if h_dist > 0.995 or h_dist < 0.7 or s_dist > 0.995 or s_dist < 0.7:
                    histogram_anomalies += 1
        
        # Create result structure
        results = {
            "file": video_file,
            "width": width,
            "height": height,
            "fps": float(fps),
            "frame_count": frame_count,
            "duration": float(duration),
            "frames_analyzed": frames_analyzed,
            "analysis": {
                "histogram_anomalies": histogram_anomalies,
                "histogram_anomaly_percentage": float(histogram_anomalies / max(1, frames_analyzed - 1)),
                "corner_anomalies": corner_anomalies,
                "corner_anomaly_percentage": float(corner_anomalies / max(1, frames_analyzed)),
                "edge_anomalies": edge_anomalies,
                "edge_anomaly_percentage": float(edge_anomalies / max(1, frames_analyzed)),
            }
        }
        
        # Calculate watermark likelihood
        hist_score = histogram_anomalies / max(1, frames_analyzed - 1)
        corner_score = corner_anomalies / max(1, frames_analyzed)
        edge_score = edge_anomalies / max(1, frames_analyzed)
        
        # Overall likelihood
        watermark_likelihood = max(hist_score, corner_score, edge_score)
        results["analysis"]["watermark_likelihood"] = float(watermark_likelihood)
        results["analysis"]["watermark_detected"] = watermark_likelihood > 0.1
        
        # Create visualizations if output directory provided
        if output_dir and os.path.exists(output_dir):
            # 1. Create a montage of analyzed frames
            if montage_frames:
                # Create a 4x4 grid (or smaller if fewer frames)
                grid_size = min(4, int(np.ceil(np.sqrt(len(montage_frames)))))
                montage_height = grid_size * 180
                montage_width = grid_size * 320
                montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
                
                # Place frames in the grid
                for i, frame in enumerate(montage_frames):
                    row = i // grid_size
                    col = i % grid_size
                    y_start = row * 180
                    x_start = col * 320
                    
                    # Add the frame to the montage
                    montage[y_start:y_start+180, x_start:x_start+320] = frame
                    
                    # Add frame number
                    cv2.putText(montage, f"Frame {montage_indices[i]}", 
                              (x_start + 5, y_start + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save the montage
                base_name = os.path.basename(video_file)
                file_name, _ = os.path.splitext(base_name)
                montage_path = os.path.join(output_dir, f"{file_name}_montage.png")
                cv2.imwrite(montage_path, montage)
                print(f"Frame montage saved to: {montage_path}")
            
            # 2. Create a histogram visualization
            if histogram_distances:
                plt.figure(figsize=(12, 10))
                
                # Plot histogram correlation distances
                h_distances = [h for h, _ in histogram_distances]
                s_distances = [s for _, s in histogram_distances]
                frames = range(1, len(histogram_distances) + 1)
                
                plt.subplot(2, 2, 1)
                plt.plot(frames, h_distances, 'b-', label='Hue Correlation')
                plt.plot(frames, s_distances, 'g-', label='Saturation Correlation')
                plt.axhline(y=0.995, color='r', linestyle='--', alpha=0.5)
                plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
                plt.title('Histogram Correlations Between Consecutive Frames')
                plt.xlabel('Frame Pair')
                plt.ylabel('Correlation (higher = more similar)')
                plt.legend()
                plt.grid(True)
                
                # Plot anomaly counts
                plt.subplot(2, 2, 2)
                anomaly_types = ['Histogram', 'Corner', 'Edge']
                anomaly_counts = [histogram_anomalies, corner_anomalies, edge_anomalies]
                anomaly_percentages = [
                    hist_score * 100,
                    corner_score * 100,
                    edge_score * 100
                ]
                
                bars = plt.bar(anomaly_types, anomaly_percentages)
                plt.title('Anomaly Percentages by Type')
                plt.ylabel('Percentage of Frames (%)')
                plt.grid(True, axis='y')
                
                # Color bars based on percentage
                for bar, percentage in zip(bars, anomaly_percentages):
                    if percentage > 10:
                        bar.set_color('red')
                    elif percentage > 5:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                # Add text labels
                for i, percentage in enumerate(anomaly_percentages):
                    plt.text(i, percentage + 1, f"{percentage:.1f}%", 
                           ha='center', va='bottom')
                
                # Plot watermark likelihood
                plt.subplot(2, 1, 2)
                plt.pie([watermark_likelihood, 1 - watermark_likelihood], 
                      labels=['Watermark Likelihood', 'No Watermark'],
                      colors=['red' if watermark_likelihood > 0.1 else 'orange', 'green'],
                      autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                
                detected_text = "WATERMARK DETECTED" if watermark_likelihood > 0.1 else "NO WATERMARK DETECTED"
                plt.title(f'Watermark Analysis: {detected_text} ({watermark_likelihood*100:.1f}%)')
                
                # Add summary
                summary = f"Video: {base_name}\n"
                summary += f"Dimensions: {width}x{height}, {fps:.1f} fps, {duration:.1f} seconds\n"
                summary += f"Frames Analyzed: {frames_analyzed} (sampled every {sample_interval} frames)\n"
                summary += f"Watermark Likelihood: {watermark_likelihood*100:.1f}%\n"
                summary += f"Potential Watermark Types: " + ", ".join(
                    anomaly_types[i] for i, p in enumerate(anomaly_percentages) if p > 5)
                
                plt.figtext(0.5, 0.01, summary, wrap=True, horizontalalignment='center', fontsize=10, 
                          bbox=dict(facecolor="lightgray", alpha=0.5, pad=5))
                
                # Save the figure
                analysis_path = os.path.join(output_dir, f"{file_name}_analysis.png")
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.savefig(analysis_path, dpi=300)
                plt.close()
                print(f"Analysis visualization saved to: {analysis_path}")
            
            # 3. Save results as JSON
            json_path = os.path.join(output_dir, f"{file_name}_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Analysis results saved to: {json_path}")
        
        # Clean up
        video.release()
        
        return results
        
    except Exception as e:
        print(f"Error during video analysis: {e}")
        return None


def main():
    """Command line interface for video watermark extraction and analysis."""
    parser = argparse.ArgumentParser(description='Video Watermark Extraction Tool')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output-dir', default='./output', 
                      help='Directory to save analysis results')
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='Generate visualizations (default: True)')
    parser.add_argument('--method', choices=['standard', 'frame'], default='standard',
                      help='Extraction method (standard=use VideoWatermarker, frame=detailed analysis)')
    parser.add_argument('--sample-interval', type=int, default=30,
                      help='Number of frames to skip between samples')
    parser.add_argument('--max-frames', type=int, default=100,
                      help='Maximum number of frames to analyze')
    
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
        # Use the VideoWatermarker for extraction
        watermarker = VideoWatermarker()
        
        output_file = None
        if args.visualize:
            output_file = os.path.join(args.output_dir, f"{file_name}_analysis.png")
        
        results = watermarker.extract_watermark(args.input, output_file)
        
        if results:
            print("\nExtraction Results:")
            print(f"Watermark Detected: {'Yes' if results.get('watermark_detected', False) else 'No'}")
            
            if results.get('watermark_detected', False):
                print(f"Confidence: {results.get('confidence', 0) * 100:.1f}%")
                print(f"Likely AI Generated: {'Yes' if results.get('likely_ai_generated', False) else 'No'}")
                
                method_scores = results.get("method_scores", {})
                print("\nDetection by Method:")
                for method, score in method_scores.items():
                    print(f"  - {method}: {score * 100:.1f}%")
            
            # Save results to JSON
            json_path = os.path.join(args.output_dir, f"{file_name}_watermark_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {json_path}")
            if output_file and os.path.exists(output_file):
                print(f"Analysis image saved to: {output_file}")
    
    else:  # frame analysis
        # Perform detailed frame analysis
        results = analyze_video_frames(
            args.input, 
            args.output_dir,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames
        )
        
        if results:
            analysis = results['analysis']
            print("\nFrame Analysis Results:")
            print(f"Dimensions: {results['width']}x{results['height']}")
            print(f"Duration: {results['duration']:.2f} seconds, {results['fps']:.1f} fps")
            print(f"Frames Analyzed: {results['frames_analyzed']} (sampled every {args.sample_interval} frames)")
            print(f"Histogram Anomalies: {analysis['histogram_anomalies']} ({analysis['histogram_anomaly_percentage']*100:.1f}%)")
            print(f"Corner Anomalies: {analysis['corner_anomalies']} ({analysis['corner_anomaly_percentage']*100:.1f}%)")
            print(f"Edge Anomalies: {analysis['edge_anomalies']} ({analysis['edge_anomaly_percentage']*100:.1f}%)")
            print(f"Watermark Likelihood: {analysis['watermark_likelihood']*100:.1f}%")
            print(f"Watermark Detected: {'Yes' if analysis['watermark_detected'] else 'No'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

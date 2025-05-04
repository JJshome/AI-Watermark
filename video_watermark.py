"""
Video Watermarking Implementation

This module implements video watermarking techniques based on the patent by Ucaretron Inc.
It can embed imperceptible watermarks in video files to indicate AI-generated content
or confidence levels of AI inference.

The watermarking is done using several methods:
1. Color histogram modification
2. Frame-by-frame watermarking
3. Inter-frame encoding
"""

import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
import tempfile
import subprocess
import hashlib
import random

class VideoWatermarker:
    """Video watermarking class for embedding and extracting watermarks."""
    
    def __init__(self, 
                 method='histogram',  # Watermarking method
                 strength=5,  # Strength of the watermark (0-10)
                 pattern_size=8,  # Size of the pattern for encoding
                 ):
        """
        Initialize the video watermarker.
        
        Args:
            method: Watermarking method ('histogram', 'frame', 'interframe', 'pixel')
            strength: Strength of the watermark (0-10)
            pattern_size: Size of the pattern for encoding
        """
        self.method = method
        self.strength = min(10, max(1, strength))  # Clamp between 1-10
        self.pattern_size = pattern_size
        
        # Watermark metadata content
        self.metadata = {
            "source": "AI-Watermark",
            "patent": "Ucaretron Inc.",
            "timestamp": "",
            "confidence": 1.0,  # Default confidence (1.0 = highest confidence)
            "ai_generated": True,
            "license": "For research purposes only"
        }
        
        # Initialize watermark pattern
        self._init_watermark_pattern()
    
    def _init_watermark_pattern(self):
        """Initialize watermark pattern based on metadata."""
        # Generate a deterministic pattern from metadata
        self.metadata["timestamp"] = datetime.now().isoformat()
        pattern_seed = hashlib.md5(json.dumps(self.metadata).encode()).hexdigest()
        
        # Use the pattern seed to initialize a random number generator
        random.seed(pattern_seed)
        
        # Create binary pattern for encoding
        self.watermark_pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=np.uint8)
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                self.watermark_pattern[i, j] = random.randint(0, 1)
    
    def _apply_histogram_watermark(self, frame):
        """
        Apply watermark by slightly modifying the color histogram.
        
        Args:
            frame: Input video frame
            
        Returns:
            numpy.ndarray: Watermarked frame
        """
        # Convert to HSV for better color manipulation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Split the channels
        h, s, v = cv2.split(hsv_frame)
        
        # Apply slight modifications to the hue channel based on pattern
        h_height, h_width = h.shape
        pattern_repeated = np.tile(
            self.watermark_pattern, 
            (h_height // self.pattern_size + 1, h_width // self.pattern_size + 1)
        )
        pattern_repeated = pattern_repeated[:h_height, :h_width]
        
        # Scale the effect based on strength
        effect_scale = self.strength * 0.5  # Scale strength to reasonable effect
        
        # Apply a subtle frequency domain watermark to avoid visible artifacts
        h_float = h.astype(np.float32)
        h_dct = cv2.dct(h_float)
        
        # Create a watermark pattern in frequency domain
        mask = np.zeros_like(h_dct)
        mid_freq_rows = h_height // 4
        mid_freq_cols = h_width // 4
        
        # Apply watermark to mid-frequency components (less visible)
        mask[mid_freq_rows:mid_freq_rows*2, mid_freq_cols:mid_freq_cols*2] = \
            pattern_repeated[mid_freq_rows:mid_freq_rows*2, mid_freq_cols:mid_freq_cols*2] * effect_scale
        
        # Apply the mask
        h_dct = h_dct + mask
        
        # Convert back to spatial domain
        h_watermarked = cv2.idct(h_dct)
        h_watermarked = np.clip(h_watermarked, 0, 255).astype(np.uint8)
        
        # Merge the channels back
        hsv_watermarked = cv2.merge([h_watermarked, s, v])
        
        # Convert back to BGR
        watermarked_frame = cv2.cvtColor(hsv_watermarked, cv2.COLOR_HSV2BGR)
        
        return watermarked_frame
    
    def _apply_frame_watermark(self, frame):
        """
        Apply watermark by embedding information in each frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            numpy.ndarray: Watermarked frame
        """
        # Create a copy of the frame
        watermarked_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate position for watermark (bottom right corner, less noticeable)
        pos_x = width - self.pattern_size - 10
        pos_y = height - self.pattern_size - 10
        
        # Scale watermark effect based on strength
        effect = max(1, min(10, self.strength)) / 10.0
        
        # Embed watermark pattern
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if self.watermark_pattern[i, j] == 1:
                    # Get the pixel at the position
                    pixel = watermarked_frame[pos_y + i, pos_x + j].astype(float)
                    
                    # Apply a subtle change to the pixel (less noticeable in darker regions)
                    # For 1 bit, slightly increase blue channel relative to green
                    b, g, r = pixel
                    
                    # Apply change based on strength
                    delta = 2 * effect
                    
                    # These changes are designed to be almost invisible
                    b = min(255, b + delta)
                    g = max(0, g - delta)
                    
                    watermarked_frame[pos_y + i, pos_x + j] = [b, g, r]
        
        return watermarked_frame
    
    def _apply_pixel_watermark(self, frame):
        """
        Apply watermark by modifying specific pixels based on confidence.
        
        Args:
            frame: Input video frame
            
        Returns:
            numpy.ndarray: Watermarked frame
        """
        # Create a copy of the frame
        watermarked_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Use confidence value to determine watermarking intensity
        confidence = self.metadata["confidence"]
        
        # Lower confidence means more visible marking
        visibility = (1.0 - confidence) * 0.3 * self.strength
        
        # Apply a grid pattern across the frame
        grid_size = 32  # Size of grid cells
        
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Use watermark pattern to determine marking
                pattern_x = (x // grid_size) % self.pattern_size
                pattern_y = (y // grid_size) % self.pattern_size
                
                if self.watermark_pattern[pattern_y, pattern_x] == 1:
                    # Mark this grid cell with subtle pixel modifications
                    cell_height = min(grid_size, height - y)
                    cell_width = min(grid_size, width - x)
                    
                    # Apply a very subtle filter to this cell
                    cell = watermarked_frame[y:y+cell_height, x:x+cell_width]
                    
                    # Slightly shift color balance based on confidence
                    # Less confident = more noticeable shift
                    b, g, r = cv2.split(cell)
                    
                    # Apply a subtle warming/cooling effect based on confidence
                    r_factor = 1.0 + visibility * 0.2
                    b_factor = 1.0 - visibility * 0.1
                    
                    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
                    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
                    
                    marked_cell = cv2.merge([b, g, r])
                    watermarked_frame[y:y+cell_height, x:x+cell_width] = marked_cell
        
        return watermarked_frame
    
    def _apply_interframe_watermark(self, frames):
        """
        Apply watermark that spans across frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            list: Watermarked frames
        """
        # Need multiple frames for this method
        if len(frames) < 2:
            print("Warning: Interframe watermarking requires at least 2 frames. Falling back to frame method.")
            return [self._apply_frame_watermark(frame) for frame in frames]
        
        # Encode information across frame sequence
        watermarked_frames = []
        
        # Convert metadata to a bit sequence (simplified)
        metadata_json = json.dumps(self.metadata)
        metadata_bytes = metadata_json.encode('utf-8')
        
        # Create a checksum for verification
        checksum = hashlib.md5(metadata_bytes).digest()
        
        # Combine metadata and checksum
        data_to_encode = metadata_bytes + checksum
        
        # Calculate how many bits we can encode per frame
        bits_per_frame = self.pattern_size * self.pattern_size
        
        # Prepare binary data
        binary_data = ''.join(format(byte, '08b') for byte in data_to_encode)
        
        # Encode across frames
        for i, frame in enumerate(frames):
            # Copy the frame
            watermarked_frame = frame.copy()
            
            # Calculate which portion of data to encode in this frame
            start_bit = (i * bits_per_frame) % len(binary_data)
            end_bit = min(start_bit + bits_per_frame, len(binary_data))
            
            if start_bit < len(binary_data):
                # Get the bit segment for this frame
                frame_bits = binary_data[start_bit:end_bit].ljust(bits_per_frame, '0')
                
                # Create a pattern for this frame
                frame_pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=np.uint8)
                for j in range(self.pattern_size):
                    for k in range(self.pattern_size):
                        bit_idx = j * self.pattern_size + k
                        if bit_idx < len(frame_bits):
                            frame_pattern[j, k] = int(frame_bits[bit_idx])
                
                # Store the original pattern
                original_pattern = self.watermark_pattern.copy()
                
                # Use this frame's pattern
                self.watermark_pattern = frame_pattern
                
                # Apply frame watermark using this pattern
                watermarked_frame = self._apply_frame_watermark(watermarked_frame)
                
                # Restore original pattern
                self.watermark_pattern = original_pattern
            
            watermarked_frames.append(watermarked_frame)
            
            # Add a marker frame every 30 frames as a synchronization point
            if i > 0 and i % 30 == 0 and i < len(frames) - 1:
                # Create a sync frame that's nearly identical to the last frame
                sync_frame = watermarked_frame.copy()
                
                # Add a very subtle sync pattern
                height, width = sync_frame.shape[:2]
                
                # Add sync markers in corners
                marker_size = 4
                sync_pattern = np.ones((marker_size, marker_size), dtype=np.uint8)
                
                # Top left
                sync_frame[:marker_size, :marker_size, 0] = \
                    (sync_frame[:marker_size, :marker_size, 0] + 1) % 256
                
                # Bottom right
                sync_frame[-marker_size:, -marker_size:, 2] = \
                    (sync_frame[-marker_size:, -marker_size:, 2] + 1) % 256
                
                watermarked_frames.append(sync_frame)
        
        return watermarked_frames
    
    def add_watermark(self, input_file, output_file, confidence=1.0, ai_generated=True):
        """
        Add a watermark to a video file.
        
        Args:
            input_file: Path to the input video file
            output_file: Path to save the watermarked video
            confidence: AI confidence level (0.0-1.0)
            ai_generated: Flag indicating if the video is AI-generated
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update metadata
            self.metadata["confidence"] = confidence
            self.metadata["ai_generated"] = ai_generated
            self.metadata["timestamp"] = datetime.now().isoformat()
            
            # Open the input video
            video = cv2.VideoCapture(input_file)
            
            if not video.isOpened():
                print(f"Error: Could not open video file {input_file}")
                return False
            
            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Processing video: {width}x{height}, {fps} fps, {frame_count} frames")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # For interframe method, we need to collect frames
            if self.method == 'interframe':
                # Collect frames in batches to manage memory
                batch_size = 60  # Process 60 frames at a time
                frames_processed = 0
                
                while frames_processed < frame_count:
                    # Collect a batch of frames
                    frames_batch = []
                    for _ in range(min(batch_size, frame_count - frames_processed)):
                        ret, frame = video.read()
                        if not ret:
                            break
                        frames_batch.append(frame)
                    
                    if not frames_batch:
                        break
                    
                    # Process this batch
                    watermarked_frames = self._apply_interframe_watermark(frames_batch)
                    
                    # Write watermarked frames
                    for wframe in watermarked_frames:
                        out.write(wframe)
                    
                    frames_processed += len(frames_batch)
                    print(f"Processed {frames_processed}/{frame_count} frames")
            else:
                # Process frame by frame
                frame_count = 0
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    # Apply watermark based on selected method
                    if self.method == 'histogram':
                        watermarked_frame = self._apply_histogram_watermark(frame)
                    elif self.method == 'pixel':
                        watermarked_frame = self._apply_pixel_watermark(frame)
                    else:  # Default to frame method
                        watermarked_frame = self._apply_frame_watermark(frame)
                    
                    # Write the watermarked frame
                    out.write(watermarked_frame)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames")
            
            # Release resources
            video.release()
            out.release()
            
            print(f"Successfully watermarked video: {output_file}")
            print(f"Metadata: {json.dumps(self.metadata, indent=2)}")
            
            return True
            
        except Exception as e:
            print(f"Error watermarking video: {e}")
            return False
    
    def extract_watermark(self, input_file, output_file=None):
        """
        Extract watermark information from a video file.
        
        Args:
            input_file: Path to the watermarked video file
            output_file: Optional path to save watermark visualization
            
        Returns:
            dict: Extracted metadata or None if extraction failed
        """
        try:
            # Open the input video
            video = cv2.VideoCapture(input_file)
            
            if not video.isOpened():
                print(f"Error: Could not open video file {input_file}")
                return None
            
            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # For analysis, we'll sample frames
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_count = min(100, frame_count)  # Sample up to 100 frames
            sample_interval = max(1, frame_count // sample_count)
            
            # Initialize results
            watermark_detected = False
            watermark_confidence = 0.0
            histogram_anomalies = 0
            pixel_anomalies = 0
            frame_anomalies = 0
            
            # Extract for visualization
            if output_file:
                output_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Sample frames for analysis
            for i in range(0, frame_count, sample_interval):
                # Set the frame position
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                
                if not ret:
                    break
                
                # Analyze the frame for watermark evidence
                
                # 1. Histogram analysis
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, _, _ = cv2.split(hsv_frame)
                
                # Analyze the hue histogram for anomalies
                h_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
                
                # Check for unusual patterns in histogram
                hist_diff = 0
                for j in range(1, len(h_hist)):
                    hist_diff += abs(int(h_hist[j]) - int(h_hist[j-1]))
                
                # High frequency changes can indicate watermarking
                if hist_diff > width * height * 0.1:
                    histogram_anomalies += 1
                
                # 2. Analyze for frame watermarks
                # Check the bottom right corner for pattern
                pos_x = width - self.pattern_size - 10
                pos_y = height - self.pattern_size - 10
                
                if pos_x > 0 and pos_y > 0:
                    corner_patch = frame[pos_y:pos_y+self.pattern_size, 
                                        pos_x:pos_x+self.pattern_size]
                    
                    # Analyze blue/green channel ratio for our specific pattern
                    if corner_patch.size > 0:
                        b, g, _ = cv2.split(corner_patch)
                        bg_ratio = np.mean(b) / (np.mean(g) + 0.001)
                        
                        # Our watermark increases blue relative to green
                        if bg_ratio > 1.05:  # Threshold
                            frame_anomalies += 1
                
                # 3. Check for pixel watermarking
                # Look for grid patterns with color shifts
                grid_size = 32
                grid_anomalies = 0
                
                for y in range(0, height, grid_size):
                    for x in range(0, width, grid_size):
                        cell_height = min(grid_size, height - y)
                        cell_width = min(grid_size, width - x)
                        
                        if cell_height > 0 and cell_width > 0:
                            cell = frame[y:y+cell_height, x:x+cell_width]
                            b, g, r = cv2.split(cell)
                            
                            # Check for unusual color balance
                            rb_ratio = np.mean(r) / (np.mean(b) + 0.001)
                            
                            # Our pixel watermark shifts R/B ratio
                            if rb_ratio > 1.1:
                                grid_anomalies += 1
                
                pixel_anomalies += grid_anomalies / ((width // grid_size) * (height // grid_size) + 0.001)
                
                # Accumulate for visualization
                if output_file:
                    # Highlight detected watermarks
                    highlight = frame.copy()
                    
                    # Highlight histogram anomalies
                    if hist_diff > width * height * 0.1:
                        highlight[:, :, 0] = np.minimum(highlight[:, :, 0] + 30, 255)
                    
                    # Highlight frame watermark
                    if frame_anomalies > 0:
                        cv2.rectangle(highlight, (pos_x, pos_y), 
                                     (pos_x+self.pattern_size, pos_y+self.pattern_size), 
                                     (0, 0, 255), 2)
                    
                    # Highlight pixel anomalies
                    if grid_anomalies > 0:
                        for y in range(0, height, grid_size):
                            for x in range(0, width, grid_size):
                                cell_height = min(grid_size, height - y)
                                cell_width = min(grid_size, width - x)
                                
                                cell = frame[y:y+cell_height, x:x+cell_width]
                                b, g, r = cv2.split(cell)
                                
                                rb_ratio = np.mean(r) / (np.mean(b) + 0.001)
                                
                                if rb_ratio > 1.1:
                                    cv2.rectangle(highlight, (x, y), 
                                                (x+cell_width, y+cell_height), 
                                                (0, 255, 0), 1)
                    
                    # Accumulate to output image
                    alpha = 1.0 / sample_count
                    output_img = cv2.addWeighted(output_img, 1.0 - alpha, highlight, alpha, 0)
            
            # Calculate overall watermark confidence
            histogram_score = histogram_anomalies / sample_count
            frame_score = frame_anomalies / sample_count
            pixel_score = pixel_anomalies / sample_count
            
            # Overall confidence
            watermark_confidence = max(histogram_score, frame_score, pixel_score)
            watermark_detected = watermark_confidence > 0.1  # Threshold
            
            # Save visualization if requested
            if output_file and watermark_detected:
                cv2.imwrite(output_file, output_img)
                print(f"Watermark visualization saved to: {output_file}")
            
            # Results
            results = {
                "watermark_detected": watermark_detected,
                "confidence": float(watermark_confidence),
                "method_scores": {
                    "histogram": float(histogram_score),
                    "frame": float(frame_score),
                    "pixel": float(pixel_score)
                },
                "likely_ai_generated": watermark_detected and watermark_confidence > 0.3
            }
            
            # Release resources
            video.release()
            
            return results
            
        except Exception as e:
            print(f"Error extracting watermark: {e}")
            return None


def main():
    """Command line interface for video watermarking."""
    parser = argparse.ArgumentParser(description='Video Watermarking Tool')
    parser.add_argument('--mode', choices=['add', 'extract'], required=True,
                      help='Operation mode: add or extract watermark')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--method', choices=['histogram', 'frame', 'interframe', 'pixel'],
                      default='histogram', help='Watermarking method')
    parser.add_argument('--confidence', type=float, default=1.0,
                      help='AI confidence level (0.0-1.0)')
    parser.add_argument('--ai-generated', type=bool, default=True,
                      help='Flag indicating if the video is AI-generated')
    parser.add_argument('--strength', type=int, default=5,
                      help='Watermark strength (1-10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create watermarker
    watermarker = VideoWatermarker(
        method=args.method,
        strength=args.strength
    )
    
    if args.mode == 'add':
        watermarker.add_watermark(
            args.input,
            args.output,
            confidence=args.confidence,
            ai_generated=args.ai_generated
        )
    else:  # extract
        result = watermarker.extract_watermark(args.input, args.output)
        if result:
            print("Extraction results:")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

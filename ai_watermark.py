#!/usr/bin/env python3
"""
AI Watermark - Main Command-line Interface

This script provides a unified command-line interface for AI watermarking
tools implemented in this repository. It handles both audio and video files
for watermarking and watermark detection.

Based on the patent by Ucaretron Inc. for methods to indicate AI-generated content.
"""

import os
import sys
import argparse
from audio_watermark import AudioWatermarker
from video_watermark import VideoWatermarker
from watermark_extractor import WatermarkExtractor


def main():
    """Main entry point for the AI Watermark CLI."""
    parser = argparse.ArgumentParser(
        description='AI Watermark - Tools for marking and detecting AI-generated content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Add watermark to an audio file
  python ai_watermark.py add --input original.wav --output watermarked.wav --confidence 0.7
  
  # Add watermark to a video file using histogram method
  python ai_watermark.py add --input original.mp4 --output watermarked.mp4 --method histogram
  
  # Extract and analyze a watermark
  python ai_watermark.py extract --input watermarked.wav --output-dir ./analysis
        '''
    )
    
    # Main operation mode
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    subparsers.required = True
    
    # Add watermark mode
    add_parser = subparsers.add_parser('add', help='Add watermark to media file')
    add_parser.add_argument('--input', required=True, help='Input media file')
    add_parser.add_argument('--output', required=True, help='Output watermarked file')
    add_parser.add_argument('--confidence', type=float, default=1.0,
                          help='AI confidence level (0.0-1.0)')
    add_parser.add_argument('--ai-generated', type=bool, default=True,
                          help='Flag indicating if the content is AI-generated')
    
    # Method-specific arguments
    add_parser.add_argument('--method', choices=['histogram', 'frame', 'interframe', 'pixel'],
                          default='histogram', help='Video watermarking method')
    add_parser.add_argument('--strength', type=int, default=5,
                          help='Watermark strength (1-10)')
    add_parser.add_argument('--frequency', type=int, default=19000,
                          help='Carrier frequency for audio watermark (Hz)')
    add_parser.add_argument('--amplitude', type=float, default=0.05,
                          help='Amplitude of audio watermark (0.0-1.0)')
    
    # Extract watermark mode
    extract_parser = subparsers.add_parser('extract', help='Extract watermark from media file')
    extract_parser.add_argument('--input', required=True, help='Input media file')
    extract_parser.add_argument('--output-dir', default='./output', 
                              help='Directory to save analysis results')
    extract_parser.add_argument('--no-visualize', action='store_true', 
                              help='Disable visualization generation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Determine file type based on extension
    file_ext = os.path.splitext(args.input.lower())[1]
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    if file_ext in audio_extensions:
        file_type = 'audio'
    elif file_ext in video_extensions:
        file_type = 'video'
    else:
        print(f"Error: Unsupported file type: {file_ext}")
        return 1
    
    # Execute requested operation
    if args.mode == 'add':
        if file_type == 'audio':
            # Audio watermarking
            watermarker = AudioWatermarker(
                carrier_frequency=args.frequency,
                amplitude=args.amplitude
            )
            result = watermarker.add_watermark(
                args.input,
                args.output,
                confidence=args.confidence,
                ai_generated=args.ai_generated
            )
        else:  # video
            # Video watermarking
            watermarker = VideoWatermarker(
                method=args.method,
                strength=args.strength
            )
            result = watermarker.add_watermark(
                args.input,
                args.output,
                confidence=args.confidence,
                ai_generated=args.ai_generated
            )
        
        if not result:
            print("Error: Watermarking operation failed")
            return 1
        
    elif args.mode == 'extract':
        # Create the output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # Extract watermark
        extractor = WatermarkExtractor()
        results = extractor.extract_watermark(
            args.input, 
            args.output_dir, 
            not args.no_visualize
        )
        
        if not results:
            print("Error: Watermark extraction failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

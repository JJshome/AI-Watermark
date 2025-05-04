"""
Audio Watermarking Implementation

This module implements audio watermarking techniques based on the patent by Ucaretron Inc.
It can embed imperceptible watermarks in audio files to indicate AI-generated content
or confidence levels of AI inference.

The watermarking is done by embedding signals in high frequency ranges that are
typically inaudible to humans but can be detected with analysis tools.
"""

import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import os
import json
from datetime import datetime

class AudioWatermarker:
    """Audio watermarking class for embedding and extracting watermarks."""
    
    def __init__(self, 
                 carrier_frequency=19000,  # Hz, near the upper limit of human hearing
                 secondary_frequency=19500,  # Hz, used for encoding binary data
                 amplitude=0.05,  # Relative amplitude of the watermark
                 message_bits=64,  # Number of bits for embedded message
                 ):
        """
        Initialize the audio watermarker.
        
        Args:
            carrier_frequency: Main frequency for the watermark (Hz)
            secondary_frequency: Secondary frequency for encoding bits (Hz)
            amplitude: Relative amplitude of the watermark (0.0-1.0)
            message_bits: Number of bits to embed in the watermark
        """
        self.carrier_frequency = carrier_frequency
        self.secondary_frequency = secondary_frequency
        self.amplitude = amplitude
        self.message_bits = message_bits
        
        # Watermark metadata content
        self.metadata = {
            "source": "AI-Watermark",
            "patent": "Ucaretron Inc.",
            "timestamp": "",
            "confidence": 1.0,  # Default confidence (1.0 = highest confidence)
            "ai_generated": True,
            "license": "For research purposes only"
        }
    
    def _create_watermark_signal(self, duration, sample_rate, message=None):
        """
        Create the watermark signal to be embedded.
        
        Args:
            duration: Duration of the watermark signal in seconds
            sample_rate: Sample rate of the audio
            message: Optional binary message to embed (if None, metadata is used)
            
        Returns:
            numpy.ndarray: The watermark signal
        """
        # Update timestamp
        self.metadata["timestamp"] = datetime.now().isoformat()
        
        # Generate the carrier signal
        t = np.arange(0, duration, 1/sample_rate)
        carrier = np.sin(2 * np.pi * self.carrier_frequency * t)
        
        # Create the watermark signal
        watermark = np.zeros_like(t)
        
        # If no specific message provided, encode metadata
        if message is None:
            # Convert metadata to a bit sequence
            metadata_json = json.dumps(self.metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            message = ''.join(format(byte, '08b') for byte in metadata_bytes)
            
            # Truncate or pad message to fit message_bits
            if len(message) > self.message_bits:
                message = message[:self.message_bits]
            else:
                message = message.ljust(self.message_bits, '0')
        
        # Embed message bits
        bit_duration = int(duration * sample_rate / len(message))
        for i, bit in enumerate(message):
            start_idx = i * bit_duration
            end_idx = min((i + 1) * bit_duration, len(t))
            
            if bit == '1':
                # For bit 1, use the secondary frequency
                watermark[start_idx:end_idx] = np.sin(2 * np.pi * self.secondary_frequency * t[start_idx:end_idx])
            else:
                # For bit 0, use the carrier frequency
                watermark[start_idx:end_idx] = carrier[start_idx:end_idx]
        
        # Apply amplitude scaling
        watermark *= self.amplitude
        
        return watermark
    
    def add_watermark(self, input_file, output_file, confidence=1.0, ai_generated=True):
        """
        Add a watermark to an audio file.
        
        Args:
            input_file: Path to the input audio file
            output_file: Path to save the watermarked audio
            confidence: AI confidence level (0.0-1.0)
            ai_generated: Flag indicating if the audio is AI-generated
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update metadata
            self.metadata["confidence"] = confidence
            self.metadata["ai_generated"] = ai_generated
            
            # Read the input audio file
            sample_rate, audio_data = wavfile.read(input_file)
            
            # Convert to float for processing if needed
            if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
                # Save original data type for later conversion
                original_dtype = audio_data.dtype
                
                # Scale based on data type
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32767.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float32) / 2147483647.0
                elif audio_data.dtype == np.uint8:
                    audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
                else:
                    print(f"Warning: Unsupported data type {audio_data.dtype}, using int16 scaling")
                    audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.copy()
                original_dtype = audio_data.dtype
            
            # Determine if mono or stereo
            is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            
            # Create watermark signal
            watermark = self._create_watermark_signal(duration, sample_rate)
            
            # Apply watermark
            if is_stereo:
                # For stereo, add watermark to both channels
                watermarked_audio = audio_float.copy()
                for channel in range(audio_float.shape[1]):
                    watermarked_audio[:, channel] = audio_float[:, channel] + watermark
            else:
                # For mono
                watermarked_audio = audio_float + watermark
            
            # Clip to prevent overflow
            watermarked_audio = np.clip(watermarked_audio, -1.0, 1.0)
            
            # Convert back to original data type
            if original_dtype == np.int16:
                watermarked_audio = (watermarked_audio * 32767.0).astype(np.int16)
            elif original_dtype == np.int32:
                watermarked_audio = (watermarked_audio * 2147483647.0).astype(np.int32)
            elif original_dtype == np.uint8:
                watermarked_audio = ((watermarked_audio * 128.0) + 128.0).astype(np.uint8)
            elif original_dtype == np.float32:
                watermarked_audio = watermarked_audio.astype(np.float32)
            elif original_dtype == np.float64:
                watermarked_audio = watermarked_audio.astype(np.float64)
                
            # Write the watermarked audio to the output file
            wavfile.write(output_file, sample_rate, watermarked_audio)
            
            print(f"Successfully watermarked audio: {output_file}")
            print(f"Metadata: {json.dumps(self.metadata, indent=2)}")
            
            return True
            
        except Exception as e:
            print(f"Error watermarking audio: {e}")
            return False
    
    def extract_watermark(self, input_file, output_file=None):
        """
        Extract watermark from an audio file.
        
        Args:
            input_file: Path to the watermarked audio file
            output_file: Optional path to save the extracted watermark signal
            
        Returns:
            dict: Extracted metadata or None if extraction failed
        """
        try:
            # Read the input audio file
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
                    print(f"Warning: Unsupported data type {audio_data.dtype}, using int16 scaling")
                    audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.copy()
            
            # Extract first channel if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_channel = audio_float[:, 0]
            else:
                audio_channel = audio_float
            
            # Apply bandpass filter to isolate the watermark frequencies
            # This is a simplified approach - real implementation would use
            # proper signal processing techniques for better extraction
            
            # Perform FFT
            fft_data = np.fft.rfft(audio_channel)
            
            # Calculate frequency bins
            freqs = np.fft.rfftfreq(len(audio_channel), 1/sample_rate)
            
            # Create a band mask around the carrier and secondary frequencies
            carrier_mask = np.abs(freqs - self.carrier_frequency) < 100
            secondary_mask = np.abs(freqs - self.secondary_frequency) < 100
            
            # Extract carrier and secondary energy
            carrier_energy = np.abs(fft_data[carrier_mask]).sum()
            secondary_energy = np.abs(fft_data[secondary_mask]).sum()
            
            # Detect if watermark is present
            watermark_present = carrier_energy > 0 or secondary_energy > 0
            
            if watermark_present:
                print("Watermark detected in audio file")
                
                # TODO: Implement more sophisticated decoding of the embedded message
                # This is a simplified placeholder implementation
                
                # Create a basic watermark signature
                extracted_metadata = {
                    "watermark_present": True,
                    "carrier_energy": float(carrier_energy),
                    "secondary_energy": float(secondary_energy),
                    "carrier_frequency": self.carrier_frequency,
                    "secondary_frequency": self.secondary_frequency,
                    "likely_ai_generated": True if carrier_energy > 0 else False
                }
                
                # If output file is specified, save the filtered signal
                if output_file:
                    # Create a filtered version that emphasizes the watermark
                    filtered_fft = np.zeros_like(fft_data)
                    filtered_fft[carrier_mask | secondary_mask] = fft_data[carrier_mask | secondary_mask]
                    filtered_signal = np.fft.irfft(filtered_fft)
                    
                    # Scale the filtered signal
                    filtered_signal = filtered_signal / np.max(np.abs(filtered_signal)) if np.max(np.abs(filtered_signal)) > 0 else filtered_signal
                    filtered_signal = (filtered_signal * 32767).astype(np.int16)
                    
                    # Save the filtered signal
                    wavfile.write(output_file, sample_rate, filtered_signal)
                    print(f"Saved filtered watermark to: {output_file}")
                
                return extracted_metadata
            else:
                print("No watermark detected in audio file")
                return {"watermark_present": False}
                
        except Exception as e:
            print(f"Error extracting watermark: {e}")
            return None


def main():
    """Command line interface for audio watermarking."""
    parser = argparse.ArgumentParser(description='Audio Watermarking Tool')
    parser.add_argument('--mode', choices=['add', 'extract'], required=True,
                      help='Operation mode: add or extract watermark')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--confidence', type=float, default=1.0,
                      help='AI confidence level (0.0-1.0)')
    parser.add_argument('--ai-generated', type=bool, default=True,
                      help='Flag indicating if the audio is AI-generated')
    parser.add_argument('--frequency', type=int, default=19000,
                      help='Carrier frequency for watermark (Hz)')
    parser.add_argument('--amplitude', type=float, default=0.05,
                      help='Amplitude of watermark (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create watermarker
    watermarker = AudioWatermarker(
        carrier_frequency=args.frequency,
        amplitude=args.amplitude
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

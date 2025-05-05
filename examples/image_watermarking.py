"""
AI Watermark - Image Watermarking Example
Based on Ucaretron Inc. patent technology

This example demonstrates how to apply visual watermarks to AI-generated images
based on confidence scores and generation masks.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageWatermarker:
    """
    Class for applying confidence-based watermarks to AI-generated images
    """
    
    def __init__(self, options=None):
        """
        Initialize the image watermarker with configurable options
        
        Args:
            options (dict): Configuration options
        """
        self.options = {
            'high_confidence_color': (0, 0, 255),  # Blue for high confidence
            'medium_confidence_color': (0, 0, 0),  # Black for medium confidence
            'low_confidence_color': (255, 0, 0),   # Red for low confidence
            'high_confidence_threshold': 0.8,
            'low_confidence_threshold': 0.5,
            'watermark_intensity': 0.3,            # Transparency level for the watermark
        }
        
        if options:
            self.options.update(options)
    
    def apply_watermark(self, image, confidence_map):
        """
        Apply confidence-based watermarks to an image
        
        Args:
            image (numpy.ndarray): The original image (HxWx3, RGB)
            confidence_map (numpy.ndarray): Confidence values (0-1) for each pixel (HxW)
            
        Returns:
            numpy.ndarray: Watermarked image
        """
        if image.shape[:2] != confidence_map.shape:
            raise ValueError("Image and confidence map dimensions must match")
        
        # Create an empty watermark layer with the same shape as the image
        watermark = np.zeros_like(image, dtype=np.float32)
        
        # Apply different colors based on confidence levels
        high_mask = confidence_map >= self.options['high_confidence_threshold']
        medium_mask = (confidence_map >= self.options['low_confidence_threshold']) & (confidence_map < self.options['high_confidence_threshold'])
        low_mask = confidence_map < self.options['low_confidence_threshold']
        
        # Set colors for each region
        for i in range(3):  # RGB channels
            watermark[:,:,i][high_mask] = self.options['high_confidence_color'][i]
            watermark[:,:,i][medium_mask] = self.options['medium_confidence_color'][i]
            watermark[:,:,i][low_mask] = self.options['low_confidence_color'][i]
        
        # Blend the watermark with the original image
        intensity = self.options['watermark_intensity']
        watermarked_image = (1 - intensity) * image + intensity * watermark
        
        # Ensure values are within valid range [0, 255]
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        
        return watermarked_image
    
    def apply_pixel_mask(self, image, generation_mask, color=(255, 0, 0), intensity=0.2):
        """
        Apply a colored mask to indicate AI-generated regions
        
        Args:
            image (numpy.ndarray): The original image (HxWx3, RGB)
            generation_mask (numpy.ndarray): Binary mask where 1 indicates AI-generated pixels (HxW)
            color (tuple): RGB color for the mask
            intensity (float): Transparency level for the mask
            
        Returns:
            numpy.ndarray: Image with AI-generated regions marked
        """
        if image.shape[:2] != generation_mask.shape:
            raise ValueError("Image and generation mask dimensions must match")
        
        # Create a colored mask
        mask = np.zeros_like(image, dtype=np.float32)
        for i in range(3):  # RGB channels
            mask[:,:,i][generation_mask == 1] = color[i]
        
        # Blend the mask with the original image
        masked_image = (1 - intensity) * image + intensity * mask
        masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
        
        return masked_image
    
    def create_histogram_watermark(self, image):
        """
        Modify the color histogram subtly to include a watermark
        
        Args:
            image (numpy.ndarray): The original image (HxWx3, RGB)
            
        Returns:
            numpy.ndarray: Image with histogram-based watermark
        """
        # This is a simplified example - in practice, more sophisticated
        # methods would be used to embed information in the histogram
        
        # Convert to HSV color space for better control
        from skimage.color import rgb2hsv, hsv2rgb
        
        hsv_image = rgb2hsv(image.astype(np.float32) / 255.0)
        
        # Modify the hue channel slightly to encode watermark
        # This pattern could encode information about AI generation
        pattern = np.sin(np.linspace(0, 10*np.pi, image.shape[0])) * 0.01
        for i in range(image.shape[0]):
            hsv_image[i,:,0] += pattern[i]
        
        # Ensure hue values stay in range [0, 1]
        hsv_image[:,:,0] = hsv_image[:,:,0] % 1.0
        
        # Convert back to RGB
        watermarked_image = hsv2rgb(hsv_image) * 255
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        
        return watermarked_image

    def save_comparison(self, original, watermarked, output_path):
        """
        Save a side-by-side comparison of original and watermarked images
        
        Args:
            original (numpy.ndarray): Original image
            watermarked (numpy.ndarray): Watermarked image
            output_path (str): Path to save the comparison image
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(watermarked)
        axes[1].set_title('Watermarked Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


# Example usage
if __name__ == "__main__":
    print("AI Watermark - Image Watermarking Example")
    print("Based on Ucaretron Inc. patent technology")
    
    # This is a demonstration - in a real implementation, you would load actual images
    # Create a sample image and confidence map
    width, height = 300, 200
    
    # Create a simple test image (a gradient)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = i * 255 // height  # R channel
            image[i, j, 1] = j * 255 // width   # G channel
            image[i, j, 2] = 128                # B channel
    
    # Create a sample confidence map (higher confidence in the center)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    confidence_map = 1 - np.sqrt(((x - center_x) / (width/2)) ** 2 + ((y - center_y) / (height/2)) ** 2)
    confidence_map = np.clip(confidence_map, 0, 1)
    
    # Create a binary mask for AI-generated regions (right half of the image in this example)
    generation_mask = np.zeros((height, width), dtype=np.uint8)
    generation_mask[:, width//2:] = 1
    
    # Apply watermarking
    watermarker = ImageWatermarker()
    
    # Example 1: Apply confidence-based color watermark
    watermarked_image = watermarker.apply_watermark(image, confidence_map)
    
    # Example 2: Apply mask to show AI-generated regions
    masked_image = watermarker.apply_pixel_mask(image, generation_mask)
    
    # Example 3: Apply histogram-based watermark
    histogram_watermarked = watermarker.create_histogram_watermark(image)
    
    print("Watermark applied successfully. In a real application, you would save these images.")
    print("This example demonstrates the principles from the Ucaretron Inc. patent on AI watermarking.")
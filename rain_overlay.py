import os
from PIL import Image
import numpy as np
from pathlib import Path


class RainOverlay:
    """Apply rain mask overlays to images."""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def apply_overlay(self, base_image, rain_mask, blend_mode='add', opacity=0.1):
        """
        Apply a rain mask overlay to a base image.
        
        Args:
            base_image: PIL Image (base image)
            rain_mask: PIL Image (grayscale rain mask)
            blend_mode: 'add', 'screen', or 'lighten'
            opacity: Float 0-1, controls rain visibility
        
        Returns:
            PIL Image with rain overlay applied
        """
        # Ensure both images are the same size
        if base_image.size != rain_mask.size:
            rain_mask = rain_mask.resize(base_image.size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if base_image.mode != 'RGB':
            base_image = base_image.convert('RGB')
        
        # Convert mask to RGB
        if rain_mask.mode != 'L':
            rain_mask = rain_mask.convert('L')
        
        # Convert to numpy arrays
        base_array = np.array(base_image, dtype=np.float32)
        mask_array = np.array(rain_mask, dtype=np.float32)
        
        # Normalize mask to 0-1 range
        mask_normalized = mask_array / 255.0
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
        # Apply blend mode
        if blend_mode == 'add':
            # Add rain brightness to image
            result = base_array + (mask_3d * 255.0 * opacity)
            result = np.clip(result, 0, 255)
        
        elif blend_mode == 'screen':
            # Screen blend mode (lighter result)
            result = 255 - (255 - base_array) * (255 - mask_3d * 255) / 255
            result = base_array + (result - base_array) * opacity
            result = np.clip(result, 0, 255)
        
        elif blend_mode == 'lighten':
            # Only lighten, never darken
            rain_component = mask_3d * 255.0 * opacity
            result = np.maximum(base_array, base_array + rain_component)
            result = np.clip(result, 0, 255)
        
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
        
        # Convert back to PIL Image
        result_image = Image.fromarray(result.astype(np.uint8), mode='RGB')
        return result_image
    
    def process_images(self, image_paths, mask_paths, output_dir, 
                      blend_mode='add', opacity=0.7):
        """
        Process multiple images with rain masks.
        
        Args:
            image_paths: List of paths to base images
            mask_paths: List of paths to rain masks
            output_dir: Directory to save results
            blend_mode: Blend mode for overlay
            opacity: Rain visibility
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for img_path in image_paths:
            img_name = Path(img_path).stem
            
            # Load base image
            base_img = Image.open(img_path)
            
            # Resize to target size
            base_img = base_img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Apply each mask
            for mask_path in mask_paths:
                mask_name = Path(mask_path).stem
                
                # Load rain mask
                rain_mask = Image.open(mask_path)
                
                # Apply overlay
                result = self.apply_overlay(base_img, rain_mask, blend_mode, opacity)
                
                # Save result
                output_filename = f"{img_name}_{mask_name}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                result.save(output_path, quality=95)
                
                results.append(output_path)
                print(f"Created: {output_filename}")
        
        return results
    
    def batch_process_folder(self, image_folder, mask_folder, output_dir,
                            blend_mode='add', opacity=0.7):
        """
        Process all images in a folder with all masks in another folder.
        
        Args:
            image_folder: Folder containing base images
            mask_folder: Folder containing rain masks
            output_dir: Directory to save results
            blend_mode: Blend mode for overlay
            opacity: Rain visibility
        """
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_folder).glob(f'*{ext}'))
            image_paths.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        # Find all masks
        mask_paths = list(Path(mask_folder).glob('*.png'))
        mask_paths = [str(p) for p in mask_paths]
        
        if not image_paths:
            print(f"No images found in {image_folder}")
            return []
        
        if not mask_paths:
            print(f"No masks found in {mask_folder}")
            return []
        
        print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
        print(f"Will generate {len(image_paths) * len(mask_paths)} overlays\n")
        
        # Process all combinations
        results = self.process_images(image_paths, mask_paths, output_dir, 
                                     blend_mode, opacity)
        
        return results


def main():
    """Apply rain masks to cumulonimbus images."""
    
    # Initialize overlay processor
    overlay = RainOverlay(target_size=(224, 224))
    
    # Define paths
    image_folder = './no-rain'  # Current directory (contains 6_cumulonimbus_*.jpg)
    mask_folder = 'output'  # Folder with rain masks
    output_folder = 'overlayed_images'
    
    # Process with different settings
    print("Creating rain overlays...")
    print("=" * 50)
    
    # Default settings (add mode, 70% opacity)
    results = overlay.batch_process_folder(
        image_folder=image_folder,
        mask_folder=mask_folder,
        output_dir=output_folder,
        blend_mode='add',
        opacity=0.1
    )
    
    print("\n" + "=" * 50)
    print(f"Done! Generated {len(results)} images in '{output_folder}/' folder")
    print(f"All images are {224}x{224} pixels")


if __name__ == '__main__':
    main()

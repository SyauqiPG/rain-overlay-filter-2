import numpy as np
from PIL import Image, ImageDraw
import random
import os


class RainMaskGenerator:
    """Generate realistic rain filter masks at 224x224 pixels."""
    
    def __init__(self, size=(224, 224)):
        self.size = size
        self.width, self.height = size
    
    def generate_rain_streaks(self, num_streaks=50, min_length=10, max_length=40, 
                             thickness=1, angle=-10, intensity=200, top_down=True):
        """
        Generate a rain mask with streaks (top-down perspective by default).
        
        Args:
            num_streaks: Number of rain streaks
            min_length: Minimum length of rain streaks
            max_length: Maximum length of rain streaks
            thickness: Thickness of rain streaks
            angle: Angle of rain streaks in degrees (used only if top_down=False)
            intensity: Brightness of rain streaks (0-255)
            top_down: If True, creates radial perspective (camera facing up). If False, parallel streaks.
        
        Returns:
            PIL Image of the rain mask
        """
        # Create a black background
        mask = Image.new('L', self.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Center point for radial perspective
        center_x = self.width / 2
        center_y = self.height / 2
        
        for _ in range(num_streaks):
            # Random starting position
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            
            # Random length
            length = random.randint(min_length, max_length)
            
            if top_down:
                # Calculate direction from center to point (radial perspective)
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    # Normalize and scale by length
                    dx = (dx / dist) * length
                    dy = (dy / dist) * length
                else:
                    # If at center, random direction
                    angle_rad = random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle_rad) * length
                    dy = np.sin(angle_rad) * length
                
                # Calculate end point (away from center)
                end_x = x + dx
                end_y = y + dy
            else:
                # Parallel streaks (original behavior)
                angle_rad = np.radians(angle)
                dx = np.cos(angle_rad + np.pi/2) * length
                dy = np.sin(angle_rad + np.pi/2) * length
                end_x = x + dx
                end_y = y + dy
            
            # Vary intensity slightly for realism
            streak_intensity = max(0, min(255, intensity + random.randint(-30, 30)))
            
            # Draw the streak
            draw.line([(x, y), (end_x, end_y)], 
                     fill=streak_intensity, 
                     width=thickness)
        
        return mask
    
    def generate_rain_drops(self, num_drops=30, min_radius=1, max_radius=3, 
                           intensity=180):
        """
        Generate a rain mask with circular drops.
        
        Args:
            num_drops: Number of rain drops
            min_radius: Minimum radius of drops
            max_radius: Maximum radius of drops
            intensity: Brightness of drops (0-255)
        
        Returns:
            PIL Image of the rain mask
        """
        mask = Image.new('L', self.size, 0)
        draw = ImageDraw.Draw(mask)
        
        for _ in range(num_drops):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            radius = random.randint(min_radius, max_radius)
            
            drop_intensity = max(0, min(255, intensity + random.randint(-20, 20)))
            
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        fill=drop_intensity)
        
        return mask
    
    def generate_combined_rain(self, num_streaks=50, num_drops=20, 
                              streak_params=None, drop_params=None):
        """
        Generate a rain mask combining both streaks and drops.
        
        Args:
            num_streaks: Number of rain streaks
            num_drops: Number of rain drops
            streak_params: Dictionary of parameters for streaks
            drop_params: Dictionary of parameters for drops
        
        Returns:
            PIL Image of the combined rain mask
        """
        # Default parameters
        if streak_params is None:
            streak_params = {}
        if drop_params is None:
            drop_params = {}
        
        # Generate streaks
        streaks = self.generate_rain_streaks(num_streaks=num_streaks, **streak_params)
        
        # Generate drops
        drops = self.generate_rain_drops(num_drops=num_drops, **drop_params)
        
        # Combine by taking maximum pixel values
        combined_array = np.maximum(np.array(streaks), np.array(drops))
        combined = Image.fromarray(combined_array.astype(np.uint8), mode='L')
        
        return combined
    
    def generate_noise_rain(self, density=0.02, intensity=200):
        """
        Generate a rain mask using noise-based approach.
        
        Args:
            density: Probability of rain pixels (0-1)
            intensity: Average brightness of rain pixels
        
        Returns:
            PIL Image of the rain mask
        """
        # Create random noise
        noise = np.random.random(self.size)
        rain_mask = np.zeros(self.size, dtype=np.uint8)
        
        # Apply threshold to create rain pixels
        rain_pixels = noise < density
        rain_mask[rain_pixels] = np.random.randint(
            max(0, intensity - 50), 
            min(255, intensity + 50), 
            size=rain_pixels.sum()
        )
        
        return Image.fromarray(rain_mask, mode='L')
    
    def save_mask(self, mask, filename):
        """Save the generated mask to a file."""
        mask.save(filename)
        print(f"Saved rain mask to {filename}")


def main():
    """Example usage of RainMaskGenerator."""
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize generator
    generator = RainMaskGenerator(size=(224, 224))
    
    # Generate different types of rain masks (top-down perspective)
    
    # 1. Light rain with streaks (top-down view)
    light_rain = generator.generate_rain_streaks(
        num_streaks=30,
        min_length=15,
        max_length=30,
        thickness=1,
        intensity=150,
        top_down=True
    )
    generator.save_mask(light_rain, 'output/light_rain_topdown_224x224.png')
    
    # 2. Heavy rain with streaks (top-down view)
    heavy_rain = generator.generate_rain_streaks(
        num_streaks=100,
        min_length=20,
        max_length=50,
        thickness=2,
        intensity=220,
        top_down=True
    )
    generator.save_mask(heavy_rain, 'output/heavy_rain_topdown_224x224.png')
    
    # 3. Rain drops
    rain_drops = generator.generate_rain_drops(
        num_drops=50,
        min_radius=1,
        max_radius=4,
        intensity=180
    )
    generator.save_mask(rain_drops, 'output/rain_drops_224x224.png')
    
    # 4. Combined rain (streaks + drops)
    combined_rain = generator.generate_combined_rain(
        num_streaks=60,
        num_drops=30,
        streak_params={'angle': -12, 'thickness': 1},
        drop_params={'min_radius': 1, 'max_radius': 3}
    )
    generator.save_mask(combined_rain, 'output/combined_rain_224x224.png')
    
    # 5. Noise-based rain
    noise_rain = generator.generate_noise_rain(
        density=0.03,
        intensity=200
    )
    generator.save_mask(noise_rain, 'output/noise_rain_224x224.png')
    
    print("\nGenerated 5 different rain masks in the 'output' folder!")
    print("All masks are 224x224 pixels.")


if __name__ == '__main__':
    main()

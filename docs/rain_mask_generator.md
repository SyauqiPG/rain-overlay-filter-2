# Rain Mask Generator Documentation

## Overview

`rain_mask_generator.py` is a Python script that generates synthetic rain filter masks at 224x224 pixels. It creates realistic rain effects with various patterns including streaks, drops, and noise-based rain.

## Purpose

Generate grayscale rain masks that can be overlayed onto images to simulate different rain conditions. This is essential for creating synthetic training data for rain detection models.

## Class: RainMaskGenerator

### Constructor

```python
RainMaskGenerator(size=(224, 224))
```

**Parameters:**
- `size`: Tuple of (width, height) for output masks (default: 224x224)

### Methods

#### 1. `generate_rain_streaks()`

Creates rain masks with diagonal streaks using a radial perspective (top-down view).

**Signature:**
```python
generate_rain_streaks(
    num_streaks=50,
    min_length=10,
    max_length=40,
    thickness=1,
    angle=-10,
    intensity=200,
    top_down=True
)
```

**Parameters:**
- `num_streaks` (int): Number of rain streaks to generate
- `min_length` (int): Minimum streak length in pixels
- `max_length` (int): Maximum streak length in pixels
- `thickness` (int): Streak width in pixels
- `angle` (float): Streak angle in degrees (used only if top_down=False)
- `intensity` (int): Brightness level 0-255
- `top_down` (bool): If True, creates radial perspective (camera facing up)

**Returns:**
- PIL Image (grayscale) with rain streaks

**Top-Down Perspective:**
When `top_down=True`, streaks radiate outward from the center, simulating rain falling toward a camera pointing upward. This creates a more realistic perspective effect.

**Example:**
```python
generator = RainMaskGenerator(size=(224, 224))
mask = generator.generate_rain_streaks(
    num_streaks=100,
    min_length=20,
    max_length=50,
    thickness=2,
    intensity=220,
    top_down=True
)
```

#### 2. `generate_rain_drops()`

Creates circular rain drops for close-up rain effects.

**Signature:**
```python
generate_rain_drops(
    num_drops=30,
    min_radius=1,
    max_radius=3,
    intensity=180
)
```

**Parameters:**
- `num_drops` (int): Number of rain drops
- `min_radius` (int): Minimum drop radius in pixels
- `max_radius` (int): Maximum drop radius in pixels
- `intensity` (int): Brightness level 0-255

**Returns:**
- PIL Image (grayscale) with rain drops

**Example:**
```python
drops = generator.generate_rain_drops(
    num_drops=50,
    min_radius=1,
    max_radius=4,
    intensity=180
)
```

#### 3. `generate_combined_rain()`

Combines both streaks and drops for more realistic rain effects.

**Signature:**
```python
generate_combined_rain(
    num_streaks=50,
    num_drops=20,
    streak_params=None,
    drop_params=None
)
```

**Parameters:**
- `num_streaks` (int): Number of streaks
- `num_drops` (int): Number of drops
- `streak_params` (dict): Additional parameters for `generate_rain_streaks()`
- `drop_params` (dict): Additional parameters for `generate_rain_drops()`

**Returns:**
- PIL Image (grayscale) combining streaks and drops

**Example:**
```python
combined = generator.generate_combined_rain(
    num_streaks=60,
    num_drops=30,
    streak_params={'thickness': 1, 'top_down': True},
    drop_params={'max_radius': 3}
)
```

#### 4. `generate_noise_rain()`

Creates rain using a noise-based approach for natural distribution.

**Signature:**
```python
generate_noise_rain(density=0.02, intensity=200)
```

**Parameters:**
- `density` (float): Probability of rain pixels (0-1)
- `intensity` (int): Average brightness of rain pixels

**Returns:**
- PIL Image (grayscale) with noise-based rain

**Example:**
```python
noise = generator.generate_noise_rain(density=0.03, intensity=200)
```

#### 5. `save_mask()`

Saves the generated mask to a file.

**Signature:**
```python
save_mask(mask, filename)
```

**Parameters:**
- `mask` (PIL.Image): The mask to save
- `filename` (str): Output file path

**Example:**
```python
generator.save_mask(mask, 'output/my_rain_mask.png')
```

## Usage Example

### Basic Usage

```python
from rain_mask_generator import RainMaskGenerator

# Initialize generator
generator = RainMaskGenerator(size=(224, 224))

# Generate light rain
light_rain = generator.generate_rain_streaks(
    num_streaks=30,
    intensity=150,
    top_down=True
)

# Save the mask
generator.save_mask(light_rain, 'light_rain.png')
```

### Running the Script

The script includes a `main()` function that generates 5 different rain mask variants:

```bash
python rain_mask_generator.py
```

**Output:**
- `output/light_rain_topdown_224x224.png` - Light rain effect
- `output/heavy_rain_topdown_224x224.png` - Heavy rain effect
- `output/rain_drops_224x224.png` - Rain drops only
- `output/combined_rain_topdown_224x224.png` - Streaks + drops
- `output/noise_rain_224x224.png` - Noise-based rain

## Output Format

All generated masks are:
- **Size**: 224x224 pixels
- **Format**: PNG (grayscale)
- **Color Mode**: 'L' (grayscale)
- **Value Range**: 0 (black/no rain) to 255 (white/bright rain)

## Use Cases

1. **Data Augmentation**: Generate synthetic rain for training datasets
2. **Rain Simulation**: Create realistic rain effects for images
3. **Model Training**: Produce labeled rain data for classification/detection
4. **Testing**: Create controlled rain conditions for algorithm testing
5. **Research**: Study rain patterns and effects

## Tips

- **Light Rain**: Use fewer streaks (20-30), lower intensity (120-150)
- **Heavy Rain**: Use more streaks (100+), higher intensity (200-230), thicker lines (2-3)
- **Drizzle**: Combine light streaks with small drops
- **Storm**: High streak count with large drops and high intensity
- **Realism**: Use `top_down=True` for perspective effect

## Dependencies

- `numpy`: Array operations
- `pillow` (PIL): Image creation and manipulation
- `random`: Randomization of rain patterns

## Technical Details

### Top-Down Radial Perspective

The radial perspective is calculated as:
1. Center point is at (width/2, height/2)
2. For each streak, calculate direction vector from center to starting point
3. Normalize and scale by streak length
4. Draw line from start point outward along the direction vector

This creates the effect of rain falling toward a camera pointing upward at the sky.

### Random Variation

Each rain element includes random variation in:
- Position (uniformly distributed)
- Length (within specified range)
- Intensity (±30 brightness variation)
- Size (for drops)

This prevents repetitive patterns and creates more natural-looking rain.

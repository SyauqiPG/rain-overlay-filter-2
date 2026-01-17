# Rain Overlay Documentation

## Overview

`rain_overlay.py` applies synthetic rain masks to cloud images, creating realistic rain-on-cloud composite images. This script combines base images (e.g., cumulonimbus clouds) with generated rain masks.

## Purpose

Create training data by overlaying synthetic rain masks onto clean cloud images. The script supports multiple blend modes and opacity levels for realistic rain effects.

## Class: RainOverlay

### Constructor

```python
RainOverlay(target_size=(224, 224))
```

**Parameters:**
- `target_size`: Tuple of (width, height) for output images (default: 224x224)

### Methods

#### 1. `apply_overlay()`

Applies a rain mask overlay to a base image.

**Signature:**
```python
apply_overlay(base_image, rain_mask, blend_mode='add', opacity=0.7)
```

**Parameters:**
- `base_image` (PIL.Image): Base image (e.g., cloud photo)
- `rain_mask` (PIL.Image): Grayscale rain mask
- `blend_mode` (str): Blending algorithm - 'add', 'screen', or 'lighten'
- `opacity` (float): Rain visibility (0.0 to 1.0)

**Returns:**
- PIL.Image (RGB) with rain overlay applied

**Blend Modes:**

1. **'add'** (Default):
   - Adds rain brightness directly to the image
   - Formula: `result = base + (mask * opacity)`
   - Best for: General purpose rain effects

2. **'screen'**:
   - Screen blend mode (lighter, more natural)
   - Formula: `result = 255 - (255 - base) * (255 - mask) / 255`
   - Best for: Subtle, atmospheric rain

3. **'lighten'**:
   - Only lightens pixels, never darkens
   - Formula: `result = max(base, base + mask * opacity)`
   - Best for: Bright rain streaks, lightning

**Example:**
```python
from PIL import Image
from rain_overlay import RainOverlay

overlay = RainOverlay(target_size=(224, 224))

# Load images
base = Image.open('clouds.jpg')
rain = Image.open('rain_mask.png')

# Apply overlay
result = overlay.apply_overlay(base, rain, blend_mode='add', opacity=0.7)
result.save('rainy_clouds.jpg')
```

#### 2. `process_images()`

Process multiple base images with multiple rain masks.

**Signature:**
```python
process_images(
    image_paths,
    mask_paths,
    output_dir,
    blend_mode='add',
    opacity=0.7
)
```

**Parameters:**
- `image_paths` (list): List of base image file paths
- `mask_paths` (list): List of rain mask file paths
- `output_dir` (str): Directory to save results
- `blend_mode` (str): Blending algorithm
- `opacity` (float): Rain visibility

**Returns:**
- List of output file paths

**Behavior:**
- Creates all combinations of base images × rain masks
- Resizes all images to target size (224x224)
- Saves results with descriptive filenames: `{base_name}_{mask_name}.jpg`

**Example:**
```python
results = overlay.process_images(
    image_paths=['cloud1.jpg', 'cloud2.jpg'],
    mask_paths=['light_rain.png', 'heavy_rain.png'],
    output_dir='output',
    blend_mode='add',
    opacity=0.7
)
# Creates: cloud1_light_rain.jpg, cloud1_heavy_rain.jpg,
#          cloud2_light_rain.jpg, cloud2_heavy_rain.jpg
```

#### 3. `batch_process_folder()`

Batch process all images in a folder with all masks in another folder.

**Signature:**
```python
batch_process_folder(
    image_folder,
    mask_folder,
    output_dir,
    blend_mode='add',
    opacity=0.7
)
```

**Parameters:**
- `image_folder` (str): Folder containing base images
- `mask_folder` (str): Folder containing rain masks
- `output_dir` (str): Directory to save results
- `blend_mode` (str): Blending algorithm
- `opacity` (float): Rain visibility

**Returns:**
- List of output file paths

**Supported Formats:**
- Input: `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive)
- Output: `.jpg` (quality=95)

**Example:**
```python
overlay = RainOverlay(target_size=(224, 224))

results = overlay.batch_process_folder(
    image_folder='.',
    mask_folder='output',
    output_dir='overlayed_images',
    blend_mode='add',
    opacity=0.7
)
```

## Usage Example

### Command-Line Usage

Run the script directly to process all images:

```bash
python rain_overlay.py
```

**Default Behavior:**
- Searches for images in current directory (`.`)
- Uses rain masks from `output/` folder
- Saves results to `overlayed_images/` folder
- Uses 'add' blend mode with 70% opacity

### Programmatic Usage

```python
from rain_overlay import RainOverlay
from PIL import Image

# Initialize
overlay = RainOverlay(target_size=(224, 224))

# Method 1: Single overlay
base_img = Image.open('cumulonimbus.jpg')
rain_mask = Image.open('heavy_rain.png')
result = overlay.apply_overlay(base_img, rain_mask, 'add', 0.7)
result.save('result.jpg')

# Method 2: Batch processing
overlay.batch_process_folder(
    image_folder='clouds',
    mask_folder='rain_masks',
    output_dir='rainy_clouds'
)
```

### Custom Settings

```python
# Light rain effect
light_rain = overlay.apply_overlay(
    base_image, 
    rain_mask, 
    blend_mode='screen',
    opacity=0.5
)

# Heavy rain effect
heavy_rain = overlay.apply_overlay(
    base_image,
    rain_mask,
    blend_mode='add',
    opacity=0.9
)

# Subtle atmospheric rain
subtle = overlay.apply_overlay(
    base_image,
    rain_mask,
    blend_mode='screen',
    opacity=0.3
)
```

## Output

### File Naming Convention

Output files are named: `{base_image_name}_{rain_mask_name}.jpg`

**Example:**
- Base: `6_cumulonimbus_000005.jpg`
- Mask: `heavy_rain_topdown_224x224.png`
- Output: `6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg`

### Output Specifications

- **Size**: 224x224 pixels (resized from originals)
- **Format**: JPEG
- **Quality**: 95 (high quality)
- **Color Mode**: RGB

## Typical Workflow

1. **Generate rain masks** using `rain_mask_generator.py`
2. **Prepare base images** (cloud photos at any resolution)
3. **Run overlay script** to create combinations
4. **Review results** in output folder
5. **Use for training** - overlayed images become "rain" class

## Parameters Guide

### Opacity Settings

| Opacity | Effect | Use Case |
|---------|--------|----------|
| 0.3-0.5 | Light rain | Drizzle, mist |
| 0.6-0.7 | Moderate rain | Normal rainfall |
| 0.8-1.0 | Heavy rain | Storm, downpour |

### Blend Mode Selection

| Mode | Characteristics | Best For |
|------|-----------------|----------|
| add | Direct addition, bold | Training data, clear rain |
| screen | Soft, natural | Realistic photos |
| lighten | Preserves darks | Bright streaks only |

## Implementation Details

### Image Processing Pipeline

1. **Load Images**: Read base image and rain mask
2. **Resize**: Scale both to target size (224x224)
3. **Convert**: Ensure base is RGB, mask is grayscale
4. **Normalize**: Convert to float32, normalize mask to 0-1
5. **Blend**: Apply selected blend mode
6. **Clip**: Ensure values stay in 0-255 range
7. **Convert**: Back to uint8 PIL Image
8. **Save**: Write to disk as JPEG

### NumPy Operations

The script uses NumPy for efficient array operations:
- Vectorized blending (fast)
- Automatic broadcasting for 3-channel masks
- Clipping to valid pixel range

## Dependencies

- `numpy`: Array operations and blending
- `pillow` (PIL): Image I/O and manipulation
- `pathlib`: File path handling

## Performance

- **Speed**: ~50-100ms per overlay (CPU)
- **Memory**: Minimal (224x224 images are small)
- **Batch**: Can process hundreds of images quickly

## Tips

1. **Quality vs Speed**: JPEG quality=95 balances file size and quality
2. **Consistency**: Use same opacity for all training images in a class
3. **Variety**: Use multiple rain masks per base image for diversity
4. **Testing**: Preview results with different blend modes before batch processing
5. **Storage**: Output folder can grow large with many combinations

## Troubleshooting

**Error: `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`**
- **Cause**: You are running with an older Pillow (often happens when not using the project virtual environment).
- **Fix**: Activate the venv and reinstall requirements, or upgrade Pillow:
    - `python -m pip install -U pillow`
    - Verify with: `python -c "import PIL; from PIL import Image; print(PIL.__version__, hasattr(Image,'Resampling'))"`

**Images look too bright:**
- Reduce opacity (try 0.5-0.6)
- Use 'screen' blend mode instead of 'add'

**Rain not visible:**
- Increase opacity (try 0.8-0.9)
- Use 'add' blend mode
- Check rain mask intensity

**Images too dark:**
- Check original image brightness
- Use 'lighten' blend mode
- Increase rain mask intensity when generating

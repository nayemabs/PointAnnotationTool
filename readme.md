# Ball Detection Dataset Tools

A comprehensive toolkit for creating and managing ball detection datasets. Convert bounding box annotations to cropped images and manually annotate ball positions with point-based labels.

## Features

- **Dataset Converter**: Converts YOLO bounding boxes to centered ball crops with adaptive sizing
- **Manual Annotation Tool**: GUI for point-based ball annotation with zoom and navigation
- **Demo System**: Complete workflow demonstration with sample data generation

## Quick Start

```bash
# Run complete demo with sample data
python demo.py

# Convert existing dataset
python ball_dataset_converter.py --dataset_path /path/to/dataset --output_path /path/to/output

# Launch manual annotation tool
python manual_annotation_tool.py
```

## File Structure

```
your_dataset/
├── images/          # Input images
├── labels/          # YOLO format labels (class x_center y_center width height)
└── output/
    ├── images/      # Generated crops
    └── labels/      # Point-based labels (class x_center y_center)
```

## Tools

### 1. Ball Dataset Converter (`ball_dataset_converter.py`)
Converts YOLO bounding box datasets to point-centered crops.

**Key Features:**
- Adaptive crop sizing based on vertical position (perspective effect)
- Automatic padding for edge cases
- Quality statistics and metadata export

**Usage:**
```bash
python ball_dataset_converter.py \
    --dataset_path /path/to/yolo/dataset \
    --output_path /path/to/crops \
    --crop_size 256
```

### 2. Manual Annotation Tool (`manual_annotation_tool.py`)
Interactive GUI for manual ball position annotation.

**Features:**
- Click-to-annotate ball centers
- Zoom in/out with mouse wheel
- Keyboard navigation (arrow keys)
- Auto-save annotations

**Controls:**
- **Click**: Add ball annotation
- **Arrow Keys**: Navigate images
- **Mouse Wheel**: Zoom in/out
- **Delete**: Undo last annotation
- **Ctrl+S**: Save annotations

### 3. Demo System (`demo.py`)
Complete workflow demonstration and testing.

**Options:**
```bash
# Full demo with synthetic data
python demo.py --num-images 10

# Process existing dataset
python demo.py --input-dataset /path/to/your/data

# Launch with manual tool
python demo.py --manual-tool
```

## Requirements

```bash
pip install opencv-python numpy matplotlib pillow pathlib
```

## Dataset Format

**Input (YOLO):**
```
# labels/image.txt
0 0.5 0.3 0.1 0.15    # person: class x_center y_center width height
1 0.7 0.8 0.05 0.05   # ball: class x_center y_center width height
```

**Output (Point-based):**
```
# labels/crop.txt
1 0.5 0.5             # ball: class x_center y_center
```

## Output Structure

The converter creates:
- **Cropped images**: Centered on ball positions
- **Point labels**: Ball center coordinates
- **Metadata**: Conversion statistics and parameters
- **Visualizations**: Sample crops for quality check

## Tips

- Use adaptive crop sizing (enabled by default) for perspective effects
- Manually review generated crops before training
- Consider data augmentation for better model generalization
- Save manual annotations frequently (Ctrl+S)

## License

MIT License - Feel free to use and modify for your projects.
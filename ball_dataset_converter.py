import os
import random
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
import argparse

class BallDatasetConverter:
    """
    Converts bounding box annotations to point-based crops for ball detection training.
    
    This class processes YOLO-format datasets containing person and ball annotations,
    extracts the center points from ball bounding boxes, and creates cropped images
    with the ball centered in the crop with adjusted labels for training.
    """
    
    def __init__(self, dataset_path: str, output_path: str, crop_size_base: int = 256):
        """
        Initialize the dataset converter.
        
        Args:
            dataset_path: Path to the original dataset containing images and labels
            output_path: Path where the new cropped dataset will be saved
            crop_size_base: Base size for crops (will be adjusted based on position)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.crop_size_base = crop_size_base
        
        # Create output directories
        self.output_images_path = self.output_path / "images"
        self.output_labels_path = self.output_path / "labels"
        self.output_images_path.mkdir(parents=True, exist_ok=True)
        self.output_labels_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_images_processed": 0,
            "total_balls_found": 0,
            "crops_created": 0,
            "failed_crops": 0
        }
    
    def parse_yolo_label(self, label_path: str) -> List[Dict]:
        """
        Parse YOLO format label file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            List of dictionaries containing class_id and normalized coordinates
        """
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            annotations.append({
                                'class_id': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
        except FileNotFoundError:
            print(f"Warning: Label file not found: {label_path}")
        except Exception as e:
            print(f"Error parsing label file {label_path}: {e}")
        
        return annotations
    
    def extract_ball_centers(self, image_path: str) -> List[Dict]:
        """
        Extract center points from ball bounding boxes in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing ball information with center points
        """
        # Get corresponding label file
        image_path = Path(image_path)
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        
        # Parse annotations
        annotations = self.parse_yolo_label(str(label_path))
        
        # Filter for ball class (class_id = 1)
        ball_annotations = [ann for ann in annotations if ann['class_id'] == 1]
        
        # Load image to get dimensions
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return []
            
            img_height, img_width = image.shape[:2]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []
        
        ball_centers = []
        for ann in ball_annotations:
            # Convert normalized coordinates to pixel coordinates
            x_center_px = ann['x_center'] * img_width
            y_center_px = ann['y_center'] * img_height
            width_px = ann['width'] * img_width
            height_px = ann['height'] * img_height
            
            ball_centers.append({
                'image_path': str(image_path),
                'original_bbox': {
                    'x_center': x_center_px,
                    'y_center': y_center_px,
                    'width': width_px,
                    'height': height_px
                },
                'center_point': {
                    'x': x_center_px,
                    'y': y_center_px
                },
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height
                }
            })
            
            self.stats["total_balls_found"] += 1
        
        return ball_centers
    
    def calculate_adaptive_crop_size(self, point_y: float, img_height: int) -> Tuple[int, int]:
        """
        Calculate crop size based on vertical position in image.
        Objects lower in the frame (closer to camera) get larger crops.
        
        Args:
            point_y: Y coordinate of the point
            img_height: Height of the image
            
        Returns:
            Tuple of (crop_width, crop_height)
        """
        # Normalize y position (0 = top, 1 = bottom)
        normalized_y = point_y / img_height
        
        # Scale factor: larger for bottom of image, smaller for top
        # Range from 0.7 (top) to 1.3 (bottom)
        scale_factor = 0.7 + (0.6 * normalized_y)
        
        crop_size = int(self.crop_size_base * scale_factor)
        
        # Ensure minimum size
        crop_size = max(crop_size, 64)
        
        return crop_size, crop_size
    
    def create_crop(self, ball_info: Dict, crop_id: str) -> bool:
        """
        Create a cropped image with the ball center as the crop center.
        
        Args:
            ball_info: Dictionary containing ball information
            crop_id: Unique identifier for the crop
            
        Returns:
            True if crop was created successfully, False otherwise
        """
        try:
            # Load original image
            image = cv2.imread(ball_info['image_path'])
            if image is None:
                print(f"Error: Could not load image {ball_info['image_path']}")
                return False
            
            img_height, img_width = image.shape[:2]
            center_x = ball_info['center_point']['x']
            center_y = ball_info['center_point']['y']
            
            # Calculate adaptive crop size
            crop_width, crop_height = self.calculate_adaptive_crop_size(center_y, img_height)
            
            # Calculate crop boundaries centered on the ball
            crop_x_min = int(center_x - crop_width / 2)
            crop_x_max = int(center_x + crop_width / 2)
            crop_y_min = int(center_y - crop_height / 2)
            crop_y_max = int(center_y + crop_height / 2)
            
            # Store original crop boundaries for label calculation
            original_crop_x_min = crop_x_min
            original_crop_y_min = crop_y_min
            original_crop_width = crop_x_max - crop_x_min
            original_crop_height = crop_y_max - crop_y_min
            
            # Ensure crop is within image boundaries
            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(img_width, crop_x_max)
            crop_y_max = min(img_height, crop_y_max)
            
            # Extract crop
            crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            # Skip if crop is too small
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                print(f"Warning: Crop too small for {crop_id}")
                return False
            
            # If crop was clipped by image boundaries, pad it to maintain the ball in center
            actual_crop_height, actual_crop_width = crop.shape[:2]
            
            if (actual_crop_width != original_crop_width or actual_crop_height != original_crop_height):
                # Create padded crop to maintain original dimensions
                padded_crop = np.zeros((original_crop_height, original_crop_width, 3), dtype=np.uint8)
                
                # Calculate where to place the actual crop in the padded image
                pad_y_start = crop_y_min - original_crop_y_min
                pad_x_start = crop_x_min - original_crop_x_min
                
                # Place the crop in the padded image
                padded_crop[pad_y_start:pad_y_start + actual_crop_height, 
                           pad_x_start:pad_x_start + actual_crop_width] = crop
                
                crop = padded_crop
                crop_actual_width = original_crop_width
                crop_actual_height = original_crop_height
                # Use original boundaries for label calculation
                label_crop_x_min = original_crop_x_min
                label_crop_y_min = original_crop_y_min
            else:
                crop_actual_width = actual_crop_width
                crop_actual_height = actual_crop_height
                label_crop_x_min = crop_x_min
                label_crop_y_min = crop_y_min
            
            # Save cropped image
            crop_image_path = self.output_images_path / f"{crop_id}.jpg"
            cv2.imwrite(str(crop_image_path), crop)
            
            # Create label for the crop
            # The ball center relative to the crop (should be at center of crop)
            ball_x_in_crop = ball_info['original_bbox']['x_center'] - label_crop_x_min
            ball_y_in_crop = ball_info['original_bbox']['y_center'] - label_crop_y_min
            
            # Convert to normalized YOLO format
            norm_x = ball_x_in_crop / crop_actual_width
            norm_y = ball_y_in_crop / crop_actual_height
            norm_width = ball_info['original_bbox']['width'] / crop_actual_width
            norm_height = ball_info['original_bbox']['height'] / crop_actual_height
            
            # Ensure normalized values are within reasonable bounds
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # Save label file (class 1 for ball)
            crop_label_path = self.output_labels_path / f"{crop_id}.txt"
            with open(crop_label_path, 'w') as f:
                f.write(f"1 {norm_x:.6f} {norm_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            self.stats["crops_created"] += 1
            return True
            
        except Exception as e:
            print(f"Error creating crop {crop_id}: {e}")
            self.stats["failed_crops"] += 1
            return False
    
    def process_dataset(self, save_metadata: bool = True) -> Dict:
        """
        Process the entire dataset and create crops.
        
        Args:
            save_metadata: Whether to save metadata about the conversion process
            
        Returns:
            Dictionary containing all extracted ball centers and statistics
        """
        print("Starting dataset processing...")
        
        # Find all images in the dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        images_dir = self.dataset_path / "images"
        if images_dir.exists():
            for ext in image_extensions:
                image_paths.extend(images_dir.glob(f"*{ext}"))
                image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"No images found in {images_dir}")
            return {}
        
        print(f"Found {len(image_paths)} images to process")
        
        all_ball_centers = []
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")
            
            # Extract ball centers from this image
            ball_centers = self.extract_ball_centers(str(image_path))
            
            # Create crops for each ball center
            for j, ball_info in enumerate(ball_centers):
                crop_id = f"{image_path.stem}_ball_{j}"
                success = self.create_crop(ball_info, crop_id)
                
                if success:
                    ball_info['crop_id'] = crop_id
                    all_ball_centers.append(ball_info)
            
            self.stats["total_images_processed"] += 1
        
        # Save metadata if requested
        if save_metadata:
            metadata = {
                'ball_centers': all_ball_centers,
                'statistics': self.stats,
                'parameters': {
                    'crop_size_base': self.crop_size_base,
                    'dataset_path': str(self.dataset_path),
                    'output_path': str(self.output_path)
                }
            }
            
            metadata_path = self.output_path / "conversion_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Metadata saved to {metadata_path}")
        
        # Print final statistics
        print("\n" + "="*50)
        print("DATASET CONVERSION COMPLETE")
        print("="*50)
        print(f"Images processed: {self.stats['total_images_processed']}")
        print(f"Balls found: {self.stats['total_balls_found']}")
        print(f"Crops created: {self.stats['crops_created']}")
        print(f"Failed crops: {self.stats['failed_crops']}")
        print(f"Success rate: {self.stats['crops_created']/(self.stats['total_balls_found'] or 1)*100:.1f}%")
        print(f"Output directory: {self.output_path}")
        
        return {
            'ball_centers': all_ball_centers,
            'statistics': self.stats
        }

def main():
    """Main function to run the dataset conversion."""
    parser = argparse.ArgumentParser(description='Convert ball bounding boxes to center-based crops')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the original dataset (should contain images/ and labels/ folders)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path for the output cropped dataset')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Base crop size (will be adapted based on position)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create converter and process dataset
    converter = BallDatasetConverter(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        crop_size_base=args.crop_size
    )
    
    result = converter.process_dataset(save_metadata=True)
    
    return result

if __name__ == "__main__":
    main()
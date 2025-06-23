import os
import sys
import argparse
import time
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Import our custom modules 
try:
    from ball_dataset_converter import BallDatasetConverter
    from manual_annotation_tool import BallAnnotationTool
except ImportError:
    print("Error: Could not import required modules. Make sure ball_dataset_converter.py and manual_annotation_tool.py are in the same directory.")
    sys.exit(1)

class DatasetDemo:
    """
    Demonstration class that showcases the complete workflow for ball detection dataset creation.
    """
    
    def __init__(self, demo_dir: str = "demo_data", preserve_filenames: bool = True):
        """Initialize the demo with a working directory."""
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(exist_ok=True)
        self.preserve_filenames = preserve_filenames
        
        # Create subdirectories
        self.input_dir = self.demo_dir / "input_dataset"
        self.output_dir = self.demo_dir / "output_crops"
        self.manual_annotations = self.demo_dir / "manual_annotations"
        self.results_dir = self.demo_dir / "results"
        
        for dir_path in [self.input_dir, self.output_dir, self.manual_annotations, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Create input subdirectories (YOLO format)
        (self.input_dir / "images").mkdir(exist_ok=True)
        (self.input_dir / "labels").mkdir(exist_ok=True)
    
    def create_sample_dataset(self, num_images: int = 5, filename_prefix: str = "sample") -> None:
        """
        Create a sample dataset with synthetic images and YOLO annotations for demonstration.
        
        Args:
            num_images: Number of sample images to create
            filename_prefix: Prefix for generated filenames (to avoid conflicts with existing data)
        """
        print("Creating sample dataset for demonstration...")
        
        np.random.seed(42)  # For reproducible demo
        
        for i in range(num_images):
            # Create synthetic image (simulating a sports field)
            img_width, img_height = 800, 600
            image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Create a gradient background (simulating field)
            for y in range(img_height):
                intensity = int(20 + (y / img_height) * 60)  # Darker at top, lighter at bottom
                image[y, :] = [intensity, intensity + 20, intensity]
            
            # Add some noise for realism
            noise = np.random.randint(0, 30, (img_height, img_width, 3))
            image = cv2.add(image, noise.astype(np.uint8))
            
            # Generate random ball positions and sizes
            balls = []
            num_balls = np.random.randint(1, 4)  # 1-3 balls per image
            
            for j in range(num_balls):
                # Ball position (avoid edges)
                center_x = np.random.randint(50, img_width - 50)
                center_y = np.random.randint(50, img_height - 50)
                
                # Ball size (smaller if higher in image - perspective effect)
                size_factor = 0.3 + 0.7 * (center_y / img_height)  # Larger towards bottom
                ball_radius = int(8 + size_factor * 15)
                
                # Draw ball
                cv2.circle(image, (center_x, center_y), ball_radius, (255, 255, 255), -1)  # White ball
                cv2.circle(image, (center_x, center_y), ball_radius, (0, 0, 0), 2)  # Black outline
                
                # Add some shading for 3D effect
                cv2.circle(image, (center_x - ball_radius//3, center_y - ball_radius//3), 
                          ball_radius//2, (200, 200, 200), -1)
                
                # Calculate YOLO format bounding box
                bbox_width = bbox_height = ball_radius * 2.2  # Slightly larger than ball
                
                # Convert to normalized YOLO format
                x_center_norm = center_x / img_width
                y_center_norm = center_y / img_height
                width_norm = bbox_width / img_width
                height_norm = bbox_height / img_height
                
                balls.append({
                    'class_id': 1,  # Ball class
                    'x_center': x_center_norm,
                    'y_center': y_center_norm,
                    'width': width_norm,
                    'height': height_norm
                })
            
            # Optionally add some "people" (rectangles)
            num_people = np.random.randint(0, 3)
            people = []
            
            for k in range(num_people):
                # Person position
                person_x = np.random.randint(100, img_width - 100)
                person_y = np.random.randint(200, img_height - 50)
                
                # Person size (perspective effect)
                size_factor = 0.4 + 0.6 * (person_y / img_height)
                person_width = int(30 + size_factor * 40)
                person_height = int(60 + size_factor * 80)
                
                # Draw simple person representation
                cv2.rectangle(image, 
                            (person_x - person_width//2, person_y - person_height),
                            (person_x + person_width//2, person_y),
                            (100, 150, 200), -1)  # Brown-ish color
                
                # YOLO format for person
                x_center_norm = person_x / img_width
                y_center_norm = (person_y - person_height/2) / img_height
                width_norm = person_width / img_width
                height_norm = person_height / img_height
                
                people.append({
                    'class_id': 0,  # Person class
                    'x_center': x_center_norm,
                    'y_center': y_center_norm,
                    'width': width_norm,
                    'height': height_norm
                })
            
            # Generate filename - use prefix to avoid conflicts with existing data
            base_filename = f"{filename_prefix}_{i:03d}"
            
            # Save image
            image_path = self.input_dir / "images" / f"{base_filename}.jpg"
            cv2.imwrite(str(image_path), image)
            
            # Save YOLO label file
            label_path = self.input_dir / "labels" / f"{base_filename}.txt"
            with open(label_path, 'w') as f:
                # Write all annotations (people + balls)
                for person in people:
                    f.write(f"{person['class_id']} {person['x_center']:.6f} {person['y_center']:.6f} "
                           f"{person['width']:.6f} {person['height']:.6f}\n")
                for ball in balls:
                    f.write(f"{ball['class_id']} {ball['x_center']:.6f} {ball['y_center']:.6f} "
                           f"{ball['width']:.6f} {ball['height']:.6f}\n")
        
        print(f"Created {num_images} sample images with annotations in {self.input_dir}")
        print(f"Files use prefix: {filename_prefix}_XXX")
    
    def process_existing_dataset(self, dataset_path: str) -> Dict:
        """
        Process an existing dataset without creating new sample data.
        
        Args:
            dataset_path: Path to existing dataset with images/ and labels/ subdirectories
            
        Returns:
            Dictionary containing conversion results and statistics
        """
        print("\n" + "="*50)
        print("PROCESSING EXISTING DATASET")
        print("="*50)
        print(f"Input dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        if not (dataset_path / "images").exists() or not (dataset_path / "labels").exists():
            raise FileNotFoundError(f"Dataset must contain 'images' and 'labels' subdirectories")
        
        # Count existing files
        image_files = list((dataset_path / "images").glob("*"))
        label_files = list((dataset_path / "labels").glob("*"))
        print(f"Found {len(image_files)} images and {len(label_files)} label files")
        
        # Initialize converter with existing dataset
        converter = BallDatasetConverter(
            dataset_path=str(dataset_path),
            output_path=str(self.output_dir),
            crop_size_base=256
        )
        
        # Record start time
        start_time = time.time()
        
        # Process dataset
        result = converter.process_dataset(save_metadata=True)
        
        # Record end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        return result
    
    def run_dataset_conversion(self) -> Dict:
        """
        Run the dataset conversion process on the demo input directory.
        
        Returns:
            Dictionary containing conversion results and statistics
        """
        print("\n" + "="*50)
        print("RUNNING DATASET CONVERSION")
        print("="*50)
        
        # Initialize converter
        converter = BallDatasetConverter(
            dataset_path=str(self.input_dir),
            output_path=str(self.output_dir),
            crop_size_base=256
        )
        
        # Record start time
        start_time = time.time()
        
        # Process dataset
        result = converter.process_dataset(save_metadata=True)
        
        # Record end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        return result
    
    def visualize_results(self, max_crops: int = 10) -> None:
        """
        Create visualizations of the conversion results.
        
        Args:
            max_crops: Maximum number of crops to visualize
        """
        print("\nCreating result visualizations...")
        
        # Find crop images
        crop_images = list((self.output_dir / "images").glob("*.jpg"))
        crop_images = crop_images[:max_crops]  # Limit for demonstration
        
        if not crop_images:
            print("No crop images found for visualization")
            return
        
        # Create subplot grid
        rows = 2
        cols = min(5, len(crop_images))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        fig.suptitle("Generated Ball Crops", fontsize=16)
        
        if len(crop_images) == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, crop_path in enumerate(crop_images[:rows*cols]):
            if i >= len(axes):
                break
                
            # Load and display crop
            crop_image = cv2.imread(str(crop_path))
            if crop_image is not None:
                crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                
                row = i // cols
                col = i % cols
                ax_idx = row * cols + col if rows > 1 else i
                
                if ax_idx < len(axes):
                    axes[ax_idx].imshow(crop_rgb)
                    axes[ax_idx].set_title(f"Crop {i+1}")
                    axes[ax_idx].axis('off')
        
        # Hide unused subplots
        for i in range(len(crop_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_dir / "crop_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {viz_path}")
    
    def generate_report(self, conversion_result: Dict) -> None:
        """
        Generate a comprehensive report of the conversion process.
        
        Args:
            conversion_result: Results from the dataset conversion
        """
        print("\nGenerating comprehensive report...")
        
        stats = conversion_result.get('statistics', {})
        
        report = {
            'conversion_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_dataset': str(self.input_dir),
                'output_dataset': str(self.output_dir),
                'processing_statistics': stats
            },
            'quality_metrics': {
                'success_rate': stats.get('crops_created', 0) / max(stats.get('total_balls_found', 1), 1) * 100,
                'average_crops_per_image': stats.get('crops_created', 0) / max(stats.get('total_images_processed', 1), 1),
                'failure_rate': stats.get('failed_crops', 0) / max(stats.get('total_balls_found', 1), 1) * 100
            },
            'dataset_composition': self._analyze_dataset_composition(),
            'recommendations': self._generate_recommendations(stats)
        }
        
        # Save report
        report_path = self.results_dir / "conversion_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary
        summary_path = self.results_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("BALL DETECTION DATASET CONVERSION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['conversion_summary']['timestamp']}\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Images processed: {stats.get('total_images_processed', 0)}\n")
            f.write(f"Balls detected: {stats.get('total_balls_found', 0)}\n")
            f.write(f"Crops created: {stats.get('crops_created', 0)}\n")
            f.write(f"Failed crops: {stats.get('failed_crops', 0)}\n")
            f.write(f"Success rate: {report['quality_metrics']['success_rate']:.1f}%\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"Report saved to {report_path}")
        print(f"Summary saved to {summary_path}")
    
    def _analyze_dataset_composition(self) -> Dict:
        """Analyze the composition of the generated dataset."""
        crop_images = list((self.output_dir / "images").glob("*.jpg"))
        crop_labels = list((self.output_dir / "labels").glob("*.txt"))
        
        return {
            'total_crops': len(crop_images),
            'total_labels': len(crop_labels),
            'label_coverage': len(crop_labels) / max(len(crop_images), 1) * 100
        }
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on processing statistics."""
        recommendations = []
        
        success_rate = stats.get('crops_created', 0) / max(stats.get('total_balls_found', 1), 1) * 100
        
        if success_rate < 90:
            recommendations.append("Consider adjusting crop size parameters to improve success rate")
        
        if stats.get('failed_crops', 0) > 0:
            recommendations.append("Review failed crops to identify common failure patterns")
        
        if stats.get('total_balls_found', 0) < stats.get('total_images_processed', 0) * 2:
            recommendations.append("Consider adding more images with multiple balls per image")
        
        recommendations.append("Validate crop quality manually before using for training")
        recommendations.append("Consider data augmentation to increase dataset diversity")
        
        return recommendations
    
    def run_complete_demo(self, create_samples: bool = True, filename_prefix: str = "sample") -> None:
        """
        Run the complete demonstration workflow.
        
        Args:
            create_samples: Whether to create sample data or use existing data
            filename_prefix: Prefix for sample filenames (only used if create_samples=True)
        """
        print("BALL DETECTION DATASET CONVERSION - COMPLETE DEMO")
        print("=" * 60)
        
        if create_samples:
            # Step 1: Create sample dataset
            self.create_sample_dataset(num_images=10, filename_prefix=filename_prefix)
            # Step 2: Run conversion on demo data
            conversion_result = self.run_dataset_conversion()
        else:
            print("Skipping sample creation - using existing data in input directory")
            # Step 2: Run conversion on existing data
            conversion_result = self.run_dataset_conversion()
        
        # Step 3: Create visualizations
        self.visualize_results()
        
        # Step 4: Generate report
        self.generate_report(conversion_result)
        
        # Step 5: Summary
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Demo files created in: {self.demo_dir}")
        print("\nTo explore results:")
        print(f"• View crops: {self.output_dir / 'images'}")
        print(f"• Check labels: {self.output_dir / 'labels'}")
        print(f"• Read report: {self.results_dir / 'summary.txt'}")
        print(f"• View visualization: {self.results_dir / 'crop_visualization.png'}")
        
        print("\nTo run manual annotation tool:")
        print("python manual_annotation_tool.py")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Ball Detection Dataset Conversion Demo')
    parser.add_argument('--demo-dir', type=str, default='demo_data',
                       help='Directory for demo files')
    parser.add_argument('--input-dataset', type=str, default=None,
                       help='Path to existing dataset (skip sample creation)')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of sample images to create')
    parser.add_argument('--filename-prefix', type=str, default='sample',
                       help='Prefix for generated sample filenames')
    parser.add_argument('--manual-tool', action='store_true',
                       help='Launch manual annotation tool after demo')
    parser.add_argument('--no-samples', action='store_true',
                       help='Skip sample creation and use existing data in demo input dir')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = DatasetDemo(demo_dir=args.demo_dir)
    
    # Determine processing mode
    if args.input_dataset:
        # Process existing external dataset
        print(f"Processing existing dataset: {args.input_dataset}")
        conversion_result = demo.process_existing_dataset(args.input_dataset)
        demo.visualize_results()
        demo.generate_report(conversion_result)
    elif args.no_samples:
        # Use existing data in demo input directory without creating samples
        demo.run_complete_demo(create_samples=False)
    else:
        # Create samples and run full demo
        demo.run_complete_demo(
            create_samples=True, 
            filename_prefix=args.filename_prefix
        )
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Optionally launch manual annotation tool
    if args.manual_tool:
        print("\nLaunching manual annotation tool...")
        app = BallAnnotationTool()
        app.run()

if __name__ == "__main__":
    main()
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
from pathlib import Path
import numpy as np

class BallAnnotationTool:
    """
    A manual annotation tool for marking ball positions with single-point labels.
    
    This tool allows users to click on ball positions in images and saves annotations
    in the format: class_id x_center y_center (normalized coordinates).
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ball Point Annotation Tool")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.image_list = []
        self.current_image_index = 0
        self.current_image = None
        self.original_image = None
        self.image_path = ""
        self.annotations = []
        self.output_dir = ""
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # UI components
        self.canvas = None
        self.filename_label = None
        self.setup_ui()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="Select Image Folder", 
                  command=self.select_image_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Select Output Folder", 
                  command=self.select_output_folder).pack(side=tk.LEFT, padx=5)
        
        # Navigation
        nav_frame = ttk.LabelFrame(control_frame, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="Previous", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.image_counter_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_counter_label.pack(side=tk.LEFT, padx=20)
        
        # Annotation controls
        ann_frame = ttk.LabelFrame(control_frame, text="Annotation Controls")
        ann_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(ann_frame, text="Clear All Points", 
                  command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(ann_frame, text="Undo Last Point", 
                  command=self.undo_last_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(ann_frame, text="Save Annotations", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        self.annotation_count_label = ttk.Label(ann_frame, text="Points: 0")
        self.annotation_count_label.pack(side=tk.LEFT, padx=20)
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(control_frame, text="Zoom Controls")
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(zoom_frame, text="Zoom In", 
                  command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom Out", 
                  command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Reset Zoom", 
                  command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=20)
        
        # Instructions
        instructions = ttk.LabelFrame(control_frame, text="Instructions")
        instructions.pack(fill=tk.X)
        
        instruction_text = ("1. Select folder with images and output folder\n"
                          "2. Click on ball centers in the image\n"
                          "3. Use navigation buttons or arrow keys to move between images\n"
                          "4. Use mouse wheel to zoom in/out, or zoom buttons\n"
                          "5. Annotations are saved automatically when moving to next image")
        ttk.Label(instructions, text=instruction_text, wraplength=800).pack(padx=5, pady=5)
        
        # Canvas frame with filename display
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor='crosshair')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status bar for filename display
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.filename_label = ttk.Label(status_frame, text="No image loaded", 
                                      font=("Arial", 9), foreground="blue")
        self.filename_label.pack(side=tk.LEFT)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)    # Linux scroll down
        
        # Bind keyboard events
        self.root.bind("<Left>", lambda e: self.previous_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.undo_last_annotation())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        
        # Make sure the root window can receive focus for key events
        self.root.focus_set()
    
    def select_image_folder(self):
        """Select folder containing images to annotate."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.load_images(folder)
    
    def select_output_folder(self):
        """Select folder for saving annotations."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir = folder
            messagebox.showinfo("Success", f"Output folder set to: {folder}")
    
    def load_images(self, folder_path):
        """Load all images from the selected folder."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_list = []
        
        for ext in image_extensions:
            self.image_list.extend(Path(folder_path).glob(f"*{ext}"))
            self.image_list.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        self.image_list.sort()
        
        if self.image_list:
            self.current_image_index = 0
            self.load_current_image()
            messagebox.showinfo("Success", f"Loaded {len(self.image_list)} images")
        else:
            messagebox.showwarning("Warning", "No images found in the selected folder")
    
    def load_current_image(self):
        """Load and display the current image."""
        if not self.image_list:
            return
        
        # Save current annotations before loading new image
        if self.current_image is not None:
            self.save_annotations()
        
        # Load new image
        self.image_path = str(self.image_list[self.current_image_index])
        self.original_image = cv2.imread(self.image_path)
        
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.image_path}")
            return
        
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Calculate display scale to fit canvas initially
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            img_height, img_width = rgb_image.shape[:2]
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            base_scale = min(scale_w, scale_h, 1.0)  # Don't scale up initially
        else:
            base_scale = 1.0
        
        # Apply zoom factor to the base scale
        self.display_scale = base_scale * self.zoom_factor
        
        # Resize image for display
        display_width = int(rgb_image.shape[1] * self.display_scale)
        display_height = int(rgb_image.shape[0] * self.display_scale)
        
        self.current_image = cv2.resize(rgb_image, (display_width, display_height))
        
        # Load existing annotations if they exist
        self.load_existing_annotations()
        
        # Display image
        self.display_image()
        
        # Update UI
        self.update_image_counter()
        self.update_annotation_counter()
        self.update_filename_display()
        self.update_zoom_display()
    
    def load_existing_annotations(self):
        """Load existing annotations for the current image if they exist."""
        if not self.output_dir:
            self.annotations = []
            return
        
        image_name = Path(self.image_path).stem
        annotation_file = Path(self.output_dir) / f"{image_name}.txt"
        
        self.annotations = []
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 3:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                self.annotations.append({
                                    'class_id': class_id,
                                    'x_center': x_center,
                                    'y_center': y_center
                                })
            except Exception as e:
                print(f"Error loading annotations: {e}")
                self.annotations = []
    
    def display_image(self):
        """Display the current image with annotations on canvas."""
        if self.current_image is None:
            return
        
        # Get original image dimensions
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate base scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            base_scale = min(scale_w, scale_h, 1.0)
        else:
            base_scale = 1.0
        
        # Apply zoom factor
        self.display_scale = base_scale * self.zoom_factor
        
        # Resize image for display
        display_width = int(img_width * self.display_scale)
        display_height = int(img_height * self.display_scale)
        
        # Convert BGR to RGB and resize
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        display_image = cv2.resize(rgb_image, (display_width, display_height))
        
        # Draw annotation points
        for ann in self.annotations:
            # Convert normalized coordinates to display coordinates
            x_display = int(ann['x_center'] * display_width)
            y_display = int(ann['y_center'] * display_height)
            
            # Scale point size with zoom
            point_size = max(4, int(8 * self.zoom_factor))
            border_size = max(6, int(12 * self.zoom_factor))
            
            # Draw point
            cv2.circle(display_image, (x_display, y_display), point_size, (255, 0, 0), -1)  # Red filled circle
            cv2.circle(display_image, (x_display, y_display), border_size, (255, 255, 255), 2)  # White border
        
        # Convert to PIL Image and display
        pil_image = Image.fromarray(display_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and add image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_click(self, event):
        """Handle canvas click to add annotation."""
        if self.current_image is None or not self.output_dir:
            if not self.output_dir:
                messagebox.showwarning("Warning", "Please select an output folder first")
            return
        
        # Get click coordinates relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to original image coordinates
        img_height, img_width = self.original_image.shape[:2]
        
        original_x = canvas_x / self.display_scale
        original_y = canvas_y / self.display_scale
        
        # Check if coordinates are within image bounds
        if 0 <= original_x < img_width and 0 <= original_y < img_height:
            # Convert to normalized coordinates
            norm_x = original_x / img_width
            norm_y = original_y / img_height
            
            # Add annotation (class_id = 1 for ball)
            self.annotations.append({
                'class_id': 1,
                'x_center': norm_x,
                'y_center': norm_y
            })
            
            # Refresh display
            self.display_image()
            self.update_annotation_counter()
    
    def clear_annotations(self):
        """Clear all annotations for current image."""
        self.annotations = []
        self.display_image()
        self.update_annotation_counter()
    
    def undo_last_annotation(self):
        """Remove the last annotation."""
        if self.annotations:
            self.annotations.pop()
            self.display_image()
            self.update_annotation_counter()
    
    def save_annotations(self):
        """Save annotations for current image."""
        if not self.output_dir or not self.image_path:
            return
        
        image_name = Path(self.image_path).stem
        annotation_file = Path(self.output_dir) / f"{image_name}.txt"
        
        try:
            with open(annotation_file, 'w') as f:
                for ann in self.annotations:
                    f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save annotations: {e}")
    
    def previous_image(self):
        """Go to previous image."""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image."""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def update_image_counter(self):
        """Update the image counter label."""
        if self.image_list:
            text = f"Image {self.current_image_index + 1} of {len(self.image_list)}"
            self.image_counter_label.config(text=text)
        else:
            self.image_counter_label.config(text="No images loaded")
    
    def update_annotation_counter(self):
        """Update the annotation counter label."""
        count = len(self.annotations)
        self.annotation_count_label.config(text=f"Points: {count}")
    
    def update_zoom_display(self):
        """Update the zoom level display."""
        zoom_percent = int(self.zoom_factor * 100)
        self.zoom_label.config(text=f"Zoom: {zoom_percent}%")
    
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom."""
        # Get mouse position relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up (zoom in)
            zoom_change = 1.1
        elif event.num == 5 or event.delta < 0:  # Scroll down (zoom out)
            zoom_change = 0.9
        else:
            return
        
        # Calculate new zoom factor
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_factor:
            # Store old scroll position
            old_scroll_x = self.canvas.canvasx(0)
            old_scroll_y = self.canvas.canvasy(0)
            
            # Update zoom
            self.zoom_factor = new_zoom
            self.display_image()
            self.update_zoom_display()
            
            # Try to maintain zoom center around mouse position
            self.canvas.update_idletasks()
    
    def zoom_in(self):
        """Zoom in by 20%."""
        new_zoom = self.zoom_factor * 1.2
        new_zoom = min(self.max_zoom, new_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.display_image()
            self.update_zoom_display()
    
    def zoom_out(self):
        """Zoom out by 20%."""
        new_zoom = self.zoom_factor * 0.8
        new_zoom = max(self.min_zoom, new_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.display_image()
            self.update_zoom_display()
    
    def reset_zoom(self):
        """Reset zoom to fit image in canvas."""
        self.zoom_factor = 1.0
        self.display_image()
        self.update_zoom_display()
    
    def update_filename_display(self):
        """Update the filename display at the bottom."""
        if self.image_path:
            filename = Path(self.image_path).name
            self.filename_label.config(text=f"Current file: {filename}")
        else:
            self.filename_label.config(text="No image loaded")
    
    def on_canvas_configure(self, event):
        """Handle canvas resize events."""
        if self.original_image is not None:
            # Don't reset zoom on resize, just update display
            self.display_image()
    
    def on_closing(self):
        """Handle application closing."""
        # Save current annotations before closing
        if self.current_image is not None:
            self.save_annotations()
        
        # Ask for confirmation
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
    
    def run(self):
        """Start the annotation tool."""
        self.root.mainloop()

def main():
    """Main function to run the annotation tool."""
    app = BallAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
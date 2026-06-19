#!/usr/bin/env python
"""
hdyst_active_learning_cli.py - Active Learning for hdyst+/hdyst- Cell Classification

This script provides a command-line interface for:
1. Initial random sampling and manual labeling of cells
2. Training a Random Forest classifier on labeled data
3. Selecting cells with highest uncertainty for subsequent labeling rounds
4. Iteratively improving the model through human-in-the-loop learning

Usage:
    python hdyst_active_learning_cli.py --csv_file 20250215_abe_df.csv --cropped_cells_dir ./cropped_cells

Requirements:
    - pandas, numpy, matplotlib, opencv-python, tkinter, scikit-learn
    - The abe_df CSV file
    - Directory containing the cropped cell pickle files
"""

import pandas as pd
import numpy as np
import os
import pickle
import cv2
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import argparse
import datetime
import time
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import threading
import random
import concurrent.futures


# ------ Cell Processing Functions ------

def process_cell_red(cell_data, pad_size):
    """
    Process a cell's image data focusing on the red channel:
      - Convert image to a NumPy array if needed
      - Dilate the binary cropped mask 5 times using a 3x3 kernel
      - Apply the dilated mask to the image
      - Extract only the red channel and scale it
      - Pad (or crop) the resulting one-channel image to a square of side pad_size
    """
    mask = cell_data['cropped_mask']
    image = cell_data['image']
    
    # Convert to numpy array if needed
    if not hasattr(image, "shape"):
        image = np.array(image)
    
    # Process mask
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=5)
    
    # Apply mask to image
    masked_image = image.copy() * dilated_mask[..., np.newaxis]
    
    # Create a binary mask for cleaner background
    background_mask = dilated_mask == 0
    
    # Set background pixels to 0 explicitly
    masked_image[background_mask] = 0
    
    # Extract and scale red channel
    processed_red = masked_image[..., 0] * 10  # Scale red channel by 10
    
    # Add channel dimension
    processed_image = processed_red[..., np.newaxis]
    
    # Pad to square
    H, W, _ = processed_image.shape
    pad_height = max(pad_size - H, 0)
    pad_width = max(pad_size - W, 0)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_image = np.pad(processed_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    
    # Crop to desired size
    H_new, W_new, _ = padded_image.shape
    start_y = (H_new - pad_size) // 2
    start_x = (W_new - pad_size) // 2
    cropped_image = padded_image[start_y:start_y+pad_size, start_x:start_x+pad_size, :]
    
    return cropped_image.squeeze()  # Return the 2D image


def load_cell_data_dict(df, folder="cropped_cells"):
    """
    Load cell data for all cells in the provided DataFrame in parallel
    """
    cell_data_dict = {}
    unique_files = df["filename"].unique()
    
    # Function to process a single file
    def process_file(filename):
        # Filter for the current image
        file_df = df[df['filename'] == filename]
        
        base = filename.split('.')[0]
        file_path = os.path.join(folder, f"{base}_cropped_cells.pkl")
        result = {}
        try:
            with open(file_path, "rb") as f:
                cell_dict = pickle.load(f)
                
            # Only load cells that are in the dataframe
            for cell_id in file_df['cell_id'].unique():
                if cell_id in cell_dict:
                    result[(filename, cell_id)] = cell_dict[cell_id]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return result
    
    print(f"Loading all {len(unique_files)} cell files at once...")
    
    # Use ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map the process_file function to all filenames
        future_to_filename = {executor.submit(process_file, filename): filename for filename in unique_files}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_filename), 
                          total=len(unique_files), 
                          desc="Loading cell files"):
            filename = future_to_filename[future]
            try:
                result = future.result()
                cell_data_dict.update(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Loaded data for {len(cell_data_dict)} cells")
    return cell_data_dict


def determine_pad_size(df, folder="cropped_cells", percentile=99.9, use_all_cells=True, 
                       sample_frac=0.1, random_state=42):
    """
    Determine padding size based on the dimensions of the cells in the DataFrame
    """
    # Use all cells or sample a fraction to reduce computation time
    if use_all_cells:
        sample_df = df.copy()
        print(f"Using all {len(sample_df)} cells for pad size determination")
    else:
        sample_df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"Using {len(sample_df)} cells ({sample_frac*100:.1f}% sample) for pad size determination")
    
    # Process all cells at once
    print("Loading all cell data for pad size determination...")
    cell_data_dict = load_cell_data_dict(sample_df, folder)
    
    # Compute dimensions
    dims = []
    for key, cell_data in tqdm(cell_data_dict.items(), desc="Computing dimensions"):
        image = cell_data['image']
        if not hasattr(image, "shape"):
            image = np.array(image)
        dims.append(max(image.shape[0], image.shape[1]))
    
    pad_size = int(np.ceil(np.percentile(dims, percentile)))
    print(f"Determined pad size: {pad_size} (based on {percentile}th percentile)")
    
    # Clean up to free memory
    del cell_data_dict, dims
    gc.collect()
    
    return pad_size


def select_random_cells(df, n_samples=600, random_state=42):
    """
    Select a random subset of cells from wildtype, untreated, and treated groups
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Identify treatment groups
    wildtype_df = df[df['plotLabel'] == 'Wildtype']
    untreated_df = df[df['plotLabel'] == 'Untreated']
    treated_df = df[~df['plotLabel'].isin(['Wildtype', 'Untreated'])]
    
    print(f"Available cells - Wildtype: {len(wildtype_df)}, Untreated: {len(untreated_df)}, Treated: {len(treated_df)}")
    
    # Calculate samples from each group (stratified sampling)
    n_wildtype = min(n_samples // 3, len(wildtype_df))
    n_untreated = min(n_samples // 3, len(untreated_df))
    n_treated = min(n_samples - n_wildtype - n_untreated, len(treated_df))
    
    # Sample from each group
    wildtype_sample = wildtype_df.sample(n=n_wildtype, random_state=random_state)
    untreated_sample = untreated_df.sample(n=n_untreated, random_state=random_state)
    treated_sample = treated_df.sample(n=n_treated, random_state=random_state)
    
    # Combine the samples
    combined_sample = pd.concat([wildtype_sample, untreated_sample, treated_sample])
    
    # Shuffle the combined sample
    combined_sample = combined_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Selected cells - Wildtype: {len(wildtype_sample)}, Untreated: {len(untreated_sample)}, "
          f"Treated: {len(treated_sample)}, Total: {len(combined_sample)}")
    
    return combined_sample


# ------ GUI for Cell Labeling ------

class CellLabelerGUI:
    def __init__(self, root, sample_df, cell_data_dict, pad_size, model=None, uncertainty_scores=None):
        """
        Interactive GUI for manual labeling of cells as hdyst+ or hdyst-
        
        Parameters:
        -----------
        root : tk.Tk
            Root Tkinter window
        sample_df : pandas DataFrame
            DataFrame containing cells to label
        cell_data_dict : dict
            Dictionary of cell data
        pad_size : int
            Size to pad cell images to
        model : RandomForestClassifier, optional
            Trained classifier model for predictions
        uncertainty_scores : dict, optional
            Dictionary of uncertainty scores for each cell
        """
        self.root = root
        self.root.title("hdyst+/hdyst- Active Learning Labeler")
        self.root.geometry("1000x750")
        
        self.sample_df = sample_df.copy()
        self.cell_data_dict = cell_data_dict
        self.pad_size = pad_size
        self.current_index = 0
        self.labels = pd.Series(index=range(len(sample_df)), dtype=str)
        self.processed_images = {}
        self.model = model
        self.uncertainty_scores = uncertainty_scores
        
        # Global normalization parameters
        self.global_min = None
        self.global_max = None
        
        # Pre-process all images
        self._preprocess_all_images()
        
        # Create the UI components
        self._create_ui()
        
        # Load first image
        self._update_display()
        
    def _preprocess_all_images(self):
        """Pre-process all images in the sample and calculate global statistics for normalization"""
        print("Pre-processing images...")
        
        # First pass: process all images and collect pixel values for global statistics
        all_pixel_values = []
        
        for idx, row in tqdm(self.sample_df.iterrows(), total=len(self.sample_df), desc="Processing images"):
            key = (row['filename'], row['cell_id'])
            if key in self.cell_data_dict:
                cell_data = self.cell_data_dict[key]
                processed_img = process_cell_red(cell_data, self.pad_size)
                self.processed_images[idx] = processed_img
                
                # Apply scaling factor (12x)
                scaled_img = processed_img * 12.0
                
                # Collect non-zero pixel values for global statistics
                non_zero_pixels = scaled_img[scaled_img > 0.001].flatten()
                if len(non_zero_pixels) > 0:
                    # Sample to avoid memory issues with very large datasets
                    sample_size = min(1000, len(non_zero_pixels))
                    sampled_pixels = np.random.choice(non_zero_pixels, sample_size, replace=False)
                    all_pixel_values.append(sampled_pixels)
            else:
                print(f"Warning: Cell data not found for {key}")
        
        # Calculate global min and max based on percentiles
        if all_pixel_values:
            all_pixels = np.concatenate(all_pixel_values)
            self.global_min = np.percentile(all_pixels, 1)  # 1st percentile
            self.global_max = np.percentile(all_pixels, 99)  # 99th percentile
            print(f"Global intensity range - 1st percentile: {self.global_min:.6f}, 99th percentile: {self.global_max:.6f}")
        else:
            self.global_min = 0
            self.global_max = 1
            print("Warning: No valid pixel data found for global statistics")
        
        print("Pre-processing complete!")
    
    def _create_ui(self):
        """Create the UI components"""
        # Main container using grid layout
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid to expand properly
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # Image column
        main_frame.columnconfigure(1, weight=1)  # Info column
        main_frame.rowconfigure(0, weight=1)
        
        # Left side - Image frame
        self.image_frame = ttk.Frame(main_frame, padding=5, relief='groove', borderwidth=2)
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Canvas for displaying the cell image
        self.canvas = tk.Canvas(self.image_frame, width=600, height=600, background='black')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right side - Info and controls
        info_control_frame = ttk.Frame(main_frame, padding=5)
        info_control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        info_control_frame.columnconfigure(0, weight=1)
        
        # Info frame
        self.info_frame = ttk.LabelFrame(info_control_frame, text="Cell Information", padding=10)
        self.info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
        self.info_frame.columnconfigure(0, weight=1)
        
        # Cell info labels
        self.filename_label = ttk.Label(self.info_frame, text="Filename: ")
        self.filename_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.cell_id_label = ttk.Label(self.info_frame, text="Cell ID: ")
        self.cell_id_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.plot_label_label = ttk.Label(self.info_frame, text="Plot Label: ")
        self.plot_label_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.current_label_label = ttk.Label(self.info_frame, text="Current label: Not labeled")
        self.current_label_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # Model prediction frame (only shown if model is available)
        if self.model is not None:
            self.prediction_frame = ttk.LabelFrame(info_control_frame, text="Model Prediction", padding=10)
            self.prediction_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
            self.prediction_frame.columnconfigure(0, weight=1)
            
            self.prediction_label = ttk.Label(self.prediction_frame, text="Prediction: ")
            self.prediction_label.grid(row=0, column=0, sticky=tk.W, pady=2)
            
            self.certainty_label = ttk.Label(self.prediction_frame, text="Certainty: ")
            self.certainty_label.grid(row=1, column=0, sticky=tk.W, pady=2)
            
            self.uncertainty_rank_label = ttk.Label(self.prediction_frame, text="Uncertainty Rank: ")
            self.uncertainty_rank_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Labeling controls frame
        self.control_frame = ttk.LabelFrame(info_control_frame, text="Controls", padding=10)
        self.control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
        self.control_frame.columnconfigure(2, weight=1)
        
        # Labeling buttons
        self.btn_positive = ttk.Button(
            self.control_frame, 
            text='hdyst+ [+]', 
            command=lambda: self._label_current('hdyst+'),
            style='Positive.TButton'
        )
        self.btn_positive.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.btn_negative = ttk.Button(
            self.control_frame, 
            text='hdyst- [-]', 
            command=lambda: self._label_current('hdyst-'),
            style='Negative.TButton'
        )
        self.btn_negative.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.btn_skip = ttk.Button(
            self.control_frame, 
            text='Skip [s]', 
            command=lambda: self._label_current('skip')
        )
        self.btn_skip.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Navigation buttons
        self.btn_prev = ttk.Button(
            self.control_frame, 
            text='Previous [←]', 
            command=self._go_to_previous,
            state=tk.DISABLED
        )
        self.btn_prev.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.btn_next = ttk.Button(
            self.control_frame, 
            text='Next [→]', 
            command=self._go_to_next
        )
        self.btn_next.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.btn_save = ttk.Button(
            self.control_frame, 
            text='Save [Ctrl+S]', 
            command=self._save_labels
        )
        self.btn_save.grid(row=1, column=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Keyboard shortcuts info
        shortcuts_frame = ttk.LabelFrame(info_control_frame, text="Keyboard Shortcuts", padding=10)
        shortcuts_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        shortcuts_text = (
            "[+] Label as hdyst+\n"
            "[-] Label as hdyst-\n"
            "[s] Skip cell\n"
            "[←] Previous cell\n"
            "[→] Next cell\n"
            "[Ctrl+S] Save labels"
        )
        shortcuts_label = ttk.Label(shortcuts_frame, text=shortcuts_text)
        shortcuts_label.pack(anchor=tk.W)
        
        # Progress frame at the bottom
        progress_frame = ttk.Frame(self.root, padding=10)
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            progress_frame, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='determinate',
            maximum=len(self.sample_df)
        )
        self.progress.pack(fill=tk.X, pady=5)
        
        # Counter label
        self.counter_label = ttk.Label(
            progress_frame, 
            text=f"0/{len(self.sample_df)} cells labeled"
        )
        self.counter_label.pack(pady=5)
        
        # Define custom styles for buttons
        style = ttk.Style()
        style.configure('Positive.TButton', background='green')
        style.configure('Negative.TButton', background='red')
        
        # Bind keyboard shortcuts
        self.root.bind('<plus>', lambda e: self._label_current('hdyst+'))
        self.root.bind('<equal>', lambda e: self._label_current('hdyst+'))  # For keyboards where + is Shift+=
        self.root.bind('<minus>', lambda e: self._label_current('hdyst-'))
        self.root.bind('<s>', lambda e: self._label_current('skip'))
        self.root.bind('<Left>', lambda e: self._go_to_previous())
        self.root.bind('<Right>', lambda e: self._go_to_next())
        self.root.bind('<Control-s>', lambda e: self._save_labels())
        
        # Make sure the canvas has focus to receive keyboard events
        self.canvas.focus_set()
    
    def _update_display(self):
        """Update the display with the current cell"""
        if self.current_index < len(self.sample_df):
            # Get current cell data
            row = self.sample_df.iloc[self.current_index]
            
            # Update info labels
            self.filename_label.config(text=f"Filename: {row['filename']}")
            self.cell_id_label.config(text=f"Cell ID: {row['cell_id']}")
            self.plot_label_label.config(text=f"Plot Label: {row['plotLabel']}")
            
            label = self.labels.get(self.current_index, 'Not labeled')
            self.current_label_label.config(text=f"Current label: {label}")
            
            # Update prediction info if model is available
            if self.model is not None and self.current_index in self.processed_images:
                img_features = self.processed_images[self.current_index].flatten().reshape(1, -1)
                probas = self.model.predict_proba(img_features)[0]
                pred_class = "hdyst+" if probas[1] > 0.5 else "hdyst-"
                certainty = max(probas)
                
                self.prediction_label.config(text=f"Prediction: {pred_class} ({certainty:.2f})")
                self.certainty_label.config(text=f"Certainty: {certainty:.2f}")
                
                if self.uncertainty_scores is not None:
                    key = (row['filename'], row['cell_id'])
                    if key in self.uncertainty_scores:
                        uncertainty = self.uncertainty_scores[key]
                        self.uncertainty_rank_label.config(text=f"Uncertainty Score: {uncertainty:.4f}")
            
            # Update image if available
            if self.current_index in self.processed_images:
                # Get the processed image (red channel only, float32)
                img = self.processed_images[self.current_index].copy()
                
                # Apply 12x scaling factor for visualization
                img = img * 12.0
                
                # Apply global normalization using the calculated 1st and 99th percentiles
                img_scaled = np.zeros_like(img)
                mask = img > 0.001  # Preserve true black background
                
                # Normalize using global min/max
                if self.global_max > self.global_min:
                    img_scaled[mask] = np.clip((img[mask] - self.global_min) / (self.global_max - self.global_min), 0, 1)
                
                # Convert to 8-bit for display
                img_8bit = (img_scaled * 255).astype(np.uint8)
                
                # Create a colored version with red channel only
                colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                colored_img[:, :, 0] = img_8bit  # Red channel
                
                # Check if image has valid dimensions
                if img.shape[0] <= 0 or img.shape[1] <= 0:
                    print(f"Warning: Image has invalid dimensions: {img.shape}")
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        300, 300,
                        text="Invalid image dimensions",
                        fill="red",
                        font=("Helvetica", 16)
                    )
                    return
                
                # Convert to PIL Image
                try:
                    pil_img = Image.fromarray(colored_img, mode='RGB')
                    
                    # Resize to fit canvas if needed
                    # Make sure we have valid canvas dimensions
                    canvas_width = max(self.canvas.winfo_width() or 600, 1)
                    canvas_height = max(self.canvas.winfo_height() or 600, 1)
                    
                    # Ensure we don't divide by zero
                    if img.shape[0] <= 0 or img.shape[1] <= 0:
                        print(f"Warning: Invalid image shape: {img.shape}")
                        scale = 1.0
                    else:
                        scale = min(canvas_width / img.shape[1], canvas_height / img.shape[0])
                    
                    # Ensure dimensions are at least 1 pixel
                    new_width = max(int(img.shape[1] * scale), 1)
                    new_height = max(int(img.shape[0] * scale), 1)
                    
                    # Resize image with error handling
                    try:
                        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Convert to PhotoImage
                        self.tk_img = ImageTk.PhotoImage(pil_img)
                        
                        # Display on canvas
                        self.canvas.delete("all")
                        self.canvas.create_image(
                            canvas_width // 2,
                            canvas_height // 2,
                            image=self.tk_img,
                            anchor=tk.CENTER
                        )
                        
                        # Add title
                        self.canvas.create_text(
                            canvas_width // 2,
                            20,
                            text=f"Cell {self.current_index+1}/{len(self.sample_df)}",
                            fill="white",
                            font=("Helvetica", 16)
                        )
                    except Exception as e:
                        print(f"Error resizing image: {e}")
                        self.canvas.delete("all")
                        self.canvas.create_text(
                            300, 300,
                            text=f"Error displaying image: {e}",
                            fill="red",
                            font=("Helvetica", 16)
                        )
                        
                except Exception as e:
                    print(f"Error creating image: {e}")
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        300, 300,
                        text=f"Error creating image: {e}",
                        fill="red",
                        font=("Helvetica", 16)
                    )
                    
            else:
                # Clear canvas and show error message
                self.canvas.delete("all")
                self.canvas.create_text(
                    300, 300,
                    text="Image not available",
                    fill="red",
                    font=("Helvetica", 16)
                )
    
    def _label_current(self, label):
        """Label the current cell"""
        if self.current_index < len(self.sample_df):
            self.labels[self.current_index] = label
            
            # Update progress
            labeled_count = self.labels.count()
            self.progress['value'] = labeled_count
            self.counter_label.config(text=f"{labeled_count}/{len(self.sample_df)} cells labeled")
            
            # Move to next cell
            self._go_to_next()
    
    def _go_to_previous(self):
        """Go to the previous cell"""
        if self.current_index > 0:
            self.current_index -= 1
            self.btn_next['state'] = tk.NORMAL
            if self.current_index == 0:
                self.btn_prev['state'] = tk.DISABLED
            self._update_display()
    
    def _go_to_next(self):
        """Go to the next cell"""
        if self.current_index < len(self.sample_df) - 1:
            self.current_index += 1
            self.btn_prev['state'] = tk.NORMAL
            if self.current_index == len(self.sample_df) - 1:
                self.btn_next['state'] = tk.DISABLED
            self._update_display()
    
    def _save_labels(self):
        """Save the current labels and return the labeled data"""
        # Create a DataFrame with the labels
        labeled_df = self.sample_df.copy()
        labeled_df['hdyst_label'] = self.labels
        
        # Filter out unlabeled cells
        labeled_df = labeled_df[~labeled_df['hdyst_label'].isna()]
        
        # Show label statistics
        label_counts = labeled_df['hdyst_label'].value_counts()
        stats_text = "Label statistics:\n"
        
        for label, count in label_counts.items():
            stats_text += f"{label}: {count} cells\n"
        
        # Show message box with statistics
        tk.messagebox.showinfo("Labels Saved", stats_text)
        
        # Also print to console
        print(stats_text)
        
        # Return labeled data for active learning
        return labeled_df
    
    def get_labeled_data(self):
        """Return the labeled data as a DataFrame"""
        labeled_df = self.sample_df.copy()
        labeled_df['hdyst_label'] = self.labels
        
        # Filter out unlabeled cells
        labeled_df = labeled_df[~labeled_df['hdyst_label'].isna()]
        
        return labeled_df


# ------ Active Learning Components ------

class ActiveLearningManager:
    def __init__(self, df, cropped_cells_dir, output_dir=None, pad_size=None, random_state=42):
        """
        Manage the active learning process for hdyst+/hdyst- cell classification
        
        Parameters:
        -----------
        df : str or pandas DataFrame
            Path to CSV file or DataFrame containing cell data
        cropped_cells_dir : str
            Directory containing cropped cell pickle files
        output_dir : str, optional
            Directory to save output files
        pad_size : int, optional
            Padding size for cell images. If None, will be determined automatically from the entire dataset.
        random_state : int
            Random seed for reproducibility
        """
        # Initialize attributes
        self.cropped_cells_dir = cropped_cells_dir
        self.random_state = random_state
        self.output_dir = output_dir or 'active_learning_output'
        self.pad_size = pad_size
        self.clf = None
        self.labeled_df = pd.DataFrame()
        self.unlabeled_df = None
        self.cell_data_dict = None
        self.active_learning_round = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load the data
        if isinstance(df, str):
            print(f"Loading dataset from {df}...")
            self.df = pd.read_csv(df)
        else:
            self.df = df.copy()
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        
        # Filter the dataset
        cols_to_check = ['filename', 'cell_id', 'feret_diameter', 'cytoplasm_nuclei_intensity', 'plotLabel']
        self.df = self.df.dropna(subset=cols_to_check).copy()
        print(f"Filtered dataset shape: {self.df.shape}")
        
        # Ensure all cells have unique (filename, cell_id) pairs
        self.df = self.df.drop_duplicates(subset=['filename', 'cell_id'])
        
        # Set all cells as unlabeled initially
        self.unlabeled_df = self.df.copy()
        
        # Determine pad size if not provided
        if self.pad_size is None:
            print(f"Determining pad size from the entire dataset (using 99.9th percentile)...")
            self.pad_size = determine_pad_size(self.df, folder=cropped_cells_dir, 
                                         percentile=99.9, use_all_cells=True,
                                         random_state=random_state)
    
    def check_for_previous_session(self):
        """
        Check if there's a previous session that can be resumed
        
        Returns:
        --------
        tuple : (bool, int) - Whether a previous session exists and the last round completed
        """
        # Look for labeled data files
        last_round = 0
        labeled_files = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            return False, 0
        
        for filename in os.listdir(self.output_dir):
            if filename.startswith("labeled_data_round_") and filename.endswith(".csv"):
                try:
                    round_num = int(filename.split("_round_")[1].split(".")[0])
                    labeled_files.append((round_num, filename))
                    if round_num > last_round:
                        last_round = round_num
                except:
                    continue
        
        # Check if we found any labeled data files
        if not labeled_files:
            return False, 0
        
        # Check if the corresponding model file exists
        model_path = os.path.join(self.output_dir, f"rf_classifier_round_{last_round}.pkl")
        if not os.path.exists(model_path):
            # If model doesn't exist, use the latest round with both data and model
            valid_rounds = []
            for round_num, _ in labeled_files:
                model_path = os.path.join(self.output_dir, f"rf_classifier_round_{round_num}.pkl")
                if os.path.exists(model_path):
                    valid_rounds.append(round_num)
            
            if valid_rounds:
                last_round = max(valid_rounds)
            else:
                return False, 0
        
        return True, last_round

    def resume_from_previous_session(self, round_num):
        """
        Resume active learning from a previous session, but train a new classifier
        on the latest labeled data instead of loading a saved model.
        
        Parameters:
        -----------
        round_num : int
            The round number to resume from
            
        Returns:
        --------
        bool : Whether the resume was successful
        """
        # Load labeled data
        labeled_path = os.path.join(self.output_dir, f"labeled_data_round_{round_num}.csv")
        if not os.path.exists(labeled_path):
            print(f"Error: Labeled data file {labeled_path} not found.")
            return False
        
        print(f"Loading labeled data from {labeled_path}...")
        try:
            self.labeled_df = pd.read_csv(labeled_path)
            print(f"Loaded {len(self.labeled_df)} labeled cells.")
        except Exception as e:
            print(f"Error loading labeled data: {e}")
            return False
        
        # Update active learning round
        self.active_learning_round = round_num
        
        # Update unlabeled dataset
        labeled_keys = set(zip(self.labeled_df['filename'], self.labeled_df['cell_id']))
        mask = ~self.df.apply(
            lambda row: (row['filename'], row['cell_id']) in labeled_keys, 
            axis=1
        )
        self.unlabeled_df = self.df[mask].reset_index(drop=True)
        
        print(f"Session resumed from round {round_num}.")
        print(f"Available cells for labeling: {len(self.unlabeled_df)}")
        
        # Train new classifier on loaded labeled data instead of loading saved model
        print("\nTraining a new classifier on the loaded labeled data...")
        self.train_classifier()
        
        return True
    
    def load_cell_data(self, df=None):
        """
        Load cell data for the specified DataFrame, adding to existing data
        
        Parameters:
        -----------
        df : pandas DataFrame, optional
            DataFrame containing cells to load. If None, uses all unlabeled cells.
            
        Returns:
        --------
        dict : Dictionary of cell data
        """
        df_to_load = df if df is not None else self.unlabeled_df
        print(f"Loading cell data for {len(df_to_load)} cells...")
        
        # Initialize dictionary if needed
        if self.cell_data_dict is None:
            self.cell_data_dict = {}
            
        # Count cells before loading
        cells_before = len(self.cell_data_dict)
        
        # Load new data and add to existing dictionary
        new_data = load_cell_data_dict(df_to_load, folder=self.cropped_cells_dir)
        self.cell_data_dict.update(new_data)
        
        # Report results
        cells_added = len(self.cell_data_dict) - cells_before
        print(f"Added {cells_added} new cells to dictionary (total: {len(self.cell_data_dict)})")
        
        return self.cell_data_dict
    
    def start_labeling_round(self, batch_size=100, mode='random', tkinter_process=False):
        """
        Start a new labeling round with improved balanced sampling
        
        Parameters:
        -----------
        batch_size : int
            Number of cells to label in this round
        mode : str
            Sampling mode: 'random', 'uncertain', or 'balanced'
        tkinter_process : bool
            Whether to run the Tkinter interface in a separate process
            
        Returns:
        --------
        pandas DataFrame : Newly labeled cells
        """
        self.active_learning_round += 1
        print(f"\n=== Starting Active Learning Round {self.active_learning_round} ===")
        
        # Load all cell data at the beginning of each round to ensure we have everything
        if self.cell_data_dict is None or len(self.cell_data_dict) < len(self.df) * 0.5:
            print("Loading cell data for all cells to ensure complete coverage...")
            self.load_cell_data(self.df)  # Load ALL cells, not just current batch
        
        # Select cells for this round
        if mode == 'random' or self.clf is None:
            # First round or explicitly random sampling
            cells_to_label = self.select_cells_random(batch_size)
            print(f"Selected {len(cells_to_label)} random cells for labeling")
        elif mode == 'uncertain':
            # Select most uncertain cells
            cells_to_label = self.select_cells_uncertain(batch_size)
            print(f"Selected {len(cells_to_label)} most uncertain cells for labeling")
        elif mode == 'balanced':
            # Select a mix of random and uncertain cells - with fixes
            n_uncertain = batch_size // 2
            n_random = batch_size - n_uncertain
            
            print(f"Selecting {n_uncertain} uncertain cells...")
            uncertain_cells = self.select_cells_uncertain(n_uncertain)
            print(f"Got {len(uncertain_cells)} uncertain cells")
            
            # Get the keys of uncertain cells to avoid duplicates
            uncertain_keys = set(zip(uncertain_cells['filename'], uncertain_cells['cell_id']))
            
            # Filter out uncertain cells from potential random selection
            remaining_mask = ~self.unlabeled_df.apply(
                lambda row: (row['filename'], row['cell_id']) in uncertain_keys, 
                axis=1
            )
            remaining_df = self.unlabeled_df[remaining_mask]
            
            # If we have enough cells left, sample from remaining cells
            if len(remaining_df) >= n_random:
                print(f"Selecting {n_random} random cells from remaining {len(remaining_df)} cells...")
                random_cells = remaining_df.sample(n=n_random, random_state=self.random_state)
            else:
                # Otherwise just take all remaining cells
                print(f"Only {len(remaining_df)} cells remain, taking all of them")
                random_cells = remaining_df
            
            # Reset indices before concatenation to avoid duplicates
            uncertain_cells.reset_index(drop=True, inplace=True)
            random_cells.reset_index(drop=True, inplace=True)
            
            # Combine with explicit ignore_index to avoid DataFrame merge issues
            cells_to_label = pd.concat([uncertain_cells, random_cells], ignore_index=True)
            
            print(f"Final selection: {len(cells_to_label)} cells ({len(uncertain_cells)} uncertain + {len(random_cells)} random)")
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        
        # Ensure we have cell data for the selected cells
        for idx, row in tqdm(cells_to_label.iterrows(), total=len(cells_to_label), desc="Verifying cell data"):
            key = (row['filename'], row['cell_id'])
            if key not in self.cell_data_dict:
                print(f"Cell data missing for {key}, loading...")
                # Create a mini dataframe just for this cell
                cell_df = pd.DataFrame([row])
                self.load_cell_data(cell_df)
        
        # Get uncertainty scores if we have a model
        uncertainty_scores = None
        if self.clf is not None:
            uncertainty_scores = self.calculate_uncertainty(cells_to_label)
        
        # Start the labeling interface
        if tkinter_process:
            # For running in a separate process (useful in notebooks)
            return self._run_labeling_in_process(cells_to_label, uncertainty_scores)
        else:
            # For running in the same process (normal CLI usage)
            return self._run_labeling(cells_to_label, uncertainty_scores)
    
    def _run_labeling(self, cells_to_label, uncertainty_scores=None):
        """Run the labeling interface in the current process"""
        root = tk.Tk()
        labeler = CellLabelerGUI(root, cells_to_label, self.cell_data_dict, 
                                self.pad_size, self.clf, uncertainty_scores)
        root.mainloop()
        
        # Get labeled data
        new_labeled_df = labeler.get_labeled_data()
        
        # Update labeled and unlabeled sets
        self._update_datasets(new_labeled_df)
        
        # Train classifier on updated labeled data
        if len(self.labeled_df) > 0:
            self.train_classifier()
        
        return new_labeled_df
    
    def _run_labeling_in_process(self, cells_to_label, uncertainty_scores=None):
        """Run the labeling interface in a separate process (for notebooks)"""
        # Create a Queue to get results back from the process
        from multiprocessing import Process, Queue
        
        result_queue = Queue()
        
        def run_labeler(queue):
            root = tk.Tk()
            labeler = CellLabelerGUI(root, cells_to_label, self.cell_data_dict, 
                                    self.pad_size, self.clf, uncertainty_scores)
            
            def on_closing():
                queue.put(labeler.get_labeled_data())
                root.destroy()
            
            root.protocol("WM_DELETE_WINDOW", on_closing)
            root.mainloop()
        
        process = Process(target=run_labeler, args=(result_queue,))
        process.start()
        process.join()  # Wait for the process to complete
        
        # Get labeled data from the queue
        if not result_queue.empty():
            new_labeled_df = result_queue.get()
            
            # Update labeled and unlabeled sets
            self._update_datasets(new_labeled_df)
            
            # Train classifier on updated labeled data
            if len(self.labeled_df) > 0:
                self.train_classifier()
            
            return new_labeled_df
        else:
            print("No labeled data received from the labeling process.")
            return pd.DataFrame()
    
    def _update_datasets(self, new_labeled_df):
        """Update labeled and unlabeled datasets with newly labeled data"""
        # Add to labeled dataset
        self.labeled_df = pd.concat([self.labeled_df, new_labeled_df]).drop_duplicates(
            subset=['filename', 'cell_id']
        ).reset_index(drop=True)
        
        # Remove from unlabeled dataset
        labeled_keys = set(zip(new_labeled_df['filename'], new_labeled_df['cell_id']))
        mask = ~self.unlabeled_df.apply(
            lambda row: (row['filename'], row['cell_id']) in labeled_keys, 
            axis=1
        )
        self.unlabeled_df = self.unlabeled_df[mask].reset_index(drop=True)
        
        print(f"Updated datasets - Labeled: {len(self.labeled_df)}, Unlabeled: {len(self.unlabeled_df)}")
    
    def select_cells_random(self, n_samples):
        """Randomly select cells for labeling"""
        if len(self.unlabeled_df) <= n_samples:
            return self.unlabeled_df.copy()
        
        # Stratified sampling by plotLabel
        plot_labels = self.unlabeled_df['plotLabel'].unique()
        n_per_label = n_samples // len(plot_labels)
        
        selected_cells = []
        for label in plot_labels:
            subset = self.unlabeled_df[self.unlabeled_df['plotLabel'] == label]
            n_select = min(n_per_label, len(subset))
            if n_select > 0:
                selected = subset.sample(n=n_select, random_state=self.random_state)
                selected_cells.append(selected)
        
        # Combine and shuffle
        combined = pd.concat(selected_cells)
        
        # If we need more samples to reach n_samples, take them randomly
        remaining = n_samples - len(combined)
        if remaining > 0:
            remaining_pool = self.unlabeled_df[~self.unlabeled_df.index.isin(combined.index)]
            if len(remaining_pool) > 0:
                remaining_samples = remaining_pool.sample(
                    n=min(remaining, len(remaining_pool)), 
                    random_state=self.random_state
                )
                combined = pd.concat([combined, remaining_samples])
        
        return combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def select_cells_uncertain(self, n_samples):
        """Select cells with highest uncertainty for labeling, randomizing ties"""
        if self.clf is None:
            print("Warning: No classifier available. Using random selection instead.")
            return self.select_cells_random(n_samples)
        
        if len(self.unlabeled_df) <= n_samples:
            return self.unlabeled_df.copy()
        
        # Calculate uncertainty for all unlabeled cells
        uncertainty_scores = self.calculate_uncertainty(self.unlabeled_df)
        
        # Create a DataFrame with uncertainty scores
        uncertainty_df = pd.DataFrame({
            'filename': [k[0] for k in uncertainty_scores.keys()],
            'cell_id': [k[1] for k in uncertainty_scores.keys()],
            'uncertainty': list(uncertainty_scores.values())
        })
        
        # Merge with unlabeled_df
        merged_df = self.unlabeled_df.merge(
            uncertainty_df, 
            on=['filename', 'cell_id'],
            how='inner'
        )
        
        # Group by uncertainty value (rounded to handle floating point precision)
        merged_df['uncertainty_group'] = merged_df['uncertainty'].apply(lambda x: round(x, 6))
        
        # Sort groups by uncertainty (highest first)
        groups = merged_df.groupby('uncertainty_group')
        sorted_groups = sorted(groups, key=lambda x: x[0], reverse=True)
        
        # Take cells from each group in order, randomizing within groups
        selected_cells = []
        cells_needed = n_samples
        
        for uncertainty_val, group_df in sorted_groups:
            group_size = len(group_df)
            
            if group_size <= cells_needed:
                # Take all cells in this group, but in random order
                selected_cells.append(group_df.sample(frac=1, random_state=self.random_state))
                cells_needed -= group_size
            else:
                # Take a random subset of cells from this group
                selected_cells.append(group_df.sample(n=cells_needed, random_state=self.random_state))
                cells_needed = 0
                
            if cells_needed == 0:
                break
        
        # Combine selected cells and drop the temporary columns
        result = pd.concat(selected_cells).reset_index(drop=True)
        
        return result.drop(columns=['uncertainty', 'uncertainty_group']).reset_index(drop=True)
        
    def calculate_uncertainty(self, df):
        """
        Calculate uncertainty scores for cells in the DataFrame using batch processing
        
        Returns:
        --------
        dict : Dictionary mapping (filename, cell_id) to uncertainty score
        """
        if self.clf is None:
            return {}
        
        # Process all cells in batch mode
        all_features = []
        valid_cells = []
        
        # First collect all valid cells and their features
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cells for uncertainty"):
            key = (row['filename'], row['cell_id'])
            if key in self.cell_data_dict:
                cell_data = self.cell_data_dict[key]
                processed_img = process_cell_red(cell_data, self.pad_size)
                features = processed_img.flatten()
                all_features.append(features)
                valid_cells.append(key)
        
        uncertainty_scores = {}
        
        if len(all_features) > 0:
            # Convert to numpy array and make ONE predict_proba call for all cells
            X = np.array(all_features)
            print(f"Calculating uncertainty for {len(X)} cells in batch...")
            
            try:
                # Get all probabilities in one call
                all_probas = self.clf.predict_proba(X)
                
                # Calculate uncertainty as 1 - max probability for each cell
                uncertainties = 1 - np.max(all_probas, axis=1)
                
                # Map back to cell keys
                for i, key in enumerate(valid_cells):
                    uncertainty_scores[key] = uncertainties[i]
                    
            except Exception as e:
                print(f"Error calculating uncertainties: {e}")
        
        return uncertainty_scores
    
    def train_classifier(self, n_estimators=100, oob_score=True):
        """
        Train a Random Forest classifier on the labeled data
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        oob_score : bool
            Whether to use out-of-bag samples to estimate accuracy
            
        Returns:
        --------
        RandomForestClassifier : Trained classifier
        """
        # Skip if no labeled data
        if len(self.labeled_df) == 0:
            print("No labeled data available for training.")
            return None
        
        # Filter out skipped cells
        train_df = self.labeled_df[self.labeled_df['hdyst_label'] != 'skip'].copy()
        
        if len(train_df) < 2:
            print("Not enough labeled data for training (at least 2 cells required).")
            return None
        
        # Check if we have at least two classes
        n_classes = train_df['hdyst_label'].nunique()
        if n_classes < 2:
            print(f"Only {n_classes} class found in labeled data. At least 2 classes required for training.")
            return None
        
        print(f"Training Random Forest classifier on {len(train_df)} labeled cells...")
        
        # Prepare features and labels
        X = []
        y = []
        
        # Process cells batch by batch to create features
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
            key = (row['filename'], row['cell_id'])
            if key in self.cell_data_dict:
                cell_data = self.cell_data_dict[key]
                processed_img = process_cell_red(cell_data, self.pad_size)
                features = processed_img.flatten()
                X.append(features)
                
                # Convert label to binary (1 for hdyst+, 0 for hdyst-)
                label = 1 if row['hdyst_label'] == 'hdyst+' else 0
                y.append(label)
            else:
                print(f"Warning: Missing cell data for {key}, skipping this cell for training")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have enough data
        if len(X) < 2:
            print("Not enough processed cells for training.")
            return None
        
        # Create and train the classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            oob_score=oob_score,
            n_jobs=-1,  # Use all cores
            verbose=1
        )
        
        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, target_names=['hdyst-', 'hdyst+']))
        
        # Store the classifier
        self.clf = clf
        
        # Save the classifier to file
        model_path = os.path.join(self.output_dir, f"rf_classifier_round_{self.active_learning_round}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Classifier saved to {model_path}")
        
        return clf
    
    def evaluate_on_all_data(self):
        """Evaluate the current classifier on all data with batch processing"""
        if self.clf is None:
            print("No trained classifier available.")
            return
        
        # Get unique treatment groups
        plot_labels = self.df['plotLabel'].unique()
        
        # Sample from each group
        n_samples_per_group = 100  # Limit to 100 cells per group for efficiency
        sample_dfs = []
        
        for label in plot_labels:
            group_df = self.df[self.df['plotLabel'] == label]
            if len(group_df) > n_samples_per_group:
                group_sample = group_df.sample(n=n_samples_per_group, random_state=self.random_state)
            else:
                group_sample = group_df
            sample_dfs.append(group_sample)
        
        # Combine samples
        eval_df = pd.concat(sample_dfs).reset_index(drop=True)
        print(f"Evaluating on {len(eval_df)} cells from {len(plot_labels)} treatment groups")
        
        # Process all cells in batch mode
        all_features = []
        valid_indices = []
        
        # First collect all valid cells and their features
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing cell images"):
            key = (row['filename'], row['cell_id'])
            if key in self.cell_data_dict:
                cell_data = self.cell_data_dict[key]
                processed_img = process_cell_red(cell_data, self.pad_size)
                features = processed_img.flatten()
                all_features.append(features)
                valid_indices.append(idx)
        
        # Get data for valid cells only
        valid_df = eval_df.iloc[valid_indices].reset_index(drop=True)
        
        if len(all_features) == 0:
            print("No valid cells found for evaluation.")
            return
        
        # Convert to numpy array and make ONE prediction call for all cells
        X = np.array(all_features)
        print(f"Making batch prediction on {len(X)} cells...")
        predictions_binary = self.clf.predict(X)
        
        # Convert binary predictions to 'hdyst+' and 'hdyst-' labels
        predictions = ['hdyst+' if pred == 1 else 'hdyst-' for pred in predictions_binary]
        
        # Create results DataFrame
        results_df = valid_df.copy()
        results_df['hdyst_prediction'] = predictions
        
        # Calculate prediction percentages by plot label
        results_by_label = pd.crosstab(
            results_df['plotLabel'], 
            results_df['hdyst_prediction'], 
            normalize='index'
        ) * 100
        
        print("\nhdyst+/hdyst- prediction distribution by treatment group:")
        print(results_by_label)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        results_by_label.plot(kind='bar')
        plt.title('hdyst+/hdyst- Distribution by Treatment Group')
        plt.ylabel('Percentage')
        plt.xlabel('Treatment Group')
        plt.xticks(rotation=45)
        plt.legend(title='hdyst Prediction')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"prediction_distribution_round_{self.active_learning_round}.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")
        
        # Save the evaluation results
        results_path = os.path.join(self.output_dir, f"evaluation_results_round_{self.active_learning_round}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Evaluation results saved to {results_path}")
        
        return results_df
    
    def save_labeled_data(self):
        """Save the current labeled dataset to a CSV file"""
        if len(self.labeled_df) == 0:
            print("No labeled data available.")
            return
        
        # Save labeled data
        labeled_path = os.path.join(self.output_dir, f"labeled_data_round_{self.active_learning_round}.csv")
        self.labeled_df.to_csv(labeled_path, index=False)
        print(f"Labeled data saved to {labeled_path}")
        
        # Display label distribution
        print("\nLabel distribution:")
        print(self.labeled_df['hdyst_label'].value_counts())
        
        # Display label distribution by plot label
        print("\nLabel distribution by plot label:")
        dist_by_plot = pd.crosstab(
            self.labeled_df['plotLabel'], 
            self.labeled_df['hdyst_label'], 
            normalize='index'
        ) * 100
        print(dist_by_plot)
        
        return labeled_path
    
    def run_active_learning_workflow(self, initial_samples=100, batch_size=50, 
                                    max_rounds=5, sampling_mode='balanced'):
        """
        Run the complete active learning workflow
        
        Parameters:
        -----------
        initial_samples : int
            Number of samples for initial random labeling
        batch_size : int
            Number of samples for each subsequent labeling round
        max_rounds : int
            Maximum number of active learning rounds
        sampling_mode : str
            Sampling mode for subsequent rounds: 'random', 'uncertain', or 'balanced'
            
        Returns:
        --------
        RandomForestClassifier : Final trained classifier
        """
        # Step 1: Initial random sampling
        print(f"=== Active Learning Workflow ===")
        print(f"Initial random sampling: {initial_samples} cells")
        print(f"Batch size for subsequent rounds: {batch_size} cells")
        print(f"Maximum number of rounds: {max_rounds}")
        print(f"Sampling mode: {sampling_mode}")
        
        # Load cell data for the entire dataset to avoid missing data
        print("Loading all cell data upfront to ensure complete coverage...")
        self.load_cell_data(self.df)
        
        # Step 2: Initial labeling round with random sampling
        self.start_labeling_round(batch_size=initial_samples, mode='random')
        self.save_labeled_data()
        
        # Step 3: Active learning loop
        for i in range(1, max_rounds):
            print(f"\n=== Active Learning Round {i+1}/{max_rounds} ===")
            
            if len(self.unlabeled_df) == 0:
                print("No more unlabeled cells available. Stopping.")
                break
            
            # Select cells and run labeling
            self.start_labeling_round(batch_size=batch_size, mode=sampling_mode)
            
            # Save results
            self.save_labeled_data()
            
            # Evaluate on all data
            self.evaluate_on_all_data()
        
        print("\n=== Active Learning Workflow Complete ===")
        print(f"Total labeled cells: {len(self.labeled_df)}")
        print(f"Final model saved in {self.output_dir}")
        
        return self.clf


# ------ Main Function ------

def main():
    """Main function to run the active learning workflow"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Active Learning for hdyst+/hdyst- Cell Classification')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the abe_df CSV file')
    parser.add_argument('--cropped_cells_dir', type=str, default='cropped_cells', help='Directory containing cropped cells pickle files')
    parser.add_argument('--output_dir', type=str, default='active_learning_output', help='Directory to save output files')
    parser.add_argument('--pad_size', type=int, help='Fixed padding size for cell images. If not provided, will be determined from the entire dataset.')
    parser.add_argument('--initial_samples', type=int, default=100, help='Number of initial random samples')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of cells to label in each subsequent round')
    parser.add_argument('--max_rounds', type=int, default=5, help='Maximum number of active learning rounds')
    parser.add_argument('--sampling_mode', type=str, default='balanced', choices=['random', 'uncertain', 'balanced'], 
                      help='Sampling mode for subsequent labeling rounds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode with step-by-step prompts')
    parser.add_argument('--resume', action='store_true', help='Try to resume from a previous session if available')
    parser.add_argument('--no_resume', action='store_true', help='Do not attempt to resume even if previous session exists')
    parser.add_argument('--preload_all', action='store_true', help='Preload all cell data at startup (slower startup, faster rounds)')
    parser.add_argument('--retrain', action='store_true', help='Always retrain the classifier on the loaded labeled data, even in resume mode')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return
    
    # Initialize active learning manager
    manager = ActiveLearningManager(
        df=args.csv_file,
        cropped_cells_dir=args.cropped_cells_dir,
        output_dir=args.output_dir,
        pad_size=args.pad_size,
        random_state=args.random_state
    )
    
    # Preload all cell data if requested
    if args.preload_all:
        print("Preloading all cell data...")
        manager.load_cell_data(manager.df)
    
    # Find the latest labeled data if any exists
    latest_round = 0
    labeled_files = []
    
    if os.path.exists(manager.output_dir):
        for filename in os.listdir(manager.output_dir):
            if filename.startswith("labeled_data_round_") and filename.endswith(".csv"):
                try:
                    round_num = int(filename.split("_round_")[1].split(".")[0])
                    labeled_files.append((round_num, filename))
                    if round_num > latest_round:
                        latest_round = round_num
                except:
                    continue
    
    # Determine if we should resume, load existing data, or start fresh
    resume_session = False
    load_existing_data = False
    
    if labeled_files and not args.no_resume:
        if args.resume:
            resume_session = True
            print(f"\nFound previous session (completed {latest_round} rounds). Will resume from there.")
        else:
            print(f"\nFound existing labeled data from round {latest_round}.")
            choice = input("Would you like to use this data as starting point? (y/n): ").strip().lower()
            if choice == 'y':
                load_existing_data = True
    
    # Load labeled data and update manager state
    if resume_session or load_existing_data:
        labeled_path = os.path.join(manager.output_dir, f"labeled_data_round_{latest_round}.csv")
        
        try:
            print(f"Loading labeled data from {labeled_path}...")
            manager.labeled_df = pd.read_csv(labeled_path)
            print(f"Loaded {len(manager.labeled_df)} labeled cells.")
            
            # Update unlabeled dataset
            labeled_keys = set(zip(manager.labeled_df['filename'], manager.labeled_df['cell_id']))
            mask = ~manager.df.apply(
                lambda row: (row['filename'], row['cell_id']) in labeled_keys, 
                axis=1
            )
            manager.unlabeled_df = manager.df[mask].reset_index(drop=True)
            
            # Set the active learning round
            if resume_session:
                manager.active_learning_round = latest_round
            
            # CRITICAL: Load the cell data for all labeled cells before training
            print("Loading image data for all labeled cells...")
            manager.load_cell_data(manager.labeled_df)
            
            # Train a new classifier on loaded labeled data
            print("\nTraining a new classifier on the loaded labeled data...")
            manager.train_classifier()
            
            print(f"Available cells for labeling: {len(manager.unlabeled_df)}")
            
        except Exception as e:
            print(f"Error loading or processing labeled data: {e}")
            print("Starting fresh with no labeled data.")
            manager.labeled_df = pd.DataFrame()
            manager.unlabeled_df = manager.df.copy()
            manager.active_learning_round = 0
            resume_session = False
            load_existing_data = False
    
    # Run in interactive or automatic mode
    if args.interactive:
        # Interactive mode - step through the process
        step = 1
        if resume_session:
            # Skip to the next round after the last completed one
            step = 3
        elif load_existing_data:
            # Skip initial labeling but start with round 1, not the loaded round
            step = 3
            manager.active_learning_round = 0
        
        continue_process = True
        
        while continue_process:
            print(f"\n=== Step {step} ===")
            
            if step == 1:
                print("\nInitializing the active learning process...")
                print(f"Dataset has {len(manager.df)} cells")
                print(f"Pad size determined: {manager.pad_size}")
                
                # Load cell data if not already loaded
                if not args.preload_all:
                    print("\nLoading cell data (this may take a while)...")
                    manager.load_cell_data()
                
                input("\nPress Enter to continue to the initial labeling round...")
                step += 1
            
            elif step == 2:
                print("\nStarting initial labeling round with random sampling...")
                print(f"Number of cells to label: {args.initial_samples}")
                
                # Start initial labeling round
                manager.start_labeling_round(batch_size=args.initial_samples, mode='random')
                manager.save_labeled_data()
                
                if len(manager.labeled_df) == 0:
                    print("\nNo cells were labeled. Exiting...")
                    return
                
                # Train initial classifier
                print("\nTraining initial classifier...")
                manager.train_classifier()
                
                input("\nPress Enter to continue to the active learning rounds...")
                step += 1
            
            elif step > 2:
                active_round = step - 2
                next_round = manager.active_learning_round + 1
                
                if resume_session and active_round <= manager.active_learning_round:
                    # Skip rounds that were already completed in the previous session
                    step += 1
                    continue
                
                print(f"\nStarting active learning round {next_round}")
                print(f"Sampling mode: {args.sampling_mode}")
                print(f"Number of cells to label: {args.batch_size}")
                
                # Start next labeling round
                manager.start_labeling_round(batch_size=args.batch_size, mode=args.sampling_mode)
                manager.save_labeled_data()
                
                # Evaluate on all data
                manager.evaluate_on_all_data()
                
                if active_round >= args.max_rounds:
                    print("\nMaximum number of rounds reached.")
                    continue_process = False
                else:
                    choice = input("\nContinue to the next round? (y/n): ").strip().lower()
                    if choice != 'y':
                        continue_process = False
                    else:
                        step += 1
        
        print("\n=== Active Learning Process Complete ===")
        print(f"Total labeled cells: {len(manager.labeled_df)}")
        print(f"Results saved in {args.output_dir}")
        
    else:
        # Non-interactive mode - run the complete workflow
        if resume_session:
            # Start from the next round after the last completed one
            next_round = manager.active_learning_round + 1
            remaining_rounds = args.max_rounds - next_round + 1
            
            if remaining_rounds <= 0:
                print(f"All {args.max_rounds} rounds have already been completed.")
                
                # Always retrain the final model on the complete labeled dataset
                print("\nRetraining the final model on all labeled data...")
                manager.train_classifier()
                
                print("Final model saved. Nothing more to do.")
                return
                
            print(f"Resuming workflow from round {next_round} ({remaining_rounds} rounds remaining)")
            
            # Continue with active learning loop for remaining rounds
            for i in range(next_round, args.max_rounds + 1):
                print(f"\n=== Active Learning Round {i}/{args.max_rounds} ===")
                
                if len(manager.unlabeled_df) == 0:
                    print("No more unlabeled cells available. Stopping.")
                    break
                
                # Select cells and run labeling
                manager.start_labeling_round(batch_size=args.batch_size, mode=args.sampling_mode)
                
                # Save results
                manager.save_labeled_data()
                
                # Evaluate on all data
                manager.evaluate_on_all_data()
        
        elif load_existing_data:
            # Start active learning workflow from round 1 but with pre-loaded data
            print(f"\nStarting new active learning workflow using existing labeled data...")
            
            for i in range(1, args.max_rounds + 1):
                print(f"\n=== Active Learning Round {i}/{args.max_rounds} ===")
                
                if len(manager.unlabeled_df) == 0:
                    print("No more unlabeled cells available. Stopping.")
                    break
                
                # Select cells and run labeling
                manager.start_labeling_round(batch_size=args.batch_size, mode=args.sampling_mode)
                
                # Save results
                manager.save_labeled_data()
                
                # Evaluate on all data
                manager.evaluate_on_all_data()
                
        else:
            # Run the complete workflow from the beginning
            manager.run_active_learning_workflow(
                initial_samples=args.initial_samples,
                batch_size=args.batch_size,
                max_rounds=args.max_rounds,
                sampling_mode=args.sampling_mode
            )


# Run the script
if __name__ == "__main__":
    main()
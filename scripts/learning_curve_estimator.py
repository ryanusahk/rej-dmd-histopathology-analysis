#!/usr/bin/env python
"""
learning_curve_estimator.py - Analyze labeled data and estimate performance with additional samples

This script:
1. Finds the latest labeled data file
2. Loads the cell image data
3. Generates learning curves
4. Estimates performance with additional training samples
5. Saves plots and statistics

Usage:
    python learning_curve_estimator.py --csv_file 20250215_abe_df.csv --cropped_cells_dir ./cropped_cells --output_dir ./active_learning_output
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pickle
import argparse
from tqdm import tqdm

# Define curve fitting models
def log_model(x, a, b):
    """Logarithmic model: y = a * log(x) + b"""
    return a * np.log(x) + b

def power_law(x, a, alpha):
    """Power law model: y = a * x^(-alpha)"""
    return a * (x ** -alpha)

def exp_model(x, a, b, c):
    """Exponential model: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def inverse_sqrt(x, a, b):
    """Inverse square root model: y = a * (1/sqrt(x)) + b"""
    return a * (1/np.sqrt(x)) + b

def process_cell_red(cell_data, pad_size):
    """Extract and process the red channel from a cell image"""
    import cv2
    
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
    """Load cell data for all cells in the provided DataFrame"""
    cell_data_dict = {}
    unique_files = df["filename"].unique()
    for filename in tqdm(unique_files, desc="Loading cell files"):
        base = filename.split('.')[0]
        file_path = os.path.join(folder, f"{base}_cropped_cells.pkl")
        try:
            with open(file_path, "rb") as f:
                cell_dict = pickle.load(f)
            for cell_id, cell_data in cell_dict.items():
                cell_data_dict[(filename, cell_id)] = cell_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return cell_data_dict

def determine_pad_size(df, sample_size=100, folder="cropped_cells", percentile=99.9):
    """Quickly determine appropriate padding size from a sample of cells"""
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size)
    else:
        sample_df = df
        
    cell_data_dict = load_cell_data_dict(sample_df, folder)
    dims = []
    for key, cell_data in cell_data_dict.items():
        image = cell_data['image']
        if not hasattr(image, "shape"):
            image = np.array(image)
        dims.append(max(image.shape[0], image.shape[1]))
    
    pad_size = int(np.ceil(np.percentile(dims, percentile)))
    print(f"Determined pad size: {pad_size}")
    return pad_size

def find_latest_labeled_data(output_dir):
    """Find the latest labeled data file"""
    labeled_files = []
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} not found.")
        return None
    
    for filename in os.listdir(output_dir):
        if filename.startswith("labeled_data_round_") and filename.endswith(".csv"):
            try:
                round_num = int(filename.split("_round_")[1].split(".")[0])
                labeled_files.append((round_num, filename))
            except:
                continue
    
    if not labeled_files:
        print("No labeled data files found.")
        return None
    
    # Get the file with the highest round number
    latest_round, latest_file = max(labeled_files, key=lambda x: x[0])
    return os.path.join(output_dir, latest_file), latest_round

def analyze_learning_curves(labeled_df, cell_data_dict, pad_size, output_dir, random_state=42):
    """Analyze learning curves and estimate future performance"""
    # Filter out skipped cells
    labeled_df = labeled_df[labeled_df['hdyst_label'] != 'skip'].copy()
    
    if len(labeled_df) < 10:
        print("Not enough labeled data for analysis (minimum 10 cells required).")
        return
    
    print(f"Analyzing learning curves with {len(labeled_df)} labeled cells...")
    print(f"Label distribution: {labeled_df['hdyst_label'].value_counts().to_dict()}")
    
    # Prepare features and labels
    X = []
    y = []
    
    # Process cells to create features
    print("Extracting features from cell images...")
    for idx, row in tqdm(labeled_df.iterrows(), total=len(labeled_df)):
        key = (row['filename'], row['cell_id'])
        if key in cell_data_dict:
            cell_data = cell_data_dict[key]
            processed_img = process_cell_red(cell_data, pad_size)
            features = processed_img.flatten()
            X.append(features)
            
            # Convert label to binary (1 for hdyst+, 0 for hdyst-)
            label = 1 if row['hdyst_label'] == 'hdyst+' else 0
            y.append(label)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into a fixed test set (20%) and the rest for training curve analysis
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Determine subset sizes
    n_samples = len(X_rest)
    min_samples = max(10, int(0.1 * n_samples))  # At least 10 samples or 10%
    
    # Create subset sizes
    if n_samples <= 50:
        # For small datasets, use smaller increments
        subset_sizes = np.linspace(min_samples, n_samples, min(10, n_samples - min_samples + 1)).astype(int)
    else:
        # For larger datasets, use logarithmically spaced sizes
        subset_sizes = np.geomspace(min_samples, n_samples, min(15, n_samples - min_samples + 1)).astype(int)
        subset_sizes = np.unique(subset_sizes)  # Remove duplicates
    
    # Results storage
    results = {
        'sizes': subset_sizes,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Train models on increasing subset sizes
    print("\nTraining models on increasing dataset sizes...")
    for size in tqdm(subset_sizes):
        # Sample subset
        indices = np.random.choice(len(X_rest), size=size, replace=False)
        X_subset = X_rest[indices]
        y_subset = y_rest[indices]
        
        # Create and train model
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        clf.fit(X_subset, y_subset)
        
        # Evaluate on test set
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        results['accuracy'].append(np.mean(y_pred == y_test))
        results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
    
    # Fit curves to learning curves
    print("\nFitting prediction models to learning curves...")
    curve_models = {
        'logarithmic': log_model,
        'power_law': power_law,
        'exponential': exp_model,
        'inverse_sqrt': inverse_sqrt
    }
    
    best_fits = {}
    extrapolation_sizes = list(range(n_samples, n_samples + 1000, 200))  # Extrapolate with 200 sample increments
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_fits[metric] = {'model': None, 'mse': float('inf'), 'params': None}
        metric_vals = np.array(results[metric])
        
        # Only fit if we have valid values
        if np.isnan(metric_vals).any() or np.all(metric_vals == 0):
            print(f"Skipping curve fitting for {metric} due to invalid values.")
            continue
        
        for model_name, model_func in curve_models.items():
            try:
                # Skip exponential model for small datasets
                if model_name == 'exponential' and len(subset_sizes) < 6:
                    continue
                    
                # Fit curve
                params, _ = curve_fit(model_func, subset_sizes, metric_vals, maxfev=10000)
                
                # Calculate mean squared error
                y_pred = model_func(subset_sizes, *params)
                mse = np.mean((y_pred - metric_vals) ** 2)
                
                # Update best model if this one is better
                if mse < best_fits[metric]['mse']:
                    best_fits[metric] = {
                        'model': model_name,
                        'mse': mse,
                        'params': params,
                        'func': model_func
                    }
            except Exception as e:
                print(f"Error fitting {model_name} to {metric}: {e}")
    
    # Create extrapolation plots
    print("\nExtrapolating future performance...")
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        plt.subplot(2, 2, i+1)
        
        # Plot actual data points
        plt.scatter(subset_sizes, results[metric], label=f'Actual {metric}', color='blue')
        
        # Plot fitted curve
        if best_fits[metric]['model'] is not None:
            model_func = best_fits[metric]['func']
            params = best_fits[metric]['params']
            model_name = best_fits[metric]['model']
            
            # Plot fitted curve
            x_fit = np.linspace(min(subset_sizes), max(subset_sizes), 100)
            y_fit = model_func(x_fit, *params)
            plt.plot(x_fit, y_fit, 'r-', label=f'Fitted ({model_name})')
            
            # Plot extrapolation
            x_extra = np.array(extrapolation_sizes)
            y_extra = model_func(x_extra, *params)
            plt.plot(x_extra, y_extra, 'r--', label='Extrapolated')
            
            # Plot incremental points
            plt.scatter(x_extra, y_extra, marker='x', color='red')
            for x, y in zip(x_extra, y_extra):
                plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                             xytext=(0,10), ha='center')
            
        plt.xlabel('Training Set Size')
        plt.ylabel(metric.capitalize())
        plt.title(f'Learning Curve: {metric.capitalize()}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_extrapolation.png'), dpi=300)
    print(f"Plot saved to {os.path.join(output_dir, 'learning_curves_extrapolation.png')}")
    
    # Print statistics and extrapolation table
    print("\n==== Current Performance ====")
    print(f"Number of labeled cells: {len(labeled_df)}")
    print(f"Label distribution: {labeled_df['hdyst_label'].value_counts().to_dict()}")
    
    print("\n==== Learning Curve Models ====")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if best_fits[metric]['model'] is not None:
            print(f"{metric.capitalize()}: {best_fits[metric]['model']} model, MSE: {best_fits[metric]['mse']:.6f}")
        else:
            print(f"{metric.capitalize()}: No valid model fit")
    
    print("\n==== Extrapolated Performance ====")
    print(f"{'Size':>10} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print(f"{'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
    
    current_size = len(X_rest)
    
    # First row: current performance
    current_metrics = [results['accuracy'][-1], results['precision'][-1], 
                       results['recall'][-1], results['f1'][-1]]
    print(f"{current_size:>10} | {current_metrics[0]:>10.3f} | {current_metrics[1]:>10.3f} | "
          f"{current_metrics[2]:>10.3f} | {current_metrics[3]:>10.3f}")
    
    # Extrapolated performance
    for size in extrapolation_sizes:
        metrics = []
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if best_fits[metric]['model'] is not None:
                model_func = best_fits[metric]['func']
                params = best_fits[metric]['params']
                val = model_func(size, *params)
                # Clip to [0, 1] range
                val = max(0, min(1, val))
                metrics.append(val)
            else:
                metrics.append(float('nan'))
        
        print(f"{size:>10} | {metrics[0]:>10.3f} | {metrics[1]:>10.3f} | "
              f"{metrics[2]:>10.3f} | {metrics[3]:>10.3f}")
    
    return results, best_fits

def main():
    """Main function to analyze learning curves from labeled data"""
    parser = argparse.ArgumentParser(description='Analyze learning curves from labeled data')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the original abe_df CSV file')
    parser.add_argument('--cropped_cells_dir', type=str, default='cropped_cells', help='Directory containing cropped cells pickle files')
    parser.add_argument('--output_dir', type=str, default='active_learning_output', help='Directory containing labeled data files')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return
    
    # Find latest labeled data file
    labeled_file, round_num = find_latest_labeled_data(args.output_dir)
    if labeled_file is None:
        print("No labeled data files found. Exiting.")
        return
    
    print(f"Found latest labeled data from round {round_num}: {labeled_file}")
    
    # Load the labeled data
    try:
        labeled_df = pd.read_csv(labeled_file)
        print(f"Loaded {len(labeled_df)} labeled cells.")
    except Exception as e:
        print(f"Error loading labeled data: {e}")
        return
    
    # Filter out skipped cells
    valid_df = labeled_df[labeled_df['hdyst_label'] != 'skip'].copy()
    print(f"After removing skipped cells: {len(valid_df)} valid labeled cells")
    
    # Determine pad size
    pad_size = determine_pad_size(valid_df, sample_size=min(100, len(valid_df)), 
                                 folder=args.cropped_cells_dir)
    
    # Load cell data for all labeled cells
    print("Loading image data for all labeled cells...")
    cell_data_dict = load_cell_data_dict(valid_df, folder=args.cropped_cells_dir)
    
    # Analyze learning curves and estimate future performance
    analyze_learning_curves(valid_df, cell_data_dict, pad_size, args.output_dir, 
                           random_state=args.random_state)

if __name__ == "__main__":
    main()
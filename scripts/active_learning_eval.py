#!/usr/bin/env python
"""
evaluate_hdyst_classifier.py - Evaluate the most recent hdyst classifier

This script loads the most recent hdyst classifier trained by hdyst_active_learning_cli.py,
evaluates its accuracy on labeled data, and analyzes classification breakdowns by plotLabel
for the entire dataset. Results are printed to the terminal without saving any outputs.

Usage:
    python evaluate_hdyst_classifier.py --csv_file your_data.csv --cropped_cells_dir ./cropped_cells
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try to import from hdyst_active_learning_cli, fall back to local functions if not available
try:
    from hdyst_active_learning_cli import process_cell_red, load_cell_data_dict, determine_pad_size
    print("Imported functions from hdyst_active_learning_cli.py")
except ImportError:
    # Define the necessary functions here if import fails
    print("Could not import from hdyst_active_learning_cli.py, using local function definitions")
    
    def process_cell_red(cell_data, pad_size):
        """Process a cell's image data focusing on the red channel"""
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
        import pickle
        
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
    
    def determine_pad_size(df, folder="cropped_cells", percentile=99.9, random_state=42):
        """
        Determine padding size based on the dimensions of the cells in the DataFrame
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing cell information
        folder : str
            Directory containing the cropped cell pickle files
        percentile : float
            Percentile value to use for determining pad size
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        int : Determined pad size
        """
        print(f"Analyzing dimensions of {len(df)} cells to determine optimal padding...")
        
        # If dataset is very large, sample a subset for efficiency
        if len(df) > 5000:
            print(f"Dataset is large ({len(df)} cells), sampling 5000 cells for pad size determination...")
            np.random.seed(random_state)
            df_sample = df.sample(n=5000, random_state=random_state)
        else:
            df_sample = df
        
        # Get unique filenames to minimize file loading
        unique_files = df_sample["filename"].unique()
        
        # Dictionary to store dimensions
        dims = []
        
        # Process each file once
        for filename in tqdm(unique_files, desc="Loading cell files for dimension analysis"):
            base = filename.split('.')[0]
            file_path = os.path.join(folder, f"{base}_cropped_cells.pkl")
            try:
                with open(file_path, "rb") as f:
                    cell_dict = pickle.load(f)
                
                # Process all relevant cells from this file
                file_cells = df_sample[df_sample['filename'] == filename]
                
                for _, row in file_cells.iterrows():
                    cell_id = row['cell_id']
                    if cell_id in cell_dict:
                        cell_data = cell_dict[cell_id]
                        image = cell_data['image']
                        if not hasattr(image, "shape"):
                            image = np.array(image)
                        dims.append(max(image.shape[0], image.shape[1]))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not dims:
            print("Warning: No valid cell dimensions found. Using default pad size of 250.")
            return 250
        
        pad_size = int(np.ceil(np.percentile(dims, percentile)))
        print(f"Determined optimal pad size: {pad_size} pixels (based on {percentile}th percentile)")
        
        # Clean up to free memory
        del dims
        gc.collect()
        
        return pad_size


def find_latest_model_and_data(output_dir):
    """Find the latest model and labeled data files in the output directory"""
    model_files = []
    data_files = []
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return None, None
    
    for filename in os.listdir(output_dir):
        if filename.startswith("rf_classifier_round_") and filename.endswith(".pkl"):
            try:
                round_num = int(filename.split("_round_")[1].split(".")[0])
                model_files.append((round_num, filename))
            except:
                continue
        elif filename.startswith("labeled_data_round_") and filename.endswith(".csv"):
            try:
                round_num = int(filename.split("_round_")[1].split(".")[0])
                data_files.append((round_num, filename))
            except:
                continue
    
    if not model_files or not data_files:
        print("No model or labeled data files found.")
        return None, None
    
    latest_model = max(model_files, key=lambda x: x[0])
    latest_data = max(data_files, key=lambda x: x[0])
    
    model_path = os.path.join(output_dir, latest_model[1])
    data_path = os.path.join(output_dir, latest_data[1])
    
    print(f"Found latest model: {latest_model[1]} (round {latest_model[0]})")
    print(f"Found latest labeled data: {latest_data[1]} (round {latest_data[0]})")
    
    return model_path, data_path


def evaluate_classifier_on_labeled_data(model, labeled_df, cell_data_dict, pad_size):
    """Evaluate classifier performance on the labeled data"""
    # Filter out skipped cells
    eval_df = labeled_df[labeled_df['hdyst_label'] != 'skip'].copy()
    
    if len(eval_df) < 2:
        print("Not enough labeled cells for evaluation.")
        return
    
    # Process all cells in batch mode
    all_features = []
    true_labels = []
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing labeled cells"):
        key = (row['filename'], row['cell_id'])
        if key in cell_data_dict:
            cell_data = cell_data_dict[key]
            processed_img = process_cell_red(cell_data, pad_size)
            features = processed_img.flatten()
            all_features.append(features)
            
            # Convert label to binary (1 for hdyst+, 0 for hdyst-)
            true_labels.append(1 if row['hdyst_label'] == 'hdyst+' else 0)
    
    if len(all_features) == 0:
        print("No valid cells found for evaluation.")
        return
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y_true = np.array(true_labels)
    
    # Make predictions in one batch
    print(f"Making predictions on {len(X)} labeled cells...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print(f"CLASSIFIER ACCURACY ON LABELED DATA: {accuracy:.4f}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['hdyst-', 'hdyst+']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return accuracy


def analyze_full_dataset(model, full_df, cell_data_dict, pad_size):
    """Analyze and print statistics for the entire dataset by plotLabel"""
    # Get unique treatment groups
    plot_labels = full_df['plotLabel'].unique()
    
    # Prepare dictionary to collect results
    results_by_label = {}
    
    # Process each treatment group separately to avoid memory issues
    for label in plot_labels:
        print(f"\nAnalyzing treatment group: {label}")
        
        # Get cells for this treatment group
        group_df = full_df[full_df['plotLabel'] == label]
        
        # Process cells in batches
        all_predictions = []
        batch_size = 500  # Process 500 cells at a time
        
        for i in range(0, len(group_df), batch_size):
            batch_df = group_df.iloc[i:i+batch_size]
            
            # Extract features for this batch
            batch_features = []
            for idx, row in tqdm(batch_df.iterrows(), 
                               total=len(batch_df), 
                               desc=f"Processing {label} (batch {i//batch_size+1}/{(len(group_df)-1)//batch_size+1})"):
                key = (row['filename'], row['cell_id'])
                if key in cell_data_dict:
                    cell_data = cell_data_dict[key]
                    processed_img = process_cell_red(cell_data, pad_size)
                    batch_features.append(processed_img.flatten())
            
            if batch_features:
                # Make predictions for this batch
                X_batch = np.array(batch_features)
                batch_predictions = model.predict(X_batch)
                all_predictions.extend(batch_predictions)
        
        # Calculate statistics for this treatment group
        total_cells = len(all_predictions)
        if total_cells == 0:
            print(f"No valid cells found for {label}")
            continue
            
        hdyst_positive = sum(all_predictions)
        hdyst_negative = total_cells - hdyst_positive
        
        pos_percentage = (hdyst_positive / total_cells) * 100
        neg_percentage = (hdyst_negative / total_cells) * 100
        
        # Store results
        results_by_label[label] = {
            'total': total_cells,
            'hdyst+': hdyst_positive,
            'hdyst-': hdyst_negative,
            'hdyst+%': pos_percentage,
            'hdyst-%': neg_percentage
        }
    
    # Print results table
    print("\n" + "="*60)
    print("CLASSIFICATION BREAKDOWN BY TREATMENT GROUP")
    print("="*60)
    
    print("\nTreatment  | Total Cells | hdyst+ |   %    | hdyst- |   %")
    print("-"*60)
    
    # Sort by hdyst+ percentage for easier comparison
    for label, stats in sorted(results_by_label.items(), key=lambda x: x[1]['hdyst+%'], reverse=True):
        print(f"{label:10s} | {stats['total']:11d} | {stats['hdyst+']:6d} | {stats['hdyst+%']:6.2f}% | {stats['hdyst-']:6d} | {stats['hdyst-%']:6.2f}%")
    
    return results_by_label


def main():
    """Main function to evaluate the most recent hdyst classifier"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the most recent hdyst classifier")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the original abe_df CSV file')
    parser.add_argument('--cropped_cells_dir', type=str, default='cropped_cells', help='Directory containing cropped cells pickle files')
    parser.add_argument('--output_dir', type=str, default='active_learning_output', help='Directory containing model and labeled data files')
    parser.add_argument('--pad_size', type=int, default=None, help='Padding size for cell images (if None, will be determined from the entire dataset)')
    parser.add_argument('--percentile', type=float, default=99.9, help='Percentile to use for pad size determination (default: 99.9)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Find latest model and labeled data
    model_path, data_path = find_latest_model_and_data(args.output_dir)
    
    if model_path is None or data_path is None:
        print("Could not find necessary files. Please check the output directory.")
        return
    
    # Load the model
    print(f"\nLoading classifier from {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the labeled data
    print(f"Loading labeled data from {data_path}...")
    try:
        labeled_df = pd.read_csv(data_path)
        print(f"Loaded {len(labeled_df)} labeled cells")
    except Exception as e:
        print(f"Error loading labeled data: {e}")
        return
    
    # Load the full dataset
    print(f"Loading full dataset from {args.csv_file}...")
    try:
        full_df = pd.read_csv(args.csv_file)
        print(f"Loaded dataset with {len(full_df)} cells")
        
        # Filter the dataset
        cols_to_check = ['filename', 'cell_id', 'plotLabel']
        full_df = full_df.dropna(subset=cols_to_check).copy()
        
        # Ensure all cells have unique (filename, cell_id) pairs
        full_df = full_df.drop_duplicates(subset=['filename', 'cell_id'])
        print(f"Filtered dataset contains {len(full_df)} unique cells")
    except Exception as e:
        print(f"Error loading full dataset: {e}")
        return
    
    # Determine pad size from the entire dataset if not provided
    if args.pad_size is None:
        print(f"\nDetermining pad size from the full dataset ({len(full_df)} cells)...")
        pad_size = determine_pad_size(
            full_df, 
            folder=args.cropped_cells_dir, 
            percentile=args.percentile,
            random_state=args.random_state
        )
    else:
        pad_size = args.pad_size
        print(f"\nUsing provided pad size: {pad_size}")
    
    # Load cell data
    print("\nLoading cell data...")
    cell_data_dict = load_cell_data_dict(full_df, folder=args.cropped_cells_dir)
    print(f"Loaded data for {len(cell_data_dict)} cells")
    
    # Evaluate classifier on labeled data
    print("\nEvaluating classifier accuracy on labeled data...")
    evaluate_classifier_on_labeled_data(model, labeled_df, cell_data_dict, pad_size)
    
    # Analyze the full dataset
    print("\nAnalyzing classification breakdown by treatment group for entire dataset...")
    analyze_full_dataset(model, full_df, cell_data_dict, pad_size)


if __name__ == "__main__":
    main()
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import re
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import curve_fit

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def find_latest_labeled_data(output_dir):
    """Find the latest labeled data file in the output directory"""
    files = glob.glob(os.path.join(output_dir, "labeled_data_round_*.csv"))
    if not files:
        return None, 0
    
    # Extract round numbers
    rounds = []
    for f in files:
        match = re.search(r'round_(\d+)', f)
        if match:
            rounds.append((int(match.group(1)), f))
    
    if not rounds:
        return None, 0
    
    # Sort by round number and return the latest
    rounds.sort(key=lambda x: x[0], reverse=True)
    return rounds[0][1], rounds[0][0]

def load_cell_data_dict(df, folder='cropped_cells'):
    """Load cell data for all cells in the dataframe into a dictionary"""
    cell_data_dict = {}
    
    # Get unique filenames
    filenames = df['filename'].unique()
    
    for filename in tqdm(filenames, desc="Loading cell data files"):
        # Construct pickle filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        pickle_path = os.path.join(folder, f"{base_name}_cropped_cells.pkl")
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    cells = pickle.load(f)
                    
                # Add to dictionary with (filename, cell_id) as key
                for cell_id, cell_data in cells.items():
                    cell_data_dict[(filename, cell_id)] = cell_data
                    # Debug first few items
                    if len(cell_data_dict) <= 3:
                        print(f"DEBUG: Loaded key: {(filename, cell_id)}")
                        print(f"DEBUG: cell_id type in pickle: {type(cell_id)}")
            except Exception as e:
                print(f"Error loading {pickle_path}: {e}")
        else:
            # Debug missing files (limit output)
            if len(cell_data_dict) == 0:
                print(f"DEBUG: Could not find {pickle_path}")
            pass
            
    print(f"DEBUG: Total loaded cells in dict: {len(cell_data_dict)}")
    return cell_data_dict

def process_cell_red(cell_data, pad_size):
    """Process a single cell to extract red channel features with padding"""
    # Extract red channel (channel 0)
    if isinstance(cell_data, dict) and 'image' in cell_data:
        img = cell_data['image']
    else:
        img = cell_data
        
    # Convert to numpy array if it's not already (e.g. if it's a PIL Image)
    if not isinstance(img, np.ndarray):
        img = np.array(img)
        
    if len(img.shape) == 3:
        red_channel = img[:, :, 0]
    else:
        red_channel = img
        
    # Pad to fixed size
    h, w = red_channel.shape
    
    # Calculate padding
    pad_h = max(0, pad_size - h)
    pad_w = max(0, pad_size - w)
    
    # Pad symmetrically
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded = np.pad(red_channel, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    # Crop if larger than pad_size (shouldn't happen if pad_size is large enough)
    if padded.shape[0] > pad_size:
        start = (padded.shape[0] - pad_size) // 2
        padded = padded[start:start+pad_size, :]
    if padded.shape[1] > pad_size:
        start = (padded.shape[1] - pad_size) // 2
        padded = padded[:, start:start+pad_size]
        
    return padded

def determine_pad_size(df, folder='cropped_cells', percentile=99.9, sample_size=None, random_state=42):
    """Determine appropriate padding size based on cell sizes"""
    sizes = []
    
    if sample_size and sample_size < len(df):
        sample_df = df.sample(n=sample_size, random_state=random_state)
    else:
        sample_df = df
        
    # Get unique filenames to minimize file I/O
    filenames = sample_df['filename'].unique()
    
    for filename in tqdm(filenames, desc="Determining pad size"):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        pickle_path = os.path.join(folder, f"{base_name}_cropped_cells.pkl")
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    cells = pickle.load(f)
                
                # Get sizes for relevant cells
                relevant_cells = sample_df[sample_df['filename'] == filename]['cell_id'].values
                for cell_id in relevant_cells:
                    if cell_id in cells:
                        img = cells[cell_id]['image'] if isinstance(cells[cell_id], dict) else cells[cell_id]
                        sizes.append(max(img.shape[:2]))
            except Exception:
                continue
                
    if not sizes:
        return 100 # Default fallback
        
    return int(np.percentile(sizes, percentile))

def prepare_data(labeled_df, cell_data_dict, pad_size):
    """Prepare X (features) and y (labels) from dataframe"""
    X = []
    y = []
    
    # Filter out skipped cells
    valid_df = labeled_df[labeled_df['hdyst_label'] != 'skip'].copy()
    
    debug_count = 0
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Preparing data"):
        key = (row['filename'], row['cell_id'])
        
        # Debug first few lookups
        if debug_count < 3:
            print(f"DEBUG: Looking up key: {key}")
            print(f"DEBUG: cell_id type in df: {type(row['cell_id'])}")
            if key in cell_data_dict:
                print("DEBUG: Key FOUND")
            else:
                print("DEBUG: Key NOT FOUND")
            debug_count += 1
            
        if key in cell_data_dict:
            cell_data = cell_data_dict[key]
            processed_img = process_cell_red(cell_data, pad_size)
            X.append(processed_img.flatten())
            # 1 for hdyst+, 0 for hdyst-
            y.append(1 if row['hdyst_label'] == 'hdyst+' else 0)
    
    if len(X) == 0:
        print("ERROR: No matching cells found! Check the debug output above for key mismatches.")
        return np.array([]), np.array([])
            
    return np.array(X), np.array(y)

def plot_roc_curve(y_true, y_prob, output_dir):
    """Plot ROC curve and calculate AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, output_dir, cv_cms=None):
    """Plot confusion matrix with optional CV statistics"""
    cm = confusion_matrix(y_true, y_pred)

    # Reorder CM to match user request:
    # Rows: Positive (Top), Negative (Bottom)
    # Cols: Negative (Left), Positive (Right)
    # Standard CM is [[TN, FP], [FN, TP]] (Row 0=Neg, Row 1=Pos)
    # We want Row 0=Pos, Row 1=Neg. So swap rows.
    # Cols are already Neg, Pos.
    
    # Swap rows of cm
    cm_swapped = cm[[1, 0], :]
    
    # If we have CV confusion matrices, calculate std dev on swapped matrices
    annot_labels = []
    if cv_cms is not None:
        cv_cms = np.array(cv_cms)
        # Swap rows for all CV matrices
        cv_cms_swapped = cv_cms[:, [1, 0], :]
        
        cm_mean = np.mean(cv_cms_swapped, axis=0)
        cm_std = np.std(cv_cms_swapped, axis=0)
        
        for i in range(cm_swapped.shape[0]):
            row = []
            for j in range(cm_swapped.shape[1]):
                # Format: "Count\n(±Std)"
                row.append(f"{cm_swapped[i, j]}\n(±{cm_std[i, j]:.1f})")
            annot_labels.append(row)
        annot = np.array(annot_labels)
        fmt = ''
    else:
        annot = True
        fmt = 'd'

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_swapped, annot=annot, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Positive', 'Negative'])
    plt.ylabel('Manual Annotation')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_saturation(X, y, output_dir, n_splits=20, test_size=0.2):
    """Analyze performance metrics vs training size"""
    print("\nAnalyzing saturation (learning curves)...")
    
    # Use specific integer sizes: 250, 500, ..., 2000
    train_sizes = np.arange(250, 2001, 250)
    
    results = {
        'size': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    # Use a smaller test set (10%) to ensure we have enough training data (90% of 2300 > 2000)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    for size in tqdm(train_sizes, desc="Training sizes"):
        # Repeat a few times for stability
        for i in range(n_splits):
            # Sample subset of training data
            # Use stratify to maintain class balance if possible
            try:
                X_subset, _, y_subset, _ = train_test_split(
                    X_train_full, y_train_full, train_size=size, 
                    random_state=42+i, stratify=y_train_full
                )
            except ValueError:
                # Fallback
                X_subset, _, y_subset, _ = train_test_split(
                    X_train_full, y_train_full, train_size=size, 
                    random_state=42+i
                )
            
            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_subset, y_subset)
            
            # Predict
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            # Record metrics
            results['size'].append(len(y_subset))
            results['accuracy'].append(accuracy_score(y_test, y_pred))
            results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            try:
                results['auc'].append(auc(*roc_curve(y_test, y_prob)[:2]))
            except:
                results['auc'].append(0.5)
                
    # Convert to dataframe for plotting
    res_df = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(5, 5))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for metric in metrics:
        sns.lineplot(data=res_df, x='size', y=metric, label=metric.capitalize(), marker='o')
        
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Training Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'saturation_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return res_df

def analyze_conditions(model, full_df, cell_data_dict, pad_size, output_dir):
    """Analyze percentage positive/negative for each condition"""
    print("\nAnalyzing conditions...")
    
    # Filter for valid cells
    cols_to_check = ['filename', 'cell_id', 'plotLabel']
    valid_df = full_df.dropna(subset=cols_to_check).copy()
    valid_df = valid_df.drop_duplicates(subset=['filename', 'cell_id'])
    
    plot_labels = valid_df['plotLabel'].unique()
    results = []
    
    for label in plot_labels:
        group_df = valid_df[valid_df['plotLabel'] == label]
        
        # Collect features
        features = []
        valid_indices = []
        
        for idx, row in tqdm(group_df.iterrows(), total=len(group_df), desc=f"Processing {label}"):
            key = (row['filename'], row['cell_id'])
            if key in cell_data_dict:
                cell_data = cell_data_dict[key]
                processed_img = process_cell_red(cell_data, pad_size)
                features.append(processed_img.flatten())
                valid_indices.append(idx)
        
        if not features:
            continue
            
        # Predict
        X_group = np.array(features)
        y_pred = model.predict(X_group)
        
        n_total = len(y_pred)
        n_pos = sum(y_pred)
        n_neg = n_total - n_pos
        pct_pos = (n_pos / n_total) * 100
        pct_neg = (n_neg / n_total) * 100
        
        results.append({
            'Condition': label,
            'Total': n_total,
            'Positive': n_pos,
            'Negative': n_neg,
            'Pct_Positive': pct_pos,
            'Pct_Negative': pct_neg
        })
        
    results_df = pd.DataFrame(results)
    
    # Save table
    results_df.to_csv(os.path.join(output_dir, 'condition_stats.csv'), index=False)
    print("\nCondition Statistics:")
    print(results_df)
    
    # Plot
    plt.figure(figsize=(5, 5))
    
    # Melt for stacked bar plot or side-by-side
    plot_df = results_df[['Condition', 'Pct_Positive', 'Pct_Negative']].melt(
        id_vars='Condition', var_name='Type', value_name='Percentage'
    )
    
    # Rename types for legend
    plot_df['Type'] = plot_df['Type'].map({
        'Pct_Positive': 'Dystrophin+', 
        'Pct_Negative': 'Dystrophin-'
    })
    
    # Define colors: Positive = Blue (default seaborn), Negative = Gray
    colors = {'Dystrophin+': sns.color_palette()[0], 'Dystrophin-': 'gray'}
    
    sns.barplot(data=plot_df, x='Condition', y='Percentage', hue='Type', palette=colors)
    plt.title('Percentage by Condition')
    plt.xticks(rotation=45)
    plt.legend(title=None, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'condition_percentages.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_composite_plot(output_dir):
    """Create a final composite plot from the generated images"""
    print("\nCreating composite summary plot...")
    
    # Define the images to combine
    images = {
        'ROC Curve': 'roc_curve.png',
        'Confusion Matrix': 'confusion_matrix.png',
        'Saturation Curves': 'saturation_curves.png',
        'Condition Percentages': 'condition_percentages.png'
    }
    
    # Create figure
    fig = plt.figure(figsize=(14, 14))
    
    for i, (title, filename) in enumerate(images.items()):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            img = plt.imread(path)
            ax = fig.add_subplot(2, 2, i+1)
            ax.imshow(img)
            ax.axis('off')
            # ax.set_title(title, fontsize=14) # Removed per user request
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Active Learning Quantitative Analysis')
    parser.add_argument('--csv_file', type=str, default='202050404_abe_df.csv', help='Path to the original abe_df CSV file')
    parser.add_argument('--cropped_cells_dir', type=str, default='cropped_cells', help='Directory containing cropped cells pickle files')
    parser.add_argument('--active_learning_dir', type=str, default='active_learning_output', help='Directory containing labeled data')
    parser.add_argument('--output_dir', type=str, default='20251129_active_training_quant', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Identify latest training set
    print("Finding latest labeled data...")
    labeled_file, round_num = find_latest_labeled_data(args.active_learning_dir)
    if not labeled_file:
        print("No labeled data found!")
        return
        
    print(f"Using labeled data from round {round_num}: {labeled_file}")
    labeled_df = pd.read_csv(labeled_file)
    print(f"Total labeled cells: {len(labeled_df)}")
    
    # Load full dataset for condition analysis later
    print(f"Loading full dataset from {args.csv_file}...")
    full_df = pd.read_csv(args.csv_file)
    
    # Determine pad size
    pad_size = determine_pad_size(labeled_df, folder=args.cropped_cells_dir)
    print(f"Using pad size: {pad_size}")
    
    # Load cell data
    print("Loading cell data...")
    # We need cell data for both labeled set and full set (for condition analysis)
    # To save memory/time, we could load only what we need, but let's load all relevant
    # For now, let's load what we need for labeled first
    cell_data_dict = load_cell_data_dict(labeled_df, folder=args.cropped_cells_dir)
    
    # Prepare training data
    X, y = prepare_data(labeled_df, cell_data_dict, pad_size)
    print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")
    
    if len(X) == 0:
        return
    
    # Train final model for evaluation
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # 2. Metrics & ROC/AUC using 20-Fold Cross-Validation
    print("\nPerforming 20-Fold Cross-Validation...")
    clf_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    
    # Initialize arrays to store results
    y_pred_cv = np.zeros(len(y))
    y_prob_cv = np.zeros(len(y))
    cv_cms = []
    
    # Store metrics for each fold
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    # Manual CV loop with progress bar
    for train_idx, test_idx in tqdm(cv.split(X, y), total=20, desc="Cross-Validation Folds"):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        clf_fold = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_fold.fit(X_train_fold, y_train_fold)
        
        # Store predictions
        y_pred_fold = clf_fold.predict(X_test_fold)
        y_prob_fold = clf_fold.predict_proba(X_test_fold)[:, 1]
        
        y_pred_cv[test_idx] = y_pred_fold
        y_prob_cv[test_idx] = y_prob_fold
        
        cv_cms.append(confusion_matrix(y_test_fold, y_pred_fold))
        
        # Calculate fold metrics
        fold_acc = accuracy_score(y_test_fold, y_pred_fold)
        fold_prec = precision_score(y_test_fold, y_pred_fold, zero_division=0)
        fold_rec = recall_score(y_test_fold, y_pred_fold, zero_division=0)
        fold_f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test_fold, y_prob_fold)
        except:
            fold_auc = 0.5
            
        fold_metrics['accuracy'].append(fold_acc)
        fold_metrics['precision'].append(fold_prec)
        fold_metrics['recall'].append(fold_rec)
        fold_metrics['f1'].append(fold_f1)
        fold_metrics['auc'].append(fold_auc)
    
    # Calculate Mean and STD
    metrics_summary = {}
    for metric, values in fold_metrics.items():
        metrics_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        
    # Generate Report
    report_content = "Final Model Performance (20-Fold Cross-Validation)\n"
    report_content += "================================================\n\n"
    report_content += f"Total Samples: {len(y)}\n"
    report_content += f"Positive Samples: {sum(y)}\n"
    report_content += f"Negative Samples: {len(y) - sum(y)}\n\n"
    
    report_content += "Metrics (Mean ± STD):\n"
    report_content += "---------------------\n"
    for metric, stats in metrics_summary.items():
        report_content += f"{metric.capitalize():<10}: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
        
    report_content += "\nDetailed Classification Report (Pooled Predictions):\n"
    report_content += classification_report(y, y_pred_cv)
    
    print("\n" + report_content)
    
    # Save report
    with open(os.path.join(args.output_dir, 'final_metrics_report.txt'), 'w') as f:
        f.write(report_content)
    
    print("\nCross-Validation Metrics (20-Fold):")
    print(classification_report(y, y_pred_cv))
    
    # Plot ROC
    plot_roc_curve(y, y_prob_cv, args.output_dir)
    plot_confusion_matrix(y, y_pred_cv, args.output_dir, cv_cms=cv_cms)
    
    # 4. Saturation Plots
    analyze_saturation(X, y, args.output_dir)
    
    # 3. Condition Analysis
    # We need to load more cell data for the full dataset
    print("\nLoading additional cell data for condition analysis...")
    # Filter full_df to only rows with valid plotLabel
    full_df_valid = full_df.dropna(subset=['plotLabel', 'filename', 'cell_id'])
    
    # Load data for full dataset (this might take a while)
    # We update the existing dict
    cell_data_dict_full = load_cell_data_dict(full_df_valid, folder=args.cropped_cells_dir)
    cell_data_dict.update(cell_data_dict_full)
    
    analyze_conditions(clf, full_df_valid, cell_data_dict, pad_size, args.output_dir)
    
    # Create composite plot
    create_composite_plot(args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

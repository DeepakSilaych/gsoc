# genetic_analysis.py
import pandas as pd
import numpy as np
import malariagen_data
import allel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import concurrent.futures
import warnings
import json

# --- 1. Initial Setup ---
print("--- Step 1: Initial Data Loading and Preprocessing Setup ---")
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize malariagen_data API
ag3 = malariagen_data.Ag3(
    url="https://vo_agam_production.cog.sanger.ac.uk/v3/full/zarr/ag3.0/",
    cohorts_url="https://vo_agam_production.cog.sanger.ac.uk/v3/full/cohorts_meta.json",
    sample_sets="3.3",
    debug=False,
    show_progress=False
)
samples_metadata = ag3.sample_metadata()

# Filter and select a subset of samples for training and validation
gambiae_coluzzii_samples = samples_metadata.query("aim_species in ['gambiae', 'coluzzii']").dropna(subset=["sample_id", "aim_species"])
num_samples_for_training = 500
sub_samples = gambiae_coluzzii_samples.sample(num_samples_for_training, random_state=42)
sub_sample_ids = sub_samples['sample_id'].tolist()
print(f"Selected {len(sub_samples)} samples for training and validation.")

# Align labels (y) with the selected samples
y_labels_all_samples = sub_samples.set_index('sample_id')['aim_species']

def encode_diploid_allel(gt_array):
    """Encodes a diploid genotype array into a dosage matrix."""
    gt = allel.GenotypeArray(gt_array)
    allele_counts = gt.to_allele_counts()
    alt_allele_dosage = allele_counts[:, :, 1]
    dosage_matrix = np.where(gt.is_missing(), np.nan, alt_allele_dosage)
    return dosage_matrix

# --- 2. Define Genomic Partitions and Directories ---
print("\n--- Step 2: Defining Genomic Partitions and Directories ---")

partitions_list_file = 'partitions_list.json'
all_partitions_to_process = []
partitions_exist = False

# Check if the partitions file exists and is not empty
if os.path.exists(partitions_list_file) and os.path.getsize(partitions_list_file) > 0:
    try:
        with open(partitions_list_file, 'r') as f:
            all_partitions_to_process = json.load(f)
        partitions_exist = True
        print(f"  ✅ Loading partitions from {partitions_list_file}...")
    except json.JSONDecodeError:
        print(f"  ❌ Error decoding {partitions_list_file}. File may be corrupted. Regenerating...")
        partitions_exist = False

if not partitions_exist:
    print(f"  🚀 Generating partitions and saving to {partitions_list_file}...")
    contigs = ag3.contigs
    partition_window_size = 1_000_000

    for contig in contigs:
        try:
            contig_callset_info = ag3.snp_calls(region=contig, sample_query=f"sample_id in {sub_sample_ids}")
            if 'variant_position' not in contig_callset_info or contig_callset_info['variant_position'].shape[0] == 0:
                continue
            max_pos = contig_callset_info['variant_position'].values.max()
        except Exception as e:
            continue
        start_pos = 1
        while start_pos <= max_pos:
            end_pos = min(start_pos + partition_window_size - 1, max_pos)
            # Convert NumPy integers to standard Python integers for JSON serialization
            all_partitions_to_process.append((contig, int(start_pos), int(end_pos)))
            start_pos = end_pos + 1
    
    with open(partitions_list_file, 'w') as f:
        json.dump(all_partitions_to_process, f, indent=4)

print(f"Generated a total of {len(all_partitions_to_process)} partitions to process.")

output_dir = 'trained_classifiers'
results_dir = 'validation_results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print(f"Trained classifiers will be saved in: {output_dir}")
print(f"Validation results will be saved in: {results_dir}")

# --- 3. Define the Worker Function for a Single Partition ---
def process_partition(partition_tuple):
    """
    Worker function to process a single genomic partition,
    skipping if the model and results already exist.
    """
    p_contig, p_start, p_end = partition_tuple
    partition_id = f"{p_contig}_{p_start}-{p_end}"
    
    model_filename = os.path.join(output_dir, f"classifier_{partition_id}.joblib")
    results_filename = os.path.join(results_dir, f"results_{partition_id}.json")

    # Check if results and model already exist to skip re-computation
    if os.path.exists(model_filename) and os.path.exists(results_filename):
        print(f"  ✅ Partition {partition_id} already processed. Loading results.")
        try:
            with open(results_filename, 'r') as f:
                partition_results = json.load(f)
            return partition_id, partition_results
        except json.JSONDecodeError:
            print(f"  ❌ Error loading results for {partition_id}. File corrupted. Re-processing...")
            # If JSON is corrupted, delete the files and proceed to re-process
            os.remove(model_filename)
            os.remove(results_filename)
            # Fall through to the training section
    
    print(f"\n  🚀 Training for Partition: {partition_id} ---")
    try:
        # Load SNP data for the current partition
        callset_partition = ag3.snp_calls(
            region=f"{p_contig}:{p_start}-{p_end}", 
            sample_query=f"sample_id in {sub_sample_ids}"
        )
        
        # Skip if no variants in the partition
        if callset_partition['call_genotype'].shape[0] == 0:
            print(f"    No variants in partition {partition_id}. Skipping.")
            return None

        # Preprocessing steps
        p_call_genotype = callset_partition['call_genotype'].values
        p_missing_genotype_mask = (p_call_genotype[:, :, 0] == -1) | (p_call_genotype[:, :, 1] == -1)
        p_variant_missingness = np.mean(p_missing_genotype_mask, axis=1)
        p_sample_missingness = np.mean(p_missing_genotype_mask, axis=0)

        p_filtered_variants_idx = np.where(p_variant_missingness <= 0.05)[0]
        p_filtered_samples_idx = np.where(p_sample_missingness <= 0.05)[0]

        # Skip if not enough valid variants or samples after filtering
        if len(p_filtered_variants_idx) == 0 or len(p_filtered_samples_idx) == 0:
            print(f"    No valid variants or samples after filtering in partition {partition_id}. Skipping.")
            return None

        p_filtered_gt = p_call_genotype[p_filtered_variants_idx, :, :][:, p_filtered_samples_idx, :]
        p_filtered_sample_ids = callset_partition['sample_id'].values[p_filtered_samples_idx]
        
        y_partition = y_labels_all_samples.loc[p_filtered_sample_ids]
        p_encoded_genotypes = encode_diploid_allel(p_filtered_gt)
        p_imputer = SimpleImputer(strategy='most_frequent')
        p_imputed_genotypes = p_imputer.fit_transform(p_encoded_genotypes.T).T

        X_partition = pd.DataFrame(
            p_imputed_genotypes.T, 
            index=p_filtered_sample_ids, 
            columns=callset_partition['variant_position'].values[p_filtered_variants_idx]
        )
        
        # Ensure X_partition and y_partition have aligned indices
        common_samples = X_partition.index.intersection(y_partition.index)
        X_partition = X_partition.loc[common_samples]
        y_partition = y_partition.loc[common_samples]

        # Skip if not enough samples or classes for training
        if X_partition.shape[0] == 0 or len(y_partition.unique()) < 2:
            print(f"    Not enough samples or classes for training in partition {partition_id}. Skipping.")
            return None
        
        # K-Fold Cross-Validation for the current partition
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        partition_accuracy_scores = []
        partition_precision_scores = []
        partition_recall_scores = []
        partition_f1_scores = []
        partition_confusion_matrices = []

        for fold, (train_index, test_index) in enumerate(kf.split(X_partition, y_partition)):
            X_train, X_test = X_partition.iloc[train_index], X_partition.iloc[test_index]
            y_train, y_test = y_partition.iloc[train_index], y_partition.iloc[test_index]

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2) # Use 2 jobs per model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate and store metrics
            partition_accuracy_scores.append(accuracy_score(y_test, y_pred))
            partition_precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            partition_recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            partition_f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            partition_confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=model.classes_))

        # Store aggregated results for this partition
        partition_results = {
            'accuracy_mean': np.mean(partition_accuracy_scores),
            'accuracy_std': np.std(partition_accuracy_scores),
            'precision_mean': np.mean(partition_precision_scores),
            'precision_std': np.std(partition_precision_scores),
            'recall_mean': np.mean(partition_recall_scores),
            'recall_std': np.std(partition_recall_scores),
            'f1_mean': np.mean(partition_f1_scores),
            'f1_std': np.std(partition_f1_scores),
            'classes': model.classes_.tolist(), # Store classes for heatmap
            'aggregated_confusion_matrix': np.sum(partition_confusion_matrices, axis=0).tolist()
        }
        
        # Save the trained model and the validation results
        joblib.dump(model, model_filename)
        with open(results_filename, 'w') as f:
            json.dump(partition_results, f, indent=4)
        print(f"  ✅ Model and results for partition {partition_id} saved.")
        return partition_id, partition_results

    except Exception as e:
        print(f"  ❌ An error occurred while processing partition {partition_id}: {e}. Skipping this partition.")
        # Clean up potentially corrupted files if an error occurred during processing
        if os.path.exists(model_filename):
            os.remove(model_filename)
        if os.path.exists(results_filename):
            os.remove(results_filename)
        return None

# --- 4. Parallel Execution ---
print("\n--- Step 4: Parallel Processing of Partitions ---")
all_partition_results = {}
overall_confusion_matrices = []
max_workers = os.cpu_count() * 2 # A common heuristic for I/O-bound tasks

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # `executor.map` applies the function to each item in the iterable
    for result in executor.map(process_partition, all_partitions_to_process):
        if result is not None:
            partition_id, partition_results = result
            all_partition_results[partition_id] = partition_results
            # Convert aggregated confusion matrix from list back to numpy array for sum operation
            overall_confusion_matrices.append(np.array(partition_results['aggregated_confusion_matrix']))

# --- 5. Overall Aggregated Results (Main thread) ---
print("\n--- Step 5: Overall Aggregated Performance Across All Partitions ---")

if not all_partition_results:
    print("No partitions were successfully processed and trained.")
else:
    # Convert results dictionary to DataFrame for easier analysis
    results_df = pd.DataFrame.from_dict(all_partition_results, orient='index')
    print("\nSummary of results per partition:")
    print(results_df[['accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean']].round(4))

    # Calculate overall average metrics
    overall_accuracy_mean = results_df['accuracy_mean'].mean()
    overall_precision_mean = results_df['precision_mean'].mean()
    overall_recall_mean = results_df['recall_mean'].mean()
    overall_f1_mean = results_df['f1_mean'].mean()

    print(f"\nOverall Average Accuracy across all processed partitions: {overall_accuracy_mean:.4f}")
    print(f"Overall Average Precision across all processed partitions: {overall_precision_mean:.4f}")
    print(f"Overall Average Recall across all processed partitions: {overall_recall_mean:.4f}")
    print(f"Overall Average F1-Score across all processed partitions: {overall_f1_mean:.4f}")

    # Plot overall aggregated confusion matrix
    if overall_confusion_matrices:
        # Sum all individual confusion matrices
        overall_avg_conf_matrix = np.sum(overall_confusion_matrices, axis=0)
        
        # Get classes from one of the partition results (assuming they are consistent)
        sample_classes = results_df['classes'].iloc[0] 

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            overall_avg_conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=sample_classes, 
            yticklabels=sample_classes
        )
        plt.title('Overall Aggregated Confusion Matrix (Sum of all Partition Folds)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        print("\nOverall Aggregated Confusion Matrix values (Sum of all Partition Folds):")
        print(overall_avg_conf_matrix)
    else:
        print("\nNo overall confusion matrix to display as no partitions were processed.")

print("\n--- Process Complete ---")

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Weights for the ensemble (Must sum to 1.0)
# Adjust based on which model had better local CV or Public LB score.
# Example: If Siglip was better, give it 0.6.
W_SIGLIP = 0.48
W_DINO   = 0.52

FILES = {
    'siglip': 'submission70.csv',
    'dino':   'submission72.csv'
}

OUTPUT_FILE = 'submission.csv'

# Target definitions required for Mass Balance
ALL_TARGETS = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enforce_mass_balance(df_wide, fixed_clover=None):
    """
    Applies Orthogonal Projection to enforce biological constraints:
    1. Dry_Green_g + Dry_Clover_g = GDM_g
    2. GDM_g + Dry_Dead_g = Dry_Total_g
    
    If fixed_clover is True, Dry_Clover_g is kept fixed and only other targets are adjusted.
    
    This finds the closest set of values to the predictions that satisfy 
    the constraints (minimizing Euclidean distance modification).
    """
    # 1. Ensure columns are in the specific order for the matrix math
    # Vector x = [Green, Clover, Dead, GDM, Total]
    ordered_cols = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']
    
    # Extract values: Shape (5, N_samples)
    Y = df_wide[ordered_cols].values.T
    
    if fixed_clover:
        # Keep Dry_Clover_g fixed, adjust only other targets
        # We have: Green + Clover_fixed = GDM, so GDM = Green + Clover_fixed
        # And: GDM + Dead = Total, so Total = GDM + Dead = Green + Clover_fixed + Dead
        # Extract fixed Clover values
        clover_fixed = Y[1, :].copy()  # Dry_Clover_g is index 1
        # Adjust: GDM = Green + Clover_fixed
        Y[3, :] = Y[0, :] + clover_fixed  # GDM_g
        # Adjust: Total = GDM + Dead = Green + Clover_fixed + Dead
        Y[4, :] = Y[3, :] + Y[2, :]  # Dry_Total_g
        # Keep Green and Dead as is (or apply minimal adjustment if needed)
        # Clover stays fixed (already set)
        Y_reconciled = Y
    else:
        # Original method: adjust all targets
        # 2. Define Constraint Matrix C where Cx = 0
        # Eq 1: 1*Gr + 1*Cl + 0*De - 1*GDM + 0*Tot = 0
        # Eq 2: 0*Gr + 0*Cl + 1*De + 1*GDM - 1*Tot = 0
        C = np.array([
            [1, 1, 0, -1,  0],
            [0, 0, 1,  1, -1]
        ])
        
        # 3. Calculate Projection Matrix P = I - C^T * (C * C^T)^-1 * C
        C_T = C.T
        try:
            inv_CCt = np.linalg.inv(C @ C_T)
            P = np.eye(5) - C_T @ inv_CCt @ C
        except np.linalg.LinAlgError:
            # Fallback if singular (unlikely with this specific matrix)
            print("Warning: Singular matrix in projection. Skipping constraint enforcement.")
            return df_wide

        # 4. Apply Projection
        Y_reconciled = P @ Y
    
    # 5. Transpose back to (N_samples, 5) and clip negatives
    Y_reconciled = Y_reconciled.T
    Y_reconciled = np.maximum(0, Y_reconciled) 
    
    # 6. Update DataFrame
    df_out = df_wide.copy()
    df_out[ordered_cols] = Y_reconciled
    
    return df_out

def robust_ensemble(file_paths, weights):
    print(f"--- Starting Ensemble ---")
    print(f"Weights: {weights}")
    print("NOTE: Using DINO-only for Dry_Clover_g (better detection)")
    
    dfs = []
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        
        # Read and sort by sample_id to ensure alignment
        df = pd.read_csv(path).sort_values('sample_id').reset_index(drop=True)
        dfs.append(df)
        print(f"Loaded {name}: {len(df)} rows")

    # 1. Check alignment
    base_ids = dfs[0]['sample_id']
    if not all(df['sample_id'].equals(base_ids) for df in dfs[1:]):
        raise ValueError("Sample IDs do not match between submission files!")

    # 2. Split by target_name to handle Dry_Clover_g separately
    # Split sample_id into image_id and target_name
    dfs_split = []
    for df in dfs:
        df_split = df.copy()
        df_split[['image_id', 'target_name']] = df_split['sample_id'].str.rsplit('__', n=1, expand=True)
        dfs_split.append(df_split)
    
    # Get DINO and SigLIP dataframes (identify by order in file_paths)
    dino_idx = list(file_paths.keys()).index('dino')
    siglip_idx = list(file_paths.keys()).index('siglip')
    dino_df = dfs_split[dino_idx]
    siglip_df = dfs_split[siglip_idx]
    
    # 3. Separate targets: Dry_Clover_g uses DINO only, others use ensemble
    ensemble_results = []
    
    for target in ALL_TARGETS:
        # Filter for this target
        dino_target = dino_df[dino_df['target_name'] == target].copy()
        siglip_target = siglip_df[siglip_df['target_name'] == target].copy()
        
        if target == 'Dry_Clover_g':
            # Use DINO only for Clover
            print(f"Using DINO-only for {target}")
            ensemble_results.append(dino_target[['sample_id', 'target']])
        else:
            # Ensemble for other targets
            # Merge on sample_id to align
            merged = pd.merge(
                dino_target[['sample_id', 'target']].rename(columns={'target': 'dino_pred'}),
                siglip_target[['sample_id', 'target']].rename(columns={'target': 'siglip_pred'}),
                on='sample_id'
            )
            
            # Weighted average
            w_vec = np.array([weights['dino'], weights['siglip']])
            w_vec = w_vec / w_vec.sum()
            
            merged['target'] = merged['dino_pred'] * w_vec[0] + merged['siglip_pred'] * w_vec[1]
            ensemble_results.append(merged[['sample_id', 'target']])
    
    # 4. Combine all targets
    ensemble_df = pd.concat(ensemble_results, ignore_index=True)
    print("Weighted average complete (DINO-only for Clover).")

    # 5. Prepare for Mass Balance (Convert Long -> Wide)
    ensemble_df[['image_id', 'target_name']] = ensemble_df['sample_id'].str.rsplit('__', n=1, expand=True)
    
    # Pivot
    wide_df = ensemble_df.pivot(index='image_id', columns='target_name', values='target').reset_index()

    # 6. Apply Robust Constraints (with Dry_Clover_g fixed)
    print("Applying Mass Balance Constraints (Dry_Clover_g fixed to DINO values)...")
    wide_balanced = enforce_mass_balance(wide_df, fixed_clover=True)

    # 7. Convert back (Wide -> Long)
    long_balanced = wide_balanced.melt(
        id_vars='image_id', 
        value_vars=ALL_TARGETS,
        var_name='target_name',
        value_name='target'
    )

    # Reconstruct sample_id
    long_balanced['sample_id'] = long_balanced['image_id'] + '__' + long_balanced['target_name']

    # 8. Final Formatting
    final_submission = long_balanced[['sample_id', 'target']].sort_values('sample_id').reset_index(drop=True)

    return final_submission

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Define weights
    ensemble_weights = {
        'siglip': W_SIGLIP,
        'dino': W_DINO
    }
    
    try:
        # Run Ensemble
        submission = robust_ensemble(FILES, ensemble_weights)
        
        # Save
        submission.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved to {OUTPUT_FILE}")
        print(submission.head())
        
        # Sanity Check Stats
        print("\nStats:")
        print(submission['target'].describe())
        
    except Exception as e:
        print(f"\nError during ensembling: {e}")
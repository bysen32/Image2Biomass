import os
import pandas as pd

def ensemble():
    print("--- [Final Step] Performing a weighted average of all model predictions ---")
    
    files = {
        "DINOv2-Giant": "/kaggle/working/submission_dino_giant.csv", 
        "ConvnextTiny": "/kaggle/working/submission_ConvnextTiny.csv", 
        "SigLIP": "/kaggle/working/submission_SigLIP.csv"
    }

    # Check if all necessary files exist
    for key, filename in files.items():
        if not os.path.exists(filename):
            print(f"Error: {filename} not found.")
            print("Please run all inference scripts before ensembling.")
            return

    # Load the submission file from each model
    # Map to df1-df3 variables and apply weights
    df1 = pd.read_csv(files["DINOv2-Giant"]).set_index("sample_id")
    df2 = pd.read_csv(files["ConvnextTiny"]).set_index("sample_id") 
    df3 = pd.read_csv(files["SigLIP"]).set_index("sample_id") 
    
    # --- Perform the weighted average ---
    final_submission = (
        0.25 * df1 +   # DINOv2-Giant
        0.40 * df2 +   # ConvnextTiny (LB 0.61)
        0.35 * df3     # Imagebind
    )    
    
    # Save the final submission file
    final_submission.to_csv("submission.csv")

    print("\n🎉 All processes are complete! The final submission file 'submission.csv' has been created.")
    print("--- First 5 rows of the submission file ---")
    print(final_submission.head())


if __name__ == "__main__":
    ensemble()
"""
Script to combine train and test CSV files for pretraining
"""

import pandas as pd
import os

def combine_soluprotmut_files():
    """Combine soluprotmut_solubility train and test files"""
    
    # File paths
    train_path = 'preprocess_data_v5/combined_ddG_train.csv'
    test_path = 'preprocess_data_v5/combined_ddG_test.csv'
    output_path = 'preprocess_data_v5/combined_ddG_all.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    combined_df.to_csv(output_path, index=False)
    fixed_columns = {'sequence', 'mutation1', 'mutation2', 'mutation3', 'mutation4', 'mutation5', 'mutation6', 'mutation7', 'source', 'original_feature'}
    all_columns = set(combined_df.columns)
    potential_targets = [col for col in all_columns if col.lower() not in {c.lower() for c in fixed_columns}]
    
    if potential_targets:
        target_col = potential_targets[0]
        print(f"   Target column: {target_col}")
        
if __name__ == "__main__":
    combine_soluprotmut_files() 
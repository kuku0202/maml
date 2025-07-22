import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Callable, Optional, Tuple, List
import warnings
import os
import glob
import re

class MutationDataPreprocessor:
    """
    A class to preprocess mutation data with different formulations
    across sources and normalize them to a common range.
    """
    
    def __init__(self):
        self.formula_transforms = {}
        self.scalers = {}
        self.source_stats = {}
        
    def validate_mutation_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate mutation positions against protein sequences and remove invalid data.
        
        Args:
            df: DataFrame with 'sequence' and mutation
            
        Returns:
            DataFrame with all valid mutations
        """
        df_valid = df.copy()
        initial_count = len(df_valid)
        
        # Standard amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Check for valid mutations
        for idx, row in df_valid.iterrows():
            mutations = []
            sequence = str(row['sequence']).upper()
            for col in ['mutation1', 'mutation2', 'mutation3', 'mutation4', 
                       'mutation5', 'mutation6', 'mutation7']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    mutations.append(str(row[col]).strip())
            if not mutations:
                continue
                
            # Validate each mutation
            for mutation in mutations:
                if mutation == '':
                    continue
                
                if len(mutation) >= 3:
                    orig_aa = mutation[0]
                    new_aa = mutation[-1]
                    position_str = mutation[1:-1]
                    
                    # Check if format is valid
                    if (position_str.isdigit() and orig_aa in valid_aa and new_aa in valid_aa):  
                        position = int(position_str) - 1
                        if position < 0 or position >= len(sequence):
                            print(f"Invalid mutation {mutation} or position {position+1} out of range")
                            df_valid = df_valid.drop(idx)
                            break
                        
                        # Check if original amino acid matches sequence
                        if sequence[position] != orig_aa:
                            print(f"Invalid mutation {mutation}: expected {orig_aa} at position {position+1}, found {sequence[position]}")
                            df_valid = df_valid.drop(idx)
                            break
                    else:
                        print(f"Invalid AA: {mutation}")
                        df_valid = df_valid.drop(idx)
                        break
                else:
                    print(f"Invalid mutation format: {mutation}")
                    df_valid = df_valid.drop(idx)
                    break
        
        # Reset index after dropping rows
        df_valid = df_valid.reset_index(drop=True)
        
        print(f"Mutation validation complete:")
        print(f"  - Initial samples: {initial_count}")
        print(f"  - Valid samples remaining: {len(df_valid)}")
        
        return df_valid
        
    def add_formula_transform(self, source: str, transform_func: Callable[[float], float]):
        """
        Add a formula transformation for a specific source.
        Args:
            source: Source identifier
            transform_func: Function to transform values
        """
        self.formula_transforms[source] = transform_func
        
    def apply_formula_normalization(self, df: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """
        Apply source-specific formula transformations to values.
        
        Args:
            df: DataFrame with columns including 'source' and value column
            value_column: Name of the column containing values to transform
            
        Returns:
            DataFrame with transformed values
        """
        df_processed = df.copy()
        
        for source in df_processed['source'].unique():
            mask = df_processed['source'] == source
            
            if source in self.formula_transforms:
                original_values = df_processed.loc[mask, value_column]
                transformed_values = original_values.apply(self.formula_transforms[source])
                df_processed.loc[mask, value_column] = transformed_values
                
                print(f"Applied formula transformation to {source}")
            else:
                print(f"Using original values")
                
        return df_processed
    
    def normalize_by_source(self, df: pd.DataFrame, value_column: str,
                           method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize values within given source to the target range.
        
        Args:
            df: DataFrame with transformed values
            value_column: Name of the column containing values to normalize
            method: 'minmax' for MinMaxScaler, 'standard' for StandardScaler
            
        Returns:
            DataFrame with normalized values
        """
        df_normalized = df.copy()
        
        for source in df_normalized['source'].unique():
            mask = df_normalized['source'] == source
            source_data = df_normalized.loc[mask, value_column].values.reshape(-1, 1)
            valid_mask = ~np.isnan(source_data.flatten())
            if valid_mask.sum() == 0:
                continue
            if method == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))  
            elif method == 'standard':
                scaler = StandardScaler()
            valid_data = source_data[valid_mask].reshape(-1, 1)
            scaler.fit(valid_data)
            normalized_values = np.full_like(source_data.flatten(), np.nan)
            normalized_values[valid_mask] = scaler.transform(valid_data).flatten()
            
            df_normalized.loc[mask, value_column] = normalized_values
            
            # Store scaler and stats for later use
            self.scalers[source] = scaler
            self.source_stats[source] = {
                'original_min': np.nanmin(source_data),
                'original_max': np.nanmax(source_data),
                'original_mean': np.nanmean(source_data),
                'original_std': np.nanstd(source_data),
                'valid_samples': valid_mask.sum(),
                'total_samples': len(source_data)
            }   
        return df_normalized
    
    def preprocess(self, df: pd.DataFrame, value_column: str,
                   normalization_method: str = 'standard', validate_mutations: bool = True, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline: validation + formula transformation + normalization + train-test split.
        
        Args:
            df: Raw DataFrame
            value_column: Name of the column containing values to process
            normalization_method: 'minmax' or 'standard' (default: 'standard')
            validate_mutations: Whether to validate mutation positions
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple of (train DataFrame, test DataFrame)
        """
        df_valid = self.validate_mutation_positions(df)
        df_transformed = self.apply_formula_normalization(df_valid, value_column)
        df_normalized = self.normalize_by_source(df_transformed, value_column, normalization_method)
        df_train, df_test = train_test_split(
            df_normalized, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df_normalized['source'] if len(df_normalized['source'].unique()) > 1 else None
        )
        print(f"Train shape: {df_train.shape}")
        print(f"Test shape: {df_test.shape}")
    
        return df_train, df_test
    
    def get_preprocessing_summary(self) -> pd.DataFrame:
        """
        Get a summary of preprocessing statistics for each source.
        
        Returns:
            DataFrame with preprocessing statistics
        """
        if not self.source_stats:
            return pd.DataFrame()
            
        summary_data = []
        for source, stats in self.source_stats.items():
            summary_data.append({
                'source': source,
                'total_samples': stats['total_samples'],
                'valid_samples': stats['valid_samples'],
                'original_min': stats['original_min'],
                'original_max': stats['original_max'],
                'original_mean': stats['original_mean'],
                'original_std': stats['original_std'],
                'has_formula_transform': source in self.formula_transforms
            })
            
        return pd.DataFrame(summary_data)

SOURCE_TRANSFORMATIONS = {

    'default': lambda x: x,
    'log10_k50_t': lambda x: 10**x,
    'enrichment': lambda x: 2**x,
    'log_fitness_by_syn_mut_fitness': lambda x: 2**x
}

NORMALIZATION_METHOD = 'standard'

def get_value_column(df: pd.DataFrame) -> str:
    """
    Get the value column from the DataFrame.
    Since we know the standard format, just find the one numeric column that's not the standard ones.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the value column
    """
    standard_cols = ['sequence', 'mutation1', 'mutation2', 'mutation3', 'mutation4', 
                    'mutation5', 'mutation6', 'mutation7', 'source', 'original_feature']
    
    value_cols = ['solubility_change', 'ddG', 'dTm', 'fluorescence_intensity', 
                       'binding_affinity', 'thermal_stability', 'stability_proxy', 
                       'functional_fitness', 'enrichment_score']
    
    for col in value_cols:
        if col in df.columns:
            return col
    value_cols = [col for col in df.columns if col not in standard_cols]
    
    if len(value_cols) == 0:
        raise ValueError(f"No value column found. Available columns: {df.columns.tolist()}")
    elif len(value_cols) == 1:
        return value_cols[0]
    else:
        raise ValueError(f"Multiple value columns found: {value_cols}")

def process_csvs(input_files, output_dir: str = None, validate_mutations: bool = True, 
                test_size: float = 0.2, random_state: int = 42) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, MutationDataPreprocessor]]:
    """
    Process one or multiple CSV files.
    
    Args:
        input_files: List of files to process
        output_dir: Directory for output files (default: same as input files)
        validate_mutations: Whether to validate mutation positions (default: True)
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Dictionary mapping input filenames to (train DataFrame, test DataFrame, preprocessor) tuples
    """
    
    if isinstance(input_files, str):
        if '*' in input_files:
            files = glob.glob(input_files)
        else:
            files = [input_files]
    elif isinstance(input_files, list):
        files = input_files
    else:
        raise ValueError("input_files must be valid")
    if not files:
        print(f"No files found matching: {input_files}")
        return {}
    results = {}
    successful = 0
    
    for input_file in files:
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        train_output_file = os.path.join(output_dir, f"{name_without_ext}_train.csv")
        test_output_file = os.path.join(output_dir, f"{name_without_ext}_test.csv")
        
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            results[input_file] = (None, None, None)
            continue
        
        # Get value column
        try:
            value_column = get_value_column(df)
            print(f"value column: {value_column}")
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            nan_count = df[value_column].isna().sum()
            if nan_count > 0:
                print(f"{nan_count} converted to NaN")
            df = df.dropna(subset=[value_column])
            print(f"Rows remaining after dropping NaN: {len(df)}")
        except ValueError as e:
            print(f"Error finding value column in {input_file}: {e}")
            results[input_file] = (None, None, None)
            continue
        
        if 'source' not in df.columns:
            print(f"Error: 'source' column not found in {input_file}")
            results[input_file] = (None, None, None)
            continue
        

        preprocessor = MutationDataPreprocessor()
        original_feature_in_file = df['original_feature'].unique()
        
        for original_feature in original_feature_in_file:
            if original_feature in SOURCE_TRANSFORMATIONS:
                preprocessor.add_formula_transform(original_feature, SOURCE_TRANSFORMATIONS[original_feature])
            else:
                preprocessor.add_formula_transform(original_feature, SOURCE_TRANSFORMATIONS['default'])
                
        try:
            df_train, df_test = preprocessor.preprocess(
                df, value_column, NORMALIZATION_METHOD, validate_mutations, 
                test_size, random_state
            )
            
            # Save results
            df_train.to_csv(train_output_file, index=False)
            df_test.to_csv(test_output_file, index=False)
            
            # Print summary
            summary = preprocessor.get_preprocessing_summary()
            
            
            results[input_file] = (df_train, df_test, preprocessor)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            results[input_file] = (None, None, None)

    print(f"PROCESSING FINISHED")
    
    # return results

def main():
    process_csvs("dataset_v3/*.csv", output_dir="preprocess_data_v5")

if __name__ == "__main__":
    main()
    

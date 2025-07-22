import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import re
from typing import List, Dict, Optional, Tuple

class ProteinDataset(Dataset):
    """
    Optimized PyTorch Dataset for protein data with pre-computed tokenization.
    """
    def __init__(self, csv_path, tokenizer, max_length=1024, target_col=None, precompute_tokens=True, model_name='prot_bert'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.csv_path = csv_path
        self.precompute_tokens = precompute_tokens
        self.model_name = model_name
        
        self.df = pd.read_csv(csv_path)
        
        if target_col is None:
            self.target_col = self._auto_detect_target_column()
        else:
            self.target_col = target_col
            
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in {csv_path}. "
                           f"Available columns: {list(self.df.columns)}")
        
        self.df = self.df.dropna(subset=['sequence', self.target_col])
        
        self.df['mutation'] = self.df.apply(self._combine_mutations, axis=1)
        self.df['mutation'] = self.df['mutation'].apply(self._clean_mutation)
        
        initial_size = len(self.df)
        self.df = self.df[self.df['sequence'].str.len() > 0]
        removed = initial_size - len(self.df)
        if removed > 0:
            print(f"Removed {removed} rows with invalid sequences")
        
        self.df = self.df.reset_index(drop=True)
        
        if self.precompute_tokens:
            print("Precomputing tokenization for faster training...")
            self._precompute_tokenization()
        
        self._print_dataset_info()
        
    def _clean_sequence(self, sequence):
        """Clean protein sequence to only contain valid amino acids"""
        if pd.isna(sequence):
            return ""
        
        sequence = str(sequence).upper()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned = ''.join([char for char in sequence if char in valid_aa])
        return cleaned
    
    def _clean_mutation(self, mutation):
        """Clean mutation string"""
        if pd.isna(mutation) or mutation == 'WT':
            return 'WT'
        
        mutation = str(mutation).strip()
        mutation = re.sub(r'[^\w\s\-\>]', '', mutation)
        return mutation if mutation else 'WT'
    
    def _auto_detect_target_column(self):
        """Auto-detect target column by excluding fixed columns"""
        fixed_columns = {
            'sequence', 'mutation1', 'mutation2', 'mutation3', 'mutation4', 'mutation5', 'mutation6', 'mutation7', 'source', 'original_feature',
            'mutation', 'index', 'id'
        }
        
        all_columns = set(self.df.columns)
        fixed_columns_lower = {col.lower() for col in fixed_columns}
        potential_targets = [col for col in all_columns 
                           if col.lower() not in fixed_columns_lower]
        
        if len(potential_targets) == 1:
            target_col = potential_targets[0]
            print(f"Auto-detected target column: '{target_col}'")
            return target_col
        
        elif len(potential_targets) > 1:
            numeric_targets = [col for col in potential_targets
                             if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            if len(numeric_targets) == 1:
                target_col = numeric_targets[0]
                print(f"Auto-detected target column: '{target_col}' (only numeric)")
                return target_col
            elif len(numeric_targets) > 1:
                target_col = numeric_targets[0]
                print(f"Auto-detected target column: '{target_col}' (first numeric)")
                return target_col
            else:
                target_col = potential_targets[0]
                print(f"Auto-detected target column: '{target_col}' (first non-fixed)")
                return target_col
        
        else:
            raise ValueError(f"Could not auto-detect target column in {self.csv_path}")
    
    def _combine_mutations(self, row):
        """Combine mutation1 and mutation2 into a single mutation string"""
        mutations = []
        if pd.notna(row.get('mutation1', '')):
            mutations.append(str(row['mutation1']))
        if pd.notna(row.get('mutation2', '')):
            mutations.append(str(row['mutation2']))
        
        return ' '.join(mutations) if mutations else 'WT'
    
    def _highlight_mutation_in_sequence(self, sequence, mutation):
        """
        Highlight mutation in sequence using SEP tokens (which are in ProtBERT vocabulary).
        Format: {before} [SEP] {orig_aa} [SEP] {new_aa} [SEP] {after}
        """
        if mutation == 'WT':
            return sequence
        
        mutation_parts = mutation.split()
        highlighted_sequence = sequence
         
        for mut in mutation_parts:
            if len(mut) >= 3:
                orig_aa = mut[0]
                new_aa = mut[-1]
                position_str = mut[1:-1]
                
                if (orig_aa.isalpha() and new_aa.isalpha() and 
                    position_str.isdigit() and 
                    orig_aa in 'ACDEFGHIKLMNPQRSTVWY' and 
                    new_aa in 'ACDEFGHIKLMNPQRSTVWY'):
                    
                    position = int(position_str) - 1
                    
                    if ' ' in highlighted_sequence:
                        space_position = position * 2
                        if space_position >= len(highlighted_sequence):
                            continue
                        
                        if highlighted_sequence[space_position] == orig_aa:
                            before = highlighted_sequence[:space_position]
                            after = highlighted_sequence[space_position+2:]  # Skip AA + space
                            
                            highlighted_sequence = f"{before} [SEP] {orig_aa} [SEP] {new_aa} [SEP] {after}"
                        else:
                            print(f"Warning: Expected {orig_aa} at position {position+1}, found {highlighted_sequence[space_position]}")
                    else:
                        if 0 <= position < len(highlighted_sequence):
                            if highlighted_sequence[position] == orig_aa:
                                before = highlighted_sequence[:position]
                                after = highlighted_sequence[position+1:]
                                
                                highlighted_sequence = f"{before} [SEP] {orig_aa} [SEP] {new_aa} [SEP] {after}"
                            else:
                                print(f"Warning: Expected {orig_aa} at position {position+1}, found {highlighted_sequence[position]}")
        
        return highlighted_sequence
    ############
    def _precompute_tokenization(self):
        """Precompute all tokenization to speed up training"""
        self.tokenized_data = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            sequence = row['sequence']
            mutation = row['mutation']
            
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = 'prot_bert' 
            
            if model_name == 'prot_bert':
                spaced_sequence = ' '.join(list(sequence))
                
                if mutation != 'WT':
                    highlighted_sequence = self._highlight_mutation_in_sequence(spaced_sequence, mutation)
                    text = highlighted_sequence
                    # text = f"{spaced_sequence} [SEP] {mutation}"
                else:
                    text = spaced_sequence
            else:
                # Regular BERT format
                if mutation != 'WT':
                    highlighted_sequence = self._highlight_mutation_in_sequence(sequence, mutation)
                    text = highlighted_sequence
                    # text = f"{sequence} [SEP] {mutation}"
                else:
                    text = sequence
            
            # Truncate if too long
            if len(text) > self.max_length - 50:
                max_seq_len = self.max_length - 100 
                sequence = sequence[:max_seq_len]
                
                if model_name == 'prot_bert':
                    spaced_sequence = ' '.join(list(sequence))
                    if mutation != 'WT':
                        # highlighted_sequence = self._highlight_mutation_in_sequence(spaced_sequence, mutation)
                        # text = highlighted_sequence
                        text = f"{spaced_sequence} [SEP] {mutation}"
                    else:
                        text = spaced_sequence
                else:
                    if mutation != 'WT':
                        # highlighted_sequence = self._highlight_mutation_in_sequence(sequence, mutation)
                        # text = highlighted_sequence
                        text = f"{sequence} [SEP] {mutation}"
                    else:
                        text = sequence
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            # Validate tokens once during preprocessing
            input_ids = encoding['input_ids'].squeeze()
            if input_ids.max() >= self.tokenizer.vocab_size:
                raise ValueError(f"Token ID {input_ids.max()} exceeds vocab size {self.tokenizer.vocab_size}")
            
            self.tokenized_data.append({
                'input_ids': input_ids,
                'attention_mask': encoding['attention_mask'].squeeze(),
                'target': torch.tensor(float(row[self.target_col]), dtype=torch.float32)
            })
    ############
    def _print_dataset_info(self):
        """Print dataset information"""
        print(f"Loaded dataset from {os.path.basename(self.csv_path)}")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target column: '{self.target_col}'")
        print(f"Target stats: mean={self.df[self.target_col].mean():.4f}, std={self.df[self.target_col].std():.4f}")
    
    def __len__(self):
        return len(self.df)
    ##############
    def __getitem__(self, idx):
        if self.precompute_tokens:
            return self.tokenized_data[idx]
        else:
            row = self.df.iloc[idx]
            sequence = row['sequence']
            mutation = row['mutation']
            target = float(row[self.target_col])
            
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = 'prot_bert'
            
            if model_name == 'prot_bert':
                spaced_sequence = ' '.join(list(sequence))
                
                if mutation != 'WT':
                    highlighted_sequence = self._highlight_mutation_in_sequence(spaced_sequence, mutation)
                    text = highlighted_sequence
                    # text = f"{spaced_sequence} [SEP] {mutation}"
                else:
                    text = spaced_sequence
            else:
                if mutation != 'WT':
                    highlighted_sequence = self._highlight_mutation_in_sequence(sequence, mutation)
                    text = highlighted_sequence
                    # text = f"{sequence} [SEP] {mutation}"
                else:
                    text = sequence
            
            # Truncate if needed
            if len(text) > self.max_length - 50:
                max_seq_len = self.max_length - 100 
                sequence = sequence[:max_seq_len]
                
                if model_name == 'prot_bert':
                    spaced_sequence = ' '.join(list(sequence))
                    if mutation != 'WT':
                        # highlighted_sequence = self._highlight_mutation_in_sequence(spaced_sequence, mutation)
                        # text = highlighted_sequence
                        text = f"{spaced_sequence} [SEP] {mutation}"
                    else:
                        text = spaced_sequence
                else:
                    if mutation != 'WT':
                        # highlighted_sequence = self._highlight_mutation_in_sequence(sequence, mutation)
                        # text = highlighted_sequence
                        text = f"{sequence} [SEP] {mutation}"
                    else:
                        text = sequence
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'target': torch.tensor(target, dtype=torch.float32)
            }
    ################
    def get_task_info(self):
        """Return information about this task"""
        return {
            'name': os.path.basename(self.csv_path) if hasattr(self, 'csv_path') else 'unknown',
            'size': len(self.df),
            'target_col': self.target_col,
            'target_stats': {
                'mean': self.df[self.target_col].mean(),
                'std': self.df[self.target_col].std(),
                'min': self.df[self.target_col].min(),
                'max': self.df[self.target_col].max()
            }
        }


class ProteinDataLoader:
    """Utility class to load multiple protein datasets for meta-learning."""
    
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_task_datasets(self, file_paths, target_cols=None):
        """Load multiple CSV files as different tasks"""
        tasks = []
        
        if target_cols is None:
            target_cols = [None] * len(file_paths)
        elif len(target_cols) != len(file_paths):
            raise ValueError(f"target_cols length must match file_paths length")
        
        for i, (file_path, target_col) in enumerate(zip(file_paths, target_cols)):
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist, skipping...")
                continue
                
            try:
                dataset = ProteinDataset(file_path, self.tokenizer, self.max_length, target_col)
                
                task = {
                    'name': f"Task_{i}_{os.path.basename(file_path).replace('.csv', '')}",
                    'file_path': file_path,
                    'dataset': dataset,
                    'target_col': dataset.target_col,
                    'info': dataset.get_task_info()
                }
                
                tasks.append(task)
                print(f"✓ Loaded task: {task['name']} with {len(dataset)} samples")
                
            except Exception as e:
                print(f"✗ Error loading {file_path}: {str(e)}")
                continue
        
        return tasks
    
    # def create_train_test_split(self, dataset, train_ratio=0.8):
    #     """Split a dataset into train and test portions"""
    #     df = dataset.df.copy()
    #     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #     split_idx = int(len(df) * train_ratio)
        
    #     train_df = df[:split_idx]
    #     test_df = df[split_idx:]
        
    #     # Create temporary CSV files
    #     train_path = '/tmp/train_split.csv'
    #     test_path = '/tmp/test_split.csv'
        
    #     train_df.to_csv(train_path, index=False)
    #     test_df.to_csv(test_path, index=False)
        
    #     train_dataset = ProteinDataset(train_path, dataset.tokenizer, dataset.max_length, dataset.target_col)
    #     test_dataset = ProteinDataset(test_path, dataset.tokenizer, dataset.max_length, dataset.target_col)
        
    #     return train_dataset, test_dataset

def validate_csv_format(csv_path, required_columns=None):
    """Validate CSV file format"""
    if required_columns is None:
        required_columns = ['sequence']
    
    try:
        df = pd.read_csv(csv_path)
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return {'valid': False, 'error': f"Missing columns: {missing_cols}"}
        
        if len(df) == 0:
            return {'valid': False, 'error': "Empty dataframe"}
        
        target_col = _auto_detect_target_helper(df)
        
        critical_cols = ['sequence']
        if target_col:
            critical_cols.append(target_col)
            
        for col in critical_cols:
            if col in df.columns and df[col].isna().all():
                return {'valid': False, 'error': f"All values missing in {col}"}
        
        return {
            'valid': True, 
            'target_col': target_col,
            'shape': df.shape,
            'columns': list(df.columns)
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def _auto_detect_target_helper(df):
    """Helper function for target column auto-detection"""
    fixed_columns = {
        'sequence', 'mutation1', 'mutation2', 'mutation3', 'mutation4', 'mutation5', 'mutation6', 'mutation7', 'source', 'original_feature',
        'mutation', 'index', 'id'
    }
    
    all_columns = set(df.columns)
    fixed_columns_lower = {col.lower() for col in fixed_columns}
    potential_targets = [col for col in all_columns 
                       if col.lower() not in fixed_columns_lower]
    
    if len(potential_targets) >= 1:
        numeric_targets = [col for col in potential_targets 
                         if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        if numeric_targets:
            return numeric_targets[0]
        else:
            return potential_targets[0]
    
    return None



import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import random
import os
import argparse
import json
import time
import copy
from model import ProteinTransformer
from data import ProteinDataset, ProteinDataLoader, validate_csv_format
from meta_learning import MAML

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_optimal_settings():
    """Auto-detect optimal training settings based on GPU(s)"""
    if not torch.cuda.is_available():
        return {
            'batch_size': 8,
            'num_workers': 2,
            'pin_memory': False,
            'use_mixed_precision': False
        }
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"Detected GPU: {torch.cuda.get_device_name()} ({gpu_memory_gb:.1f} GB)")
    
    if gpu_memory_gb >= 24:    
        settings = {'batch_size': 32, 'num_workers': 8, 'pin_memory': True, 'use_mixed_precision': True}
    elif gpu_memory_gb >= 16: 
        settings = {'batch_size': 24, 'num_workers': 6, 'pin_memory': True, 'use_mixed_precision': True}
    elif gpu_memory_gb >= 12:
        settings = {'batch_size': 16, 'num_workers': 4, 'pin_memory': True, 'use_mixed_precision': True}
    elif gpu_memory_gb >= 8:   
        settings = {'batch_size': 12, 'num_workers': 4, 'pin_memory': True, 'use_mixed_precision': True}
    else:      
        settings = {'batch_size': 8, 'num_workers': 2, 'pin_memory': True, 'use_mixed_precision': True}
    return settings

def fast_pretrain(model, dataset, epochs=10, freeze_bert=True, use_scheduler=True):
    """pretraining"""
    print(f"\n FAST PRETRAINING")
    print(f"Dataset size: {len(dataset)} samples")
    
    settings = get_optimal_settings()
    
    if freeze_bert:
        print(" Freezing BERT backbone...")
        for param in model.bert.parameters():
            param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.predictor.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Training {trainable_params:,} / {total_params:,} parameters ({trainable_params/total_params*100:.1f}%)")
    
    train_loader = DataLoader(
        dataset,
        batch_size=settings['batch_size'],
        shuffle=True,
        num_workers=settings['num_workers'],
        pin_memory=settings['pin_memory'],
        persistent_workers=settings['num_workers'] > 0,
        prefetch_factor=2,
        drop_last=True
    )
    
    if freeze_bert:
        optimizer = AdamW(model.predictor.parameters(), lr=5e-3, weight_decay=0.01)
        max_lr = 3e-3
    else:
        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        max_lr = 5e-5
    
    scheduler = None
    if use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    
    scaler = GradScaler() if settings['use_mixed_precision'] else None
    
    print(f" Training setup:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {settings['batch_size']}")
    print(f"   Steps per epoch: {len(train_loader)}")
    print(f"   Mixed precision: {settings['use_mixed_precision']}")
    print(f"   Learning rate: {max_lr}")
    
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            targets = batch['target'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    predictions = model(input_ids, attention_mask)
                    loss = F.mse_loss(predictions.squeeze(), targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if scheduler:
                    scheduler.step()
                scaler.update()
            else:
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions.squeeze(), targets)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                lr_display = scheduler.get_last_lr()[0] if scheduler else max_lr
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_display:.2e}"
                })
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Fast pretraining completed in {total_time:.1f}s!")
    print(f"Average time per epoch: {total_time/(epoch+1):.1f}s")
    
    return model, losses

def finetune_model(base_model, dataset, epochs=20, freeze_bert=False, use_scheduler=True, use_wandb=False):
    """
    Finetune a model on a specific dataset
    """
    print(f"\n FINETUNING MODEL")
    print(f"Target column: {dataset.target_col}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create a copy of the model to avoid modifying the original
    model = copy.deepcopy(base_model)
    
    settings = get_optimal_settings()
    
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    train_loader = DataLoader(
        dataset,
        batch_size=settings['batch_size'] // 2,  # Smaller batch for finetuning
        shuffle=True,
        num_workers=settings['num_workers'],
        pin_memory=settings['pin_memory'],
        persistent_workers=settings['num_workers'] > 0,
        prefetch_factor=2,
        drop_last=True
    )
    
    # Lower learning rate for finetuning
    if freeze_bert:
        optimizer = AdamW(model.predictor.parameters(), lr=5e-4, weight_decay=0.01)
        max_lr = 1e-3
    else:
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        max_lr = 1e-5
    
    scheduler = None
    if use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    
    scaler = GradScaler() if settings['use_mixed_precision'] else None
    
    print(f" Finetuning setup:")
    print(f" Epochs: {epochs}")
    print(f" Batch size: {settings['batch_size'] // 2}")
    print(f" Learning rate: {max_lr}")
    print(f" Freeze BERT: {freeze_bert}")
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Finetune Epoch {epoch+1}/{epochs}",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            targets = batch['target'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    predictions = model(input_ids, attention_mask)
                    loss = F.mse_loss(predictions.squeeze(), targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if scheduler:
                    scheduler.step()
                scaler.update()
            else:
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions.squeeze(), targets)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                lr_display = scheduler.get_last_lr()[0] if scheduler else max_lr
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_display:.2e}"
                })
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"Finetune Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Log to wandb if enabled
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "finetune/epoch": epoch + 1,
                    "finetune/loss": avg_loss
                })
            except ImportError:
                pass
    return model, losses

def validate_datasets(file_paths):
    """Quick validation of all datasets"""
    print("\nVALIDATING INPUT DATASETS...")
    for file_path in file_paths:
        result = validate_csv_format(file_path)
        if not result['valid']:
            print(f" Dataset validation failed for {file_path}: {result['error']}")
            return False
        else:
            print(f"{os.path.basename(file_path)}: {result['shape'][0]} samples")
    return True

def quick_evaluate_model(model, dataset, batch_size=16):
    """Quick model evaluation with fixed dimension handling"""
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    model.eval()
    total_loss = 0
    predictions_list = []
    targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            targets = batch['target'].to(DEVICE, non_blocking=True)
            
            predictions = model(input_ids, attention_mask)
            
            # Fix dimension mismatch
            predictions_flat = predictions.squeeze()
            if predictions_flat.dim() == 0:
                predictions_flat = predictions_flat.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
                
            loss = F.mse_loss(predictions_flat, targets)
            total_loss += loss.item()
            predictions_list.extend(predictions_flat.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(test_loader)
    return {
        'mse': avg_loss
    }

def compare_all_approaches(pretrained_model, test_tasks, train_tasks, finetune_epochs=20, freeze_bert_finetune=True, maml_model=None, support_size=32, query_size=32, use_wandb=False):
    """Compare pretrained vs finetuned vs MAML approaches"""
    print("\n" + "="*70)
    print("COMPARING ALL APPROACHES ON TEST TASKS")
    print("="*70)
    results = {}
    for train_task, task in zip(train_tasks, test_tasks):
        print(f"Evaluating task: {task['name']}")
        # 1. Evaluate pretrained model (zero-shot)
        print(" Testing pretrained model")
        pretrained_metrics = quick_evaluate_model(pretrained_model, task['dataset'])
        print(f"Pretrained MSE: {pretrained_metrics['mse']:.4f}")
        # 2. Finetune and evaluate
        print(f" Finetuning model on task: {train_task['name']}...")
        finetune_start = time.time()
        finetuned_model, finetune_losses = finetune_model(
            pretrained_model, 
            train_task['dataset'], 
            epochs=finetune_epochs,
            freeze_bert=freeze_bert_finetune,
            use_wandb=use_wandb
        )
        finetune_time = time.time() - finetune_start
        finetuned_metrics = quick_evaluate_model(finetuned_model, task['dataset'])
        print(f"Finetuned MSE: {finetuned_metrics['mse']:.4f}")
        finetuned_metrics['finetune_time'] = finetune_time
        finetuned_metrics['finetune_data_size'] = len(train_task['dataset'])
        
        if maml_model:
            maml_metrics = maml_model.meta_evaluate(task, support_size=support_size, query_size=query_size)
            print(f"MAML MSE: {maml_metrics['mse']:.4f}")
        results[task['name']] = {
            'pretrained': pretrained_metrics,
            'finetuned': finetuned_metrics
        }
        if maml_model:
            results[task['name']]['maml'] = maml_metrics
            
        pretrain_to_finetune = ((pretrained_metrics['mse'] - finetuned_metrics['mse']) / pretrained_metrics['mse']) * 100
        print(f"Finetuning improvement: {pretrain_to_finetune:.1f}%")
        if maml_model:
            pretrain_to_maml = ((pretrained_metrics['mse'] - maml_metrics['mse']) / pretrained_metrics['mse']) * 100
            print(f"MAML improvement: {pretrain_to_maml:.1f}%")
    return results

def load_pretrained_model(model_path, model_name='prot_bert'):
    """Load a pretrained model from file"""
    print(f"Loading pretrained model from {model_path}")
    model = ProteinTransformer(num_outputs=1, dropout=0.1, model_name=model_name)
    model.to(DEVICE)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f" Error loading pretrained model: {e}")
        return None

def get_tokenizer_for_model(model_name):
    """Get appropriate tokenizer for model"""
    if model_name == 'prot_bert':
        return BertTokenizer.from_pretrained('Rostlab/prot_bert')
    elif model_name == 'esm2':
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    else: 
        return BertTokenizer.from_pretrained('bert-base-uncased')

def save_all_results(results, save_dir='./results'):
    """Save all results to files"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'all_approaches_comparison.json'), 'w') as f:
        serializable_results = {}
        for task_name, task_results in results.items():
            serializable_results[task_name] = {
                'pretrained': {k: float(v) for k, v in task_results['pretrained'].items()},
                'finetuned': {k: float(v) for k, v in task_results['finetuned'].items()}
            }
            if 'maml' in task_results:
                serializable_results[task_name]['maml'] = {k: float(v) for k, v in task_results['maml'].items()}
        json.dump(serializable_results, f, indent=2)
    print(f"All results saved to {save_dir}")

def unseen_task_comparison(pretrained_model, datasets, save_dir='./results', finetune_epochs=20, maml_epochs=30, support_size=16, query_size=16, use_wandb=False):
    """
    Systematic comparison: For each dataset, compare finetuning vs MAML to see if MAML is better on unseen datasets
    Args:
        pretrained_model: The pretrained model
        datasets: List of dataset dictionaries with 'name', 'train_dataset', 'test_dataset', 'info'
        save_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("SYSTEMATIC COMPARISON: FINETUNING vs MAML")
    print("="*70)
    results = {}
    for i, target_dataset in enumerate(datasets):
        print(f"\ Dataset {i+1}/{len(datasets)}: {target_dataset['name']}")
        other_datasets = [d for j, d in enumerate(datasets) if j != i]
        print(f"  Target dataset: {target_dataset['name']}")
        print(f"  Target train size: {len(target_dataset['train_dataset'])}")
        print(f"  Target test size: {len(target_dataset['test_dataset'])}")
        print(f"  MAML training datasets: {[d['name'] for d in other_datasets]}")
        print(f"  MAML total training size: {sum(len(d['train_dataset']) + len(d['test_dataset']) for d in other_datasets)}")
        
        # 1. FINETUNING - Train on the SAME dataset's training part
        print(f"\n FINETUNING on {target_dataset['name']}")
        print(f"   Training on: {target_dataset['name']}_train.csv")
        print(f"   Testing on:  {target_dataset['name']}_test.csv")
        
        finetune_start = time.time()
        finetuned_model, finetune_losses = finetune_model(
            pretrained_model, 
            target_dataset['train_dataset'],
            epochs=finetune_epochs,
            freeze_bert=False,
            use_wandb=use_wandb
        )
        total_finetune_time = time.time() - finetune_start
        finetuned_metrics = quick_evaluate_model(finetuned_model, target_dataset['test_dataset'])
        print(f"   Finetuning completed:")
        print(f"   Training time: {total_finetune_time:.2f}s")
        print(f"   Training dataset size: {len(target_dataset['train_dataset'])}")
        print(f"   Test MSE: {finetuned_metrics['mse']:.4f}")
        print(f"   MAML training on OTHER datasets")
        print(f"   Training on: {[d['name'] for d in other_datasets]}")
        print(f"   Testing on:  {target_dataset['name']}_test.csv")
        
        maml_start = time.time()
        maml = MAML(
            model=copy.deepcopy(pretrained_model),
            inner_lr=0.01,
            meta_lr=0.001,
            first_order=True,
            num_adaptation_steps=5
        )
        maml_training_tasks = []
        for dataset in other_datasets:
            combined_dataset = torch.utils.data.ConcatDataset([
                dataset['train_dataset'], 
                dataset['test_dataset']
            ])
            maml_training_tasks.append({
                'name': dataset['name'],
                'dataset': combined_dataset,
                'info': {
                    'size': len(combined_dataset),
                    'target_col': dataset['info']['target_col']
                }
            })
        maml_results = maml.meta_train(
            train_tasks=maml_training_tasks,
            num_epochs=maml_epochs,
            support_size=support_size,
            query_size=query_size,
            batch_size=4,
            evaluate_every=10,
            patience=10
        )
        total_maml_time = time.time() - maml_start
        maml_metrics = maml.meta_evaluate(
            {'name': target_dataset['name'], 'dataset': target_dataset['test_dataset']},
            support_size=support_size,
            query_size=query_size
        )
        print(f"MAML completed:")
        print(f"Training time: {total_maml_time:.2f}s")
        print(f"Training dataset size: {sum(len(d['train_dataset']) + len(d['test_dataset']) for d in other_datasets)} ({[d['name'] for d in other_datasets]})")
        for dataset in other_datasets:
            total_size = len(dataset['train_dataset']) + len(dataset['test_dataset'])
            print(f"{dataset['name']}: {total_size} samples")
        print(f"Test MSE: {maml_metrics['mse']:.4f}")
        
        results[target_dataset['name']] = {
            'finetuning': {
                'mse': float(finetuned_metrics['mse']),
                'training_time': float(total_finetune_time),
                'training_dataset_size': len(target_dataset['train_dataset']),
                'finetune_epochs': finetune_epochs,
                'training_dataset': target_dataset['name']  # Same dataset
            },
            'maml': {
                'mse': float(maml_metrics['mse']),
                'training_time': float(total_maml_time),
                'training_dataset_size': sum(len(d['train_dataset']) + len(d['test_dataset']) for d in other_datasets),
                'maml_epochs': maml_epochs,
                'training_datasets': [d['name'] for d in other_datasets],  # Other datasets
                'training_dataset_sizes': {d['name']: len(d['train_dataset']) + len(d['test_dataset']) for d in other_datasets}  # Individual sizes
            },
            'dataset_info': {
                'target_dataset': target_dataset['name'],
                'target_train_size': len(target_dataset['train_dataset']),
                'target_test_size': len(target_dataset['test_dataset']),
                'maml_training_datasets': [d['name'] for d in other_datasets]
            }
        }
        print(f"COMPARISON for {target_dataset['name']}:")
        print(f"Finetuning: MSE={finetuned_metrics['mse']:.4f}, Time={total_finetune_time:.1f}s, Data={len(target_dataset['train_dataset'])} ({target_dataset['name']}_train)")
        print(f"MAML:       MSE={maml_metrics['mse']:.4f}, Time={total_maml_time:.1f}s, Data={sum(len(d['train_dataset']) + len(d['test_dataset']) for d in other_datasets)} ({[d['name'] for d in other_datasets]})")
        
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'unseen_task_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Unseen task comparison completed!")
    print(f"Results saved to {save_dir}/unseen_task_comparison.json")

    print(f"OVERALL SUMMARY:")
    finetune_mses = [r['finetuning']['mse'] for r in results.values()]
    maml_mses = [r['maml']['mse'] for r in results.values()]
    finetune_times = [r['finetuning']['training_time'] for r in results.values()]
    maml_times = [r['maml']['training_time'] for r in results.values()]
    print(f"   Average Finetuning MSE: {np.mean(finetune_mses):.4f}")
    print(f"   Average MAML MSE: {np.mean(maml_mses):.4f}")
    print(f"   Average Finetuning Time: {np.mean(finetune_times):.1f}s")
    print(f"   Average MAML Time: {np.mean(maml_times):.1f}s")
    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced Protein Property Prediction with Pretrain/Finetune/MAML')
    parser.add_argument('--initial_task', type=str, required=True,
                       help='Path to initial task CSV file for pretraining')
    parser.add_argument('--meta_learning_tasks', type=str, nargs='*', default=[],
                       help='Paths to CSV files for meta-learning tasks')
    parser.add_argument('--test_tasks', type=str, nargs='*', default=[],
                       help='Paths to CSV files for testing (optional)')
    parser.add_argument('--finetune_train_tasks', type=str, nargs='*', default=[],
                       help='Paths to CSV files for training (optional)')
    parser.add_argument('--model_name', type=str, default='prot_bert',
                       choices=['bert-base-uncased', 'prot_bert', 'esm2'],
                       help='Model to use (default: prot_bert)')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                       help='Skip pretraining phase and load existing model')
    parser.add_argument('--pretrained_model_path', type=str, default='lightning_results/lightning_pretrained_model.pt',
                       help='Path to pretrained model file')
    parser.add_argument('--run_maml', action='store_true', default=False,
                       help='Run MAML meta-learning (requires meta_learning_tasks)')
    parser.add_argument('--run_finetune', action='store_true', default=False,
                       help='Run finetuning comparison')
    parser.add_argument('--unseen_task_comparison', action='store_true', default=False,
                       help='Run unseen task comparison: finetuning vs MAML for each dataset')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='protein-maml-comparison',
                       help='Weights & Biases project name')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                       help='Number of pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                       help='Number of finetuning epochs')
    parser.add_argument('--maml_epochs', type=int, default=30,
                       help='Number of MAML epochs')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate for MAML')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                       help='Meta learning rate for MAML')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                       help='Number of adaptation steps in MAML')
    parser.add_argument('--support_size', type=int, default=8,
                       help='Support set size for meta-learning')
    parser.add_argument('--query_size', type=int, default=8,
                       help='Query set size for meta-learning')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                       help='Freeze BERT parameters for faster training')
    parser.add_argument('--freeze_bert_finetune', action='store_true', default=False,
                       help='Freeze BERT during finetuning (default: True)')
    parser.add_argument('--no_scheduler', action='store_true', default=False,
                       help='Disable learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    args = parser.parse_args()
    set_seeds(args.seed)
    print("ENHANCED PROTEIN PROPERTY PREDICTION PIPELINE")
    print(f"Device: {DEVICE}")
    print(f"Skip pretrain: {args.skip_pretrain}")
    print(f"Run MAML: {args.run_maml}")
    print(f"Run finetune: {args.run_finetune}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nInitializing tokenizer for {args.model_name}...")
    tokenizer = get_tokenizer_for_model(args.model_name)
    print("\n" + "="*50)
    if args.skip_pretrain:
        print("PHASE 1: LOADING PRETRAINED MODEL")
        pretrained_model = load_pretrained_model(args.pretrained_model_path, args.model_name)
        if pretrained_model is None:
            print("Failed to load pretrained model. Exiting.")
            return
    else:
        print("PHASE 1: PRETRAINING")
        initial_dataset = ProteinDataset(
            args.initial_task,
            tokenizer,
            args.max_length,
            precompute_tokens=True,
            model_name=args.model_name
        )
        model = ProteinTransformer(num_outputs=1, dropout=0.1, model_name=args.model_name)
        model = model.to(DEVICE)
        pretrained_model, pretrain_losses = fast_pretrain(
            model=model,
            dataset=initial_dataset,
            epochs=args.pretrain_epochs,
            freeze_bert=args.freeze_bert,
            use_scheduler=not args.no_scheduler
        )
        torch.save(pretrained_model.state_dict(), os.path.join(args.save_dir, "lightning_pretrained_model.pt"))
        print("Pretrained model saved!")
         
    print("="*50)
    all_results = {}
    maml_results = {}

    if args.run_maml and args.meta_learning_tasks:
        print("PHASE 2: META-LEARNING")
        data_loader = ProteinDataLoader(tokenizer, args.max_length)
        meta_tasks = data_loader.load_task_datasets(args.meta_learning_tasks)
        
        if meta_tasks:
            print(f"\nLoaded {len(meta_tasks)} meta-learning tasks:")
            for task in meta_tasks:
                info = task['info']
                print(f"  - {task['name']}: {info['size']} samples, target='{info['target_col']}'")
            maml = MAML(
                model=copy.deepcopy(pretrained_model),
                inner_lr=args.inner_lr,
                meta_lr=args.meta_lr,
                first_order=True,
                num_adaptation_steps=args.adaptation_steps
            )
            try:
                cleanup_memory()
                meta_start_time = time.time()
                maml_results = maml.meta_train(
                    train_tasks=meta_tasks,
                    num_epochs=args.maml_epochs,
                    support_size=args.support_size,
                    query_size=args.query_size,
                    batch_size=4,
                    evaluate_every=10,
                    patience=15
                )
                total_maml_time = time.time() - meta_start_time
                torch.save(maml.model.state_dict(), os.path.join(args.save_dir, "maml_model.pt"))
                print("MAML model saved!")
                if maml_results:
                    train_task_sizes = {task['name']: task['info']['size'] for task in meta_tasks}
                    with open(os.path.join(args.save_dir, 'maml_training_results.json'), 'w') as f:
                        serializable_maml = {
                            'meta_losses': [float(x) for x in maml_results.get('meta_losses', [])],
                            'pre_adapt_losses': [float(x) for x in maml_results.get('pre_adapt_losses', [])],
                            'post_adapt_losses': [float(x) for x in maml_results.get('post_adapt_losses', [])],
                            'validation_losses': [float(x) for x in maml_results.get('validation_losses', [])],
                            'total_maml_time': float(total_maml_time),
                            'train_task_sizes': train_task_sizes
                        }
                        json.dump(serializable_maml, f, indent=2)
                    print("MAML training results saved!")
                
            except Exception as e:
                print(f"MAML training failed: {e}")
                maml_results = {}
        else:
            print("No valid meta-learning tasks found!")    

    if args.test_tasks:
        print("PHASE 3: EVALUATION")
        data_loader = ProteinDataLoader(tokenizer, args.max_length)
        test_tasks = data_loader.load_task_datasets(args.test_tasks)
        finetune_train_tasks = data_loader.load_task_datasets(args.finetune_train_tasks)
        
        if test_tasks:
            print(f"\nLoaded {len(test_tasks)} test tasks:")
            for task in test_tasks:
                info = task['info']
                print(f"  - {task['name']}: {info['size']} samples")
            
            # Compare all approaches
            all_results = compare_all_approaches(
                pretrained_model, 
                test_tasks,
                finetune_train_tasks,
                finetune_epochs=args.finetune_epochs,
                freeze_bert_finetune=args.freeze_bert_finetune,
                maml_model=maml,
                support_size=args.support_size,
                query_size=args.query_size
            )
    # SAVE RESULTS AND SUMMARY
    save_all_results(all_results, args.save_dir)
    
    # PHASE 4: SYSTEMATIC COMPARISON (if requested)
    if args.unseen_task_comparison and args.test_tasks:
        print("PHASE 4: UNSEEN TASK COMPARISON")
        data_loader = ProteinDataLoader(tokenizer, args.max_length)

        dataset_names = []
        for test_path in args.test_tasks:
            dataset_name = os.path.basename(test_path).replace('_test.csv', '')
            dataset_names.append(dataset_name)
            
        print(f"Loading {len(dataset_names)} datasets for systematic comparison...")
        systematic_datasets = []
        
        for dataset_name in dataset_names:
            try:
                train_path = os.path.join('preprocess_data', f'{dataset_name}_train.csv')
                test_path = os.path.join('preprocess_data', f'{dataset_name}_test.csv')
                train_dataset = ProteinDataset(
                    train_path,
                    tokenizer,
                    args.max_length,
                    precompute_tokens=True,
                    model_name=args.model_name
                )
                test_dataset = ProteinDataset(
                    test_path,
                    tokenizer,
                    args.max_length,
                    precompute_tokens=True,
                    model_name=args.model_name
                )
                systematic_datasets.append({
                    'name': dataset_name,
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset,
                    'info': {
                        'size': len(train_dataset) + len(test_dataset),
                        'target_col': train_dataset.target_col
                    }
                })
                print(f"{dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test")
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue
        print(f"Total datasets: {len(systematic_datasets)}")
        systematic_results = unseen_task_comparison(
            pretrained_model, 
            systematic_datasets, 
            args.save_dir,
            finetune_epochs=args.finetune_epochs,
            maml_epochs=args.maml_epochs,
            support_size=args.support_size,
            query_size=args.query_size
        )
        with open(os.path.join(args.save_dir, 'unseen_task_comparison_summary.json'), 'w') as f:
            summary = {
                'setup': {
                    'total_datasets': len(systematic_datasets),
                    'datasets': [d['name'] for d in systematic_datasets],
                    'pretrain_epochs': args.pretrain_epochs,
                    'finetune_epochs': args.finetune_epochs,
                    'maml_epochs': args.maml_epochs,
                    'support_size': args.support_size,
                    'query_size': args.query_size,
                    'seed': args.seed,
                    'data_folder': 'preprocess_data'
                },
                'results': systematic_results
            }
            json.dump(summary, f, indent=2)
        print(f"Unseen task comparison summary saved!")

if __name__ == "__main__":
    main()
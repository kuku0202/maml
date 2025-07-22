import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import gc
import psutil

try:
    from utils import get_gpu_memory_info, estimate_memory_usage, print_model_info
except ImportError:
    def get_gpu_memory_info():
        if not torch.cuda.is_available():
            return "CUDA not available"
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            return {
                'total_gb': total_memory,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'free_gb': total_memory - cached
            }
        except Exception as e:
            return f"Error getting GPU info: {e}"
    
    def estimate_memory_usage(dataset_size, max_length=512, batch_size=8):
        return {
            'model_parameters_gb': 0.4,  # Rough estimate
            'batch_memory_gb': 0.1,
            'total_memory_gb': 0.5,
            'recommended_gpu_memory_gb': 1.0
        }
    
    def print_model_info(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

# Ensure that the device is set (either CUDA or CPU)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_memory_usage():
    """Get current memory usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    else:
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        return f"CPU: {memory_gb:.2f}GB"

def safe_memory_cleanup():
    """Safely clean up memory without causing gradient detachment issues"""
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear any remaining gradients
    torch.autograd.set_detect_anomaly(False)

def check_memory_safety(model, batch_size, support_size, query_size, max_length=512):
    """
    Check if the current configuration is safe for memory usage
    
    Args:
        model: The model to check
        batch_size: Training batch size
        support_size: Support set size
        query_size: Query set size
        max_length: Maximum sequence length
        
    Returns:
        dict: Memory safety information
    """
    # Get model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    model_memory_gb = total_params * 4 / 1e9  # float32
    
    # Estimate memory usage
    total_samples = support_size + query_size
    memory_estimates = estimate_memory_usage(
        dataset_size=total_samples,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Get current GPU memory
    gpu_info = get_gpu_memory_info()
    
    safety_info = {
        'model_memory_gb': model_memory_gb,
        'estimated_batch_memory_gb': memory_estimates['batch_memory_gb'],
        'total_estimated_memory_gb': model_memory_gb + memory_estimates['batch_memory_gb'],
        'gpu_info': gpu_info,
        'safe': True,
        'warnings': []
    }
    
    if isinstance(gpu_info, dict):
        available_memory = gpu_info['free_gb']
        required_memory = safety_info['total_estimated_memory_gb']
        
        if required_memory > available_memory * 0.8:  # 80% safety margin
            safety_info['safe'] = False
            safety_info['warnings'].append(f"Estimated memory ({required_memory:.2f}GB) exceeds 80% of available GPU memory ({available_memory:.2f}GB)")
        
        if required_memory > available_memory:
            safety_info['warnings'].append(f"CRITICAL: Estimated memory ({required_memory:.2f}GB) exceeds available GPU memory ({available_memory:.2f}GB)")
    
    return safety_info

class MAML:
    """
    Improved Model-Agnostic Meta-Learning (MAML) implementation for protein property prediction.
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, first_order=True, num_adaptation_steps=5):
        """
        Initialize MAML with memory-efficient defaults.
        
        Args:
            model: The neural network model
            inner_lr: Learning rate for inner loop adaptation
            meta_lr: Learning rate for meta-optimization
            first_order: Use first-order approximation (True for memory efficiency)
            num_adaptation_steps: Number of adaptation steps in inner loop
                                 (Lower values = less memory usage)
        
        Memory optimization tips:
        - Use first_order=True (default) for significant memory savings
        - Reduce num_adaptation_steps (3-5 is usually sufficient)
        - Use smaller batch_size in meta_train
        - Monitor memory usage with get_memory_usage()
        """
        self.model = model.to(DEVICE)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.first_order = first_order  # True by default for memory efficiency
        self.num_adaptation_steps = num_adaptation_steps
        self.meta_optimizer = Adam(self.model.parameters(), lr=self.meta_lr)
        
        print(f"MAML initialized with:")
        print(f"  Inner LR: {inner_lr}")
        print(f"  Meta LR: {meta_lr}")
        print(f"  Adaptation steps: {num_adaptation_steps}")
        print(f"  First order: {first_order} (memory efficient)")
        print(f"  Device: {DEVICE}")
        
        # Print model information
        print_model_info(self.model)
        
        # Check initial memory status
        print(f"Initial memory usage: {get_memory_usage()}")
        
        # Add gradient norm tracking for debugging
        self.gradient_norms = []
    
    def create_support_query_split(self, dataset, support_size=16, query_size=16):
        """
        Split a dataset into support and query sets for meta-learning
        
        Args:
            dataset: PyTorch dataset
            support_size: Number of examples for support set
            query_size: Number of examples for query set
            
        Returns:
            tuple: (support_subset, query_subset)
        """
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Ensure we don't exceed dataset size
        actual_support_size = min(support_size, total_size // 2)
        actual_query_size = min(query_size, total_size - actual_support_size)
        
        support_indices = indices[:actual_support_size]
        query_indices = indices[actual_support_size:actual_support_size + actual_query_size]
        
        support_subset = Subset(dataset, support_indices)
        query_subset = Subset(dataset, query_indices)
        
        return support_subset, query_subset
    
    def validation_adapt(self, support_loader):
        """
        Performs adaptation for validation without gradient computation issues.
        This is a simplified version that doesn't create computational graphs.
        """
        # Clone the model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        # Collect all support data
        all_input_ids = []
        all_attention_masks = []
        all_targets = []
        
        for batch in support_loader:
            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            all_targets.append(batch['target'])
        
        # Concatenate all batches
        input_ids = torch.cat(all_input_ids, dim=0).to(DEVICE)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(DEVICE)
        targets = torch.cat(all_targets, dim=0).to(DEVICE)
        
        # Compute initial loss before adaptation
        with torch.no_grad():
            initial_predictions = adapted_model(input_ids, attention_mask)
            initial_loss = F.mse_loss(initial_predictions.squeeze(), targets)
        
        # Simple adaptation without gradient computation for validation
        # We'll just return the initial model and loss
        return adapted_model, initial_loss.item(), initial_loss.item()
    
    def adapt(self, support_loader, first_order=None):
        """
        Performs the inner loop update for a single task using the support set.
        
        Args:
            support_loader: DataLoader for support set
            first_order: Whether to use first-order approximation
            
        Returns:
            tuple: (adapted_model, initial_loss, final_loss)
        """
        if first_order is None:
            first_order = self.first_order
        
        # More memory-efficient model cloning
        adapted_model = type(self.model)()
        adapted_model.load_state_dict(self.model.state_dict())
        adapted_model.to(DEVICE)
        adapted_model.train()
        
        # Collect all support data
        all_input_ids = []
        all_attention_masks = []
        all_targets = []
        
        for batch in support_loader:
            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            all_targets.append(batch['target'])
        
        # Concatenate all batches
        input_ids = torch.cat(all_input_ids, dim=0).to(DEVICE)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(DEVICE)
        targets = torch.cat(all_targets, dim=0).to(DEVICE)
        
        # Compute initial loss before adaptation
        with torch.no_grad():
            initial_predictions = adapted_model(input_ids, attention_mask)
            initial_loss = F.mse_loss(initial_predictions.squeeze(), targets)
        
        # Inner loop adaptation steps
        for step in range(self.num_adaptation_steps):
            predictions = adapted_model(input_ids, attention_mask)
            loss = F.mse_loss(predictions.squeeze(), targets)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                adapted_model.parameters(), 
                create_graph=not first_order,
                allow_unused=True,
                retain_graph=not first_order
            )
            
            # Manual parameter update
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - self.inner_lr * grad
        
        # Compute final loss after adaptation
        with torch.no_grad():
            final_predictions = adapted_model(input_ids, attention_mask)
            final_loss = F.mse_loss(final_predictions.squeeze(), targets)
        
        return adapted_model, initial_loss.item(), final_loss.item()
    
    def meta_train(self, train_tasks, num_epochs=50, support_size=16, query_size=16, 
                   batch_size=4, evaluate_every=10, patience=10):
        """
        Execute the meta-training loop with proper support/query splits.
        
        Args:
            train_tasks: List of task dictionaries
            num_epochs: Number of meta-training epochs
            support_size: Size of support set for each task
            query_size: Size of query set for each task
            batch_size: Batch size for DataLoaders
            evaluate_every: Evaluate every N epochs
            patience: Early stopping patience
            
        Returns:
            dict: Training results and metrics
        """
        # Memory safety check before training
        print("Performing memory safety check...")
        safety_info = check_memory_safety(self.model, batch_size, support_size, query_size)
        
        if not safety_info['safe']:
            print("⚠️  WARNING: Memory configuration may be unsafe!")
            for warning in safety_info['warnings']:
                print(f"  {warning}")
            print("\nRecommendations:")
            print("  - Reduce batch_size")
            print("  - Reduce support_size and query_size")
            print("  - Use first_order=True (already set)")
            print("  - Reduce num_adaptation_steps")
            
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled due to memory concerns.")
                return None
        else:
            print("✅ Memory configuration appears safe.")
        
        print(f"Model memory: {safety_info['model_memory_gb']:.2f}GB")
        print(f"Estimated batch memory: {safety_info['estimated_batch_memory_gb']:.2f}GB")
        if isinstance(safety_info['gpu_info'], dict):
            print(f"Available GPU memory: {safety_info['gpu_info']['free_gb']:.2f}GB")
        
        results = {
            'meta_losses': [],
            'pre_adapt_losses': [],
            'post_adapt_losses': [],
            'task_specific_results': {},
            'validation_losses': []
        }
        
        # Initialize task-specific results tracking
        for task in train_tasks:
            results['task_specific_results'][task['name']] = {
                'pre_adapt_losses': [],
                'post_adapt_losses': [],
                'improvements': []
            }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting meta-training for {num_epochs} epochs...")
        print(f"Training on {len(train_tasks)} tasks")
        print(f"Initial memory usage: {get_memory_usage()}")
        
        for epoch in range(num_epochs):
            epoch_meta_losses = []
            epoch_pre_adapt = []
            epoch_post_adapt = []
            task_specific_metrics = {task['name']: {'pre': [], 'post': []} for task in train_tasks}
            
            # Shuffle tasks for each epoch
            shuffled_tasks = train_tasks.copy()
            random.shuffle(shuffled_tasks)
            
            self.model.train()
            
            # Process each task
            for task in tqdm(shuffled_tasks, desc=f"Epoch {epoch+1}/{num_epochs}"):
                try:
                    # Clear memory before processing each task
                    safe_memory_cleanup()
                    
                    # Create support and query splits
                    support_subset, query_subset = self.create_support_query_split(
                        task['dataset'], support_size, query_size
                    )
                    
                    # Create DataLoaders
                    support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=True)
                    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False)
                    
                    # Use the existing adapt method for adaptation
                    adapted_model, pre_loss, post_loss = self.adapt(support_loader)
                    
                    # Track adaptation metrics
                    epoch_pre_adapt.append(pre_loss)
                    epoch_post_adapt.append(post_loss)
                    task_specific_metrics[task['name']]['pre'].append(pre_loss)
                    task_specific_metrics[task['name']]['post'].append(post_loss)

                    # Evaluate adapted model on query set for meta-update
                    query_losses = []
                    adapted_model.train()
                    
                    for query_batch in query_loader:
                        input_ids = query_batch['input_ids'].to(DEVICE)
                        attention_mask = query_batch['attention_mask'].to(DEVICE)
                        targets = query_batch['target'].to(DEVICE)
                        
                        predictions = adapted_model(input_ids, attention_mask)
                        query_loss = F.mse_loss(predictions.squeeze(), targets)
                        query_losses.append(query_loss)
                    
                    if query_losses:
                        task_avg_loss = torch.stack(query_losses).mean()
                        epoch_meta_losses.append(task_avg_loss.item())
                        
                        # Meta-update - compute gradients with respect to original model
                        self.meta_optimizer.zero_grad()
                        
                        # Memory-efficient meta-update
                        # We need to compute the meta-loss properly for MAML
                        # This involves computing gradients through the adaptation process
                        
                        # Collect support data for meta-update
                        all_support_input_ids = []
                        all_support_attention_masks = []
                        all_support_targets = []
                        
                        for batch in support_loader:
                            all_support_input_ids.append(batch['input_ids'])
                            all_support_attention_masks.append(batch['attention_mask'])
                            all_support_targets.append(batch['target'])
                        
                        support_input_ids = torch.cat(all_support_input_ids, dim=0).to(DEVICE)
                        support_attention_mask = torch.cat(all_support_attention_masks, dim=0).to(DEVICE)
                        support_targets = torch.cat(all_support_targets, dim=0).to(DEVICE)
                        
                        # Compute support loss with gradient tracking
                        support_predictions = self.model(support_input_ids, support_attention_mask)
                        support_loss = F.mse_loss(support_predictions.squeeze(), support_targets)
                        
                        # Compute gradients for adaptation
                        grads = torch.autograd.grad(
                            support_loss, 
                            self.model.parameters(), 
                            create_graph=not self.first_order,
                            allow_unused=True,
                            retain_graph=not self.first_order
                        )
                        
                        # Create adapted parameters
                        adapted_params = []
                        for param, grad in zip(self.model.parameters(), grads):
                            if grad is not None:
                                adapted_params.append(param - self.inner_lr * grad)
                            else:
                                adapted_params.append(param)
                        
                        # Compute query loss with adapted parameters
                        query_loss_with_grads = []
                        for query_batch in query_loader:
                            input_ids = query_batch['input_ids'].to(DEVICE)
                            attention_mask = query_batch['attention_mask'].to(DEVICE)
                            targets = query_batch['target'].to(DEVICE)
                            
                            # Forward pass with adapted parameters
                            predictions = self.model(input_ids, attention_mask)
                            query_loss = F.mse_loss(predictions.squeeze(), targets)
                            query_loss_with_grads.append(query_loss)
                        
                        if query_loss_with_grads:
                            meta_loss = torch.stack(query_loss_with_grads).mean()
                            meta_loss.backward()
                        
                        # Track gradient norms for debugging
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.gradient_norms.append(total_norm.item())
                        
                        self.meta_optimizer.step()
                        
                        # Clear the computational graph and intermediate variables
                        del meta_loss, task_avg_loss, query_losses, support_loss, grads, adapted_params
                        del support_input_ids, support_attention_mask, support_targets, support_predictions
                        del query_loss_with_grads
                        
                        # Clear gradients properly
                        self.meta_optimizer.zero_grad()
                        
                        safe_memory_cleanup()
                    
                    # Clean up adapted model
                    del adapted_model
                    safe_memory_cleanup()
                
                except Exception as e:
                    print(f"Error processing task {task['name']}: {str(e)}")
                    print(f"Memory usage at error: {get_memory_usage()}")
                    # Force garbage collection
                    safe_memory_cleanup()
                    continue
                
            # Calculate epoch averages
            if epoch_meta_losses:
                avg_meta_loss = np.mean(epoch_meta_losses)
                avg_pre_adapt = np.mean(epoch_pre_adapt)
                avg_post_adapt = np.mean(epoch_post_adapt)
                
                results['meta_losses'].append(avg_meta_loss)
                results['pre_adapt_losses'].append(avg_pre_adapt)
                results['post_adapt_losses'].append(avg_post_adapt)
                
                # Store task-specific results
                for task_name, metrics in task_specific_metrics.items():
                    if metrics['pre'] and metrics['post']:
                        task_pre_avg = np.mean(metrics['pre'])
                        task_post_avg = np.mean(metrics['post'])
                        improvement = task_pre_avg - task_post_avg
                        
                        results['task_specific_results'][task_name]['pre_adapt_losses'].append(task_pre_avg)
                        results['task_specific_results'][task_name]['post_adapt_losses'].append(task_post_avg)
                        results['task_specific_results'][task_name]['improvements'].append(improvement)
                
                print(f"Epoch {epoch+1}: Meta-Loss: {avg_meta_loss:.4f}, "
                      f"Pre-Adapt: {avg_pre_adapt:.4f}, Post-Adapt: {avg_post_adapt:.4f}, "
                      f"Improvement: {avg_pre_adapt - avg_post_adapt:.4f}")
                print(f"Memory usage: {get_memory_usage()}")
                
                # Add gradient norm and learning rate info
                if self.gradient_norms:
                    recent_grad_norm = np.mean(self.gradient_norms[-len(shuffled_tasks):])
                    current_lr = self.meta_optimizer.param_groups[0]['lr']
                    print(f"  Grad Norm: {recent_grad_norm:.4f}, Meta LR: {current_lr:.6f}")
                
                # Validation and early stopping
                if (epoch + 1) % evaluate_every == 0 and len(train_tasks) > 1:
                    val_loss = self.validate(train_tasks[-1:], support_size, query_size, batch_size)
                    results['validation_losses'].append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(self.model.state_dict(), 'best_model_v2.pt')
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: No valid tasks processed")
            
            # Force garbage collection after each epoch
            safe_memory_cleanup()
        
        print("Meta-training completed!")
        return results
    
    def validate(self, val_tasks, support_size=16, query_size=16, batch_size=4):
        """
        Validate the meta-learned model on validation tasks
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for task in val_tasks:
                try:
                    support_subset, query_subset = self.create_support_query_split(
                        task['dataset'], support_size, query_size
                    )
                    
                    support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=False)
                    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False)
                    
                    # Quick adaptation
                    adapted_model, _, _ = self.validation_adapt(support_loader)
                    # adapted_model, _, _ = self.adapt(support_loader)
                    adapted_model.eval()
                    
                    # Evaluate on query set
                    for query_batch in query_loader:
                        input_ids = query_batch['input_ids'].to(DEVICE)
                        attention_mask = query_batch['attention_mask'].to(DEVICE)
                        targets = query_batch['target'].to(DEVICE)
                        
                        predictions = adapted_model(input_ids, attention_mask)
                        loss = F.mse_loss(predictions.squeeze(), targets)
                        val_losses.append(loss.item())
                
                except Exception as e:
                    print(f"Error validating task {task['name']}: {str(e)}")
                    continue
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def meta_evaluate(self, eval_task, support_size=16, query_size=16, batch_size=4):
        """
        Evaluate the meta-learned model on a new task
        """
        print(f"Meta-evaluating on task: {eval_task['name']}")
        
        self.model.eval()
        
        try:
            support_subset, query_subset = self.create_support_query_split(
                eval_task['dataset'], support_size, query_size
            )
            
            support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=False)
            query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False)
            
            # Adapt on support set
            # adapted_model, pre_loss, post_loss = self.validation_adapt(support_loader)
            adapted_model, pre_loss, post_loss = self.adapt(support_loader)
            adapted_model.eval()
            
            # print(f"Adaptation: Pre-adapt loss: {pre_loss:.4f}, Post-adapt loss: {post_loss:.4f}")
            # print(f"Improvement: {pre_loss - post_loss:.4f} ({((pre_loss - post_loss) / pre_loss * 100):.1f}%)")
            
            # Evaluate on query set
            # test_losses = []
            predictions_list = []
            targets_list = []
            with torch.no_grad():
                for query_batch in query_loader:
                    input_ids = query_batch['input_ids'].to(DEVICE)
                    attention_mask = query_batch['attention_mask'].to(DEVICE)
                    targets = query_batch['target'].to(DEVICE)
                    
                    predictions = adapted_model(input_ids, attention_mask)
                    # loss = F.mse_loss(predictions.squeeze(), targets)
                    # test_losses.append(loss.item())
                    predictions_flat = predictions.squeeze()
                    if predictions_flat.dim() == 0:
                        predictions_flat = predictions_flat.unsqueeze(0)
                    if targets.dim() == 0:
                        targets = targets.unsqueeze(0)
                    predictions_list.extend(predictions_flat.cpu().numpy())
                    targets_list.extend(targets.cpu().numpy())

            predictions_array = np.array(predictions_list)
            targets_array = np.array(targets_list)
            mse = np.mean((predictions_array - targets_array) ** 2)
            mae = np.mean(np.abs(predictions_array - targets_array))

            return {'mse': mse, 'mae': mae}
            # avg_test_loss = np.mean(test_losses) if test_losses else float('inf')
            # return avg_test_loss
            
        except Exception as e:
            print(f"Error evaluating task {eval_task['name']}: {str(e)}")
            return float('inf')
    
    def visualize_results(self, results, save_path='maml_results.png'):
        """
        Visualize the meta-training results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Meta-training loss curves
        epochs = range(1, len(results['meta_losses']) + 1)
        axes[0, 0].plot(epochs, results['meta_losses'], 'b-', label='Meta Loss', linewidth=2)
        axes[0, 0].plot(epochs, results['pre_adapt_losses'], 'r--', label='Pre-Adaptation', linewidth=2)
        axes[0, 0].plot(epochs, results['post_adapt_losses'], 'g--', label='Post-Adaptation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Meta-Training Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Adaptation improvement over time
        improvements = [pre - post for pre, post in zip(results['pre_adapt_losses'], results['post_adapt_losses'])]
        axes[0, 1].plot(epochs, improvements, 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Improvement')
        axes[0, 1].set_title('Adaptation Improvement Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Task-specific adaptation improvements
        task_names = list(results['task_specific_results'].keys())
        if task_names:
            final_improvements = []
            for task_name in task_names:
                task_results = results['task_specific_results'][task_name]
                if task_results['improvements']:
                    final_improvements.append(np.mean(task_results['improvements'][-5:]))  # Average of last 5 epochs
                else:
                    final_improvements.append(0)
            
            axes[1, 0].bar(range(len(task_names)), final_improvements, color='skyblue', alpha=0.7)
            axes[1, 0].set_xlabel('Tasks')
            axes[1, 0].set_ylabel('Average Improvement')
            axes[1, 0].set_title('Task-Specific Adaptation Improvements')
            axes[1, 0].set_xticks(range(len(task_names)))
            axes[1, 0].set_xticklabels([name.split('_')[0] + '...' for name in task_names], rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Validation loss if available
        if results['validation_losses']:
            val_epochs = range(len(results['validation_losses']))
            axes[1, 1].plot(val_epochs, results['validation_losses'], 'orange', marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Validation Check')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].set_title('Validation Performance')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Validation Performance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MAML TRAINING SUMMARY")
        print("="*60)
        
        if results['pre_adapt_losses'] and results['post_adapt_losses']:
            final_pre = results['pre_adapt_losses'][-1]
            final_post = results['post_adapt_losses'][-1]
            total_improvement = final_pre - final_post
            improvement_pct = (total_improvement / final_pre) * 100
            
            print(f"Final Pre-Adaptation Loss: {final_pre:.4f}")
            print(f"Final Post-Adaptation Loss: {final_post:.4f}")
            print(f"Final Improvement: {total_improvement:.4f} ({improvement_pct:.2f}%)")
            
            avg_pre = np.mean(results['pre_adapt_losses'])
            avg_post = np.mean(results['post_adapt_losses'])
            avg_improvement = avg_pre - avg_post
            avg_improvement_pct = (avg_improvement / avg_pre) * 100
            
            print(f"Average Pre-Adaptation Loss: {avg_pre:.4f}")
            print(f"Average Post-Adaptation Loss: {avg_post:.4f}")
            print(f"Average Improvement: {avg_improvement:.4f} ({avg_improvement_pct:.2f}%)")
        
        print("="*60)
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
from utils import print_model_info, safe_memory_cleanup, get_memory_usage, check_memory_safety

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MAML:
    """
    First-Order Model-Agnostic Meta-Learning (FOMAML) implementation for protein property prediction.
    
    FOMAML is a simplified version of MAML that uses first-order approximation to avoid
    the computational complexity of second-order gradients. This makes it much more
    robust and easier to implement correctly.
    
    Key features:
    - Uses first-order approximation (ignores second-order terms)
    - Much simpler gradient computation
    - More stable training
    - Avoids computational graph issues
    - Still achieves meta-learning objectives
    """
    
    def __init__(self, model, inner_lr=0.01, first_order=True, meta_lr=0.001, num_adaptation_steps=5):
        """
        Initialize FOMAML with first-order approximation.
        
        Args:
            model: The neural network model
            inner_lr: Learning rate for inner loop adaptation
            meta_lr: Learning rate for meta-optimization
            num_adaptation_steps: Number of adaptation steps in inner loop
        """
        self.model = model.to(DEVICE)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_adaptation_steps = num_adaptation_steps
        self.meta_optimizer = Adam(self.model.parameters(), lr=self.meta_lr)
        
        print(f"FOMAML initialized with:")
        print(f"  Inner LR: {inner_lr}")
        print(f"  Meta LR: {meta_lr}")
        print(f"  Adaptation steps: {num_adaptation_steps}")
        print(f"  Device: {DEVICE}")
        print(f"  Note: Using first-order approximation (FOMAML)")
        
        print_model_info(self.model)
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
        actual_support_size = min(support_size, total_size // 2)
        actual_query_size = min(query_size, total_size - actual_support_size)
        support_indices = indices[:actual_support_size]
        query_indices = indices[actual_support_size:actual_support_size + actual_query_size] 
        
        support_subset = Subset(dataset, support_indices)
        query_subset = Subset(dataset, query_indices)
        return support_subset, query_subset
    
    def collect_batch_data(self, data_loader):
        """
        Collect all data from a DataLoader into single tensors.
        
        Args:
            data_loader: DataLoader to collect data from
            
        Returns:
            tuple: (input_ids, attention_mask, targets)
        """
        all_input_ids = []
        all_attention_masks = []
        all_targets = []
        
        for batch in data_loader:
            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            all_targets.append(batch['target'])
        
        input_ids = torch.cat(all_input_ids, dim=0).to(DEVICE)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(DEVICE)
        targets = torch.cat(all_targets, dim=0).to(DEVICE)
        
        return input_ids, attention_mask, targets
    
    def adapt(self, support_loader):
        """
        Performs the inner loop update for a single task using the support set.
        
        FOMAML implementation with correct gradient flow:
        1. Store original parameters
        2. Adapt original model parameters in-place
        3. Return adaptation info for restoration
        
        Args:
            support_loader: DataLoader for support set
            
        Returns:
            tuple: (original_params, initial_loss, final_loss)
        """
        # Collect support data
        input_ids, attention_mask, targets = self.collect_batch_data(support_loader)
        
        # Store original parameters for restoration
        original_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Compute initial loss with original parameters
        with torch.no_grad():
            initial_predictions = self.model(input_ids, attention_mask)
            initial_loss = F.mse_loss(initial_predictions.squeeze(), targets)
        
        # Inner loop adaptation steps - modify original model in-place
        for step in range(self.num_adaptation_steps):
            # Forward pass with current model parameters
            predictions = self.model(input_ids, attention_mask)
            loss = F.mse_loss(predictions.squeeze(), targets)
            
            # Compute gradients w.r.t. current model parameters
            grads = torch.autograd.grad(
                loss, 
                self.model.parameters(), 
                create_graph=False,  # FOMAML: no computational graph needed
                allow_unused=True
            )
            
            # Update model parameters in-place using gradient descent
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - self.inner_lr * grad
        
        # Compute final loss with adapted parameters
        with torch.no_grad():
            final_predictions = self.model(input_ids, attention_mask)
            final_loss = F.mse_loss(final_predictions.squeeze(), targets)
        
        return original_params, initial_loss.item(), final_loss.item()
    
    def restore_parameters(self, original_params):
        """
        Restore model parameters to their original values.
        
        Args:
            original_params: List of original parameter tensors
        """
        with torch.no_grad():
            for param, orig_param in zip(self.model.parameters(), original_params):
                param.data = orig_param.data.clone()
    
    def validation_adapt(self, support_loader):
        """
        Performs adaptation for validation without gradient computation issues.
        """
        # For validation, we don't want to modify the original model
        # So we use the old approach with model copying
        adapted_model = copy.deepcopy(self.model)
        adapted_model.to(DEVICE)
        adapted_model.train()
        
        input_ids, attention_mask, targets = self.collect_batch_data(support_loader)
        
        # Compute initial loss
        with torch.no_grad():
            initial_predictions = adapted_model(input_ids, attention_mask)
            initial_loss = F.mse_loss(initial_predictions.squeeze(), targets)
        
        # Inner loop adaptation steps
        for step in range(self.num_adaptation_steps):
            predictions = adapted_model(input_ids, attention_mask)
            loss = F.mse_loss(predictions.squeeze(), targets)
            
            grads = torch.autograd.grad(
                loss, 
                adapted_model.parameters(), 
                create_graph=False,
                allow_unused=True
            )
            
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - self.inner_lr * grad
        
        # Compute final loss
        with torch.no_grad():
            final_predictions = adapted_model(input_ids, attention_mask)
            final_loss = F.mse_loss(final_predictions.squeeze(), targets)
        
        return adapted_model, initial_loss.item(), final_loss.item()
    
    def meta_train(self, train_tasks, num_epochs=50, support_size=16, query_size=16, 
                   batch_size=4, evaluate_every=10, patience=10):
        """
        Execute the meta-training loop for FOMAML.
        
        FOMAML outer loop:
        1. For each task: adapt model using support set (inner loop)
        2. Evaluate adapted model on query set
        3. Update original model parameters using query loss
        
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
        print("Performing memory safety check...")
        safety_info = check_memory_safety(self.model, batch_size, support_size, query_size)
        
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
        
        for task in train_tasks:
            results['task_specific_results'][task['name']] = {
                'pre_adapt_losses': [],
                'post_adapt_losses': [],
                'improvements': []
            }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting FOMAML meta-training for {num_epochs} epochs...")
        print(f"Training on {len(train_tasks)} tasks")
        
        for epoch in range(num_epochs):
            epoch_meta_losses = []
            epoch_pre_adapt = []
            epoch_post_adapt = []
            task_specific_metrics = {task['name']: {'pre': [], 'post': []} for task in train_tasks}
            
            shuffled_tasks = train_tasks.copy()
            random.shuffle(shuffled_tasks)
            
            self.model.train()
            
            for task in tqdm(shuffled_tasks, desc=f"Epoch {epoch+1}/{num_epochs}"):
                try:
                    safe_memory_cleanup()
                    
                    # Create support/query splits
                    support_subset, query_subset = self.create_support_query_split(
                        task['dataset'], support_size, query_size
                    )
                    support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=True)
                    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False)
                    
                    # Inner loop: Adapt model parameters in-place
                    original_params, pre_loss, post_loss = self.adapt(support_loader)
                    epoch_pre_adapt.append(pre_loss)
                    epoch_post_adapt.append(post_loss)
                    task_specific_metrics[task['name']]['pre'].append(pre_loss)
                    task_specific_metrics[task['name']]['post'].append(post_loss)
                    
                    # Outer loop: Compute query loss using adapted model parameters
                    query_input_ids, query_attention_mask, query_targets = self.collect_batch_data(query_loader)
                    
                    # Use adapted model parameters to compute query loss
                    query_predictions = self.model(query_input_ids, query_attention_mask)
                    meta_loss = F.mse_loss(query_predictions.squeeze(), query_targets)
                    
                    epoch_meta_losses.append(meta_loss.item())
                    
                    # Meta-optimization step - now gradients flow correctly!
                    self.meta_optimizer.zero_grad()
                    meta_loss.backward()
                    
                    # Gradient clipping
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.gradient_norms.append(total_norm.item())
                    
                    # Update meta-parameters (this updates the model permanently)
                    self.meta_optimizer.step()
                    
                    # In FOMAML, we DON'T restore parameters after meta-optimization
                    # Each task should start with the current model state (updated by meta-optimization)
                    # The restoration would undo the meta-learning progress!
                    
                    # Cleanup
                    del meta_loss, query_predictions, original_params
                    del query_input_ids, query_attention_mask, query_targets
                    safe_memory_cleanup()
                    
                except Exception as e:
                    print(f"Error processing task {task['name']}: {str(e)}")
                    print(f"Memory usage at error: {get_memory_usage()}")
                    safe_memory_cleanup()
                    continue
            
            if epoch_meta_losses:
                avg_meta_loss = np.mean(epoch_meta_losses)
                avg_pre_adapt = np.mean(epoch_pre_adapt)
                avg_post_adapt = np.mean(epoch_post_adapt)
                results['meta_losses'].append(avg_meta_loss)
                results['pre_adapt_losses'].append(avg_pre_adapt)
                results['post_adapt_losses'].append(avg_post_adapt)
                
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
                
                if self.gradient_norms:
                    recent_grad_norm = np.mean(self.gradient_norms[-len(shuffled_tasks):])
                    current_lr = self.meta_optimizer.param_groups[0]['lr']
                    print(f"  Grad Norm: {recent_grad_norm:.4f}, Meta LR: {current_lr:.6f}")
                
                # Validation
                if (epoch + 1) % evaluate_every == 0 and len(train_tasks) > 1:
                    val_loss = self.validate(train_tasks[-1:], support_size, query_size, batch_size)
                    results['validation_losses'].append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(self.model.state_dict(), 'best_model_fomaml.pt')
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: No valid tasks processed")
            safe_memory_cleanup()
        
        print("FOMAML meta-training completed!")
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
                    
                    adapted_model, _, _ = self.validation_adapt(support_loader)
                    adapted_model.eval()
                    
                    query_input_ids, query_attention_mask, query_targets = self.collect_batch_data(query_loader)
                    predictions = adapted_model(query_input_ids, query_attention_mask)
                    loss = F.mse_loss(predictions.squeeze(), query_targets)
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
            
            # Use the correct adapt method
            original_params, pre_loss, post_loss = self.adapt(support_loader)
            
            # Evaluate on query set using adapted model parameters
            query_input_ids, query_attention_mask, query_targets = self.collect_batch_data(query_loader)
            
            with torch.no_grad():
                predictions = self.model(query_input_ids, query_attention_mask)
                
                predictions_flat = predictions.squeeze()
                if predictions_flat.dim() == 0:
                    predictions_flat = predictions_flat.unsqueeze(0)
                if query_targets.dim() == 0:
                    query_targets = query_targets.unsqueeze(0)
                
                predictions_array = predictions_flat.cpu().numpy()
                targets_array = query_targets.cpu().numpy()
                mse = np.mean((predictions_array - targets_array) ** 2)
                
                # Restore original parameters AFTER using adapted model for evaluation
                self.restore_parameters(original_params)
                
            return {'mse': mse, 'pre_adapt_loss': pre_loss, 'post_adapt_loss': post_loss}
            
        except Exception as e:
            print(f"Error evaluating task {eval_task['name']}: {str(e)}")
            return {'mse': float('inf'), 'pre_adapt_loss': float('inf'), 'post_adapt_loss': float('inf')}

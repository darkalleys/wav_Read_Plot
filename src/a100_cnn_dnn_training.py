"""
A100-optimized CNN-DNN training script with robust error handling.
Implements mixed precision training with proper GradScaler handling and batch processing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
import numpy as np
import logging
from pathlib import Path
import time
from typing import Dict, Tuple, Optional, List
import warnings

from cnn_dnn_model import create_model_medium


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class A100TrainingConfig:
    """Configuration for A100-optimized training."""
    
    def __init__(self):
        # A100 optimizations
        self.enable_tf32 = True  # Enable TensorFloat-32 for A100
        self.mixed_precision = True  # Enable mixed precision training
        self.compile_model = True  # Enable torch.compile for A100
        
        # Training parameters
        self.batch_size = 8  # Minimum batch size to avoid BatchNorm issues
        self.min_batch_size = 2  # Minimum allowable batch size
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.num_epochs = 100
        
        # Model parameters
        self.input_channels = 2
        self.input_height = 64
        self.input_width = 64
        self.num_classes = 10
        
        # Error handling
        self.max_consecutive_failures = 5
        self.retry_with_smaller_batch = True
        self.fallback_to_fp32 = True


def setup_a100_optimizations(config: A100TrainingConfig):
    """Setup A100-specific optimizations."""
    
    if config.enable_tf32 and torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("A100: Enabled TensorFloat-32 (TF32) optimization")
    
    # Enable cuDNN benchmarking for consistent input sizes
    if torch.cuda.is_available():
        cudnn.benchmark = True
        logger.info("A100: Enabled cuDNN benchmarking")


def validate_batch_size(batch_size: int, min_batch_size: int = 2) -> int:
    """Validate and adjust batch size for BatchNorm requirements."""
    if batch_size < min_batch_size:
        logger.warning(f"Batch size {batch_size} is too small for training. Adjusting to {min_batch_size}")
        return min_batch_size
    return batch_size


def create_synthetic_data(config: A100TrainingConfig, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic training data for testing."""
    
    # CNN input: spectrograms
    cnn_data = torch.randn(num_samples, config.input_channels, config.input_height, config.input_width)
    
    # Additional features
    additional_features = torch.randn(num_samples, 128)
    
    # Random labels
    labels = torch.randint(0, config.num_classes, (num_samples,))
    
    return cnn_data, additional_features, labels


class RobustDataLoader:
    """Data loader with robust batch handling and error recovery."""
    
    def __init__(self, cnn_data, additional_features, labels, batch_size, min_batch_size=2):
        self.cnn_data = cnn_data
        self.additional_features = additional_features
        self.labels = labels
        self.batch_size = validate_batch_size(batch_size, min_batch_size)
        self.min_batch_size = min_batch_size
        self.num_samples = len(cnn_data)
        
    def __iter__(self):
        # Shuffle indices
        indices = torch.randperm(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            batch_indices = indices[i:end_idx]
            
            # Ensure minimum batch size
            if len(batch_indices) < self.min_batch_size:
                if i == 0:  # If this is the only batch, adjust batch size
                    if len(batch_indices) == 1:
                        # Add a duplicate sample to meet minimum batch size
                        batch_indices = torch.cat([batch_indices, batch_indices])
                        logger.warning("Duplicated sample to meet minimum batch size requirement")
                else:
                    # Skip small final batch
                    continue
            
            batch_cnn = self.cnn_data[batch_indices]
            batch_additional = self.additional_features[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            yield batch_cnn, batch_additional, batch_labels
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class A100Trainer:
    """A100-optimized trainer with robust error handling."""
    
    def __init__(self, config: A100TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = create_model_medium().to(self.device)
        
        # A100 optimizations
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("A100: Enabled torch.compile optimization")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize mixed precision scaler with proper error handling
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None
        self.use_mixed_precision = self.scaler is not None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Error tracking
        self.consecutive_failures = 0
        self.total_failures = 0
        
        logger.info(f"Initialized A100 trainer on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
    
    def train_batch(self, cnn_input, additional_features, labels) -> Dict[str, float]:
        """Train on a single batch with robust error handling."""
        
        try:
            # Move data to device
            cnn_input = cnn_input.to(self.device)
            additional_features = additional_features.to(self.device)
            labels = labels.to(self.device)
            
            # Validate tensor shapes
            if cnn_input.size(0) < self.config.min_batch_size:
                raise ValueError(f"Batch size {cnn_input.size(0)} is too small for training")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(cnn_input, additional_features)
                    loss = self.criterion(outputs, labels)
                
                # Scale loss and backward pass
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                # Unscale gradients and check for infs/nans
                try:
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Step optimizer with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                except RuntimeError as e:
                    if "Attempted unscale_ but _scale is None" in str(e):
                        logger.warning("GradScaler unscale error - attempting recovery")
                        # Reset scaler and try again without mixed precision for this batch
                        self.scaler = GradScaler()
                        return self._fallback_training(cnn_input, additional_features, labels)
                    else:
                        raise e
            else:
                # Standard precision training
                outputs = self.model(cnn_input, additional_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Step optimizer
                self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean().item()
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy,
                'batch_size': cnn_input.size(0)
            }
            
        except Exception as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            
            error_msg = f"Error in training batch: {str(e)}"
            logger.warning(error_msg)
            
            # Try fallback strategies
            if self.consecutive_failures <= self.config.max_consecutive_failures:
                if "Expected more than 1 value per channel" in str(e):
                    logger.info("Attempting to recover from BatchNorm error...")
                    return self._handle_batchnorm_error(cnn_input, additional_features, labels)
                elif "scaler" in str(e).lower() or "gradscaler" in str(e).lower():
                    logger.info("Attempting to recover from GradScaler error...")
                    return self._handle_gradscaler_error(cnn_input, additional_features, labels)
            
            # Return error metrics
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'batch_size': cnn_input.size(0) if cnn_input is not None else 0,
                'error': str(e)
            }
    
    def _fallback_training(self, cnn_input, additional_features, labels) -> Dict[str, float]:
        """Fallback training without mixed precision."""
        try:
            logger.info("Attempting fallback training without mixed precision")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Standard precision forward pass
            outputs = self.model(cnn_input, additional_features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Step optimizer
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean().item()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy,
                'batch_size': cnn_input.size(0),
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Fallback training also failed: {e}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'batch_size': cnn_input.size(0),
                'error': str(e),
                'fallback_failed': True
            }
    
    def _handle_batchnorm_error(self, cnn_input, additional_features, labels) -> Dict[str, float]:
        """Handle BatchNorm errors by ensuring proper batch size."""
        try:
            # This shouldn't happen with GroupNorm/LayerNorm, but handle it anyway
            if cnn_input.size(0) == 1:
                # Duplicate the sample to create a valid batch
                cnn_input = torch.cat([cnn_input, cnn_input], dim=0)
                additional_features = torch.cat([additional_features, additional_features], dim=0)
                labels = torch.cat([labels, labels], dim=0)
                logger.info("Duplicated sample to handle BatchNorm error")
            
            return self.train_batch(cnn_input, additional_features, labels)
            
        except Exception as e:
            logger.error(f"Failed to handle BatchNorm error: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0, 'batch_size': 0, 'error': str(e)}
    
    def _handle_gradscaler_error(self, cnn_input, additional_features, labels) -> Dict[str, float]:
        """Handle GradScaler errors by resetting scaler or disabling mixed precision."""
        try:
            # Reset the scaler
            self.scaler = GradScaler()
            logger.info("Reset GradScaler")
            
            # Try again with new scaler
            return self.train_batch(cnn_input, additional_features, labels)
            
        except Exception as e:
            logger.warning(f"GradScaler reset failed: {e}")
            
            if self.config.fallback_to_fp32:
                # Disable mixed precision for this batch
                original_use_mp = self.use_mixed_precision
                self.use_mixed_precision = False
                
                try:
                    result = self.train_batch(cnn_input, additional_features, labels)
                    result['fp32_fallback'] = True
                    return result
                finally:
                    self.use_mixed_precision = original_use_mp
            
            return {'loss': float('inf'), 'accuracy': 0.0, 'batch_size': 0, 'error': str(e)}
    
    def train_epoch(self, data_loader: RobustDataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        successful_batches = 0
        failed_batches = 0
        
        for batch_idx, (cnn_input, additional_features, labels) in enumerate(data_loader):
            # Debug logging
            logger.debug(f"Debug: cnn_input shape: {cnn_input.shape}")
            if hasattr(self.model, 'cnn'):
                with torch.no_grad():
                    test_features = self.model.cnn(cnn_input[:1].to(self.device))
                    logger.debug(f"Debug: cnn_features shape: {test_features.shape}")
            
            combined_features = torch.cat([
                torch.zeros(cnn_input.size(0), 512),  # CNN features placeholder
                additional_features
            ], dim=1)
            logger.debug(f"Debug: combined_features shape: {combined_features.shape}")
            
            # Train on batch
            batch_results = self.train_batch(cnn_input, additional_features, labels)
            
            if 'error' not in batch_results:
                total_loss += batch_results['loss'] * batch_results['batch_size']
                total_accuracy += batch_results['accuracy'] * batch_results['batch_size']
                total_samples += batch_results['batch_size']
                successful_batches += 1
            else:
                failed_batches += 1
                logger.warning(f"Batch {batch_idx} failed: {batch_results.get('error', 'Unknown error')}")
            
            # Early stopping if too many consecutive failures
            if self.consecutive_failures > self.config.max_consecutive_failures:
                logger.error("Too many consecutive failures. Stopping training.")
                break
        
        # Calculate averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = total_accuracy / total_samples
        else:
            avg_loss = float('inf')
            avg_accuracy = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'total_failures': self.total_failures
        }
    
    def train(self, data_loader: RobustDataLoader, num_epochs: int) -> List[Dict[str, float]]:
        """Train the model for multiple epochs."""
        
        logger.info(f"Starting A100 CNN-DNN training for {num_epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            epoch_results = self.train_epoch(data_loader)
            epoch_time = time.time() - start_time
            
            epoch_results['epoch'] = epoch
            epoch_results['epoch_time'] = epoch_time
            history.append(epoch_results)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Loss: {epoch_results['loss']:.4f}, "
                f"Accuracy: {epoch_results['accuracy']:.4f}, "
                f"Success/Failed batches: {epoch_results['successful_batches']}/{epoch_results['failed_batches']}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping if no successful batches
            if epoch_results['successful_batches'] == 0:
                logger.error("No successful batches in epoch. Stopping training.")
                break
        
        return history


def main():
    """Main training function."""
    
    # Setup configuration
    config = A100TrainingConfig()
    
    # Setup A100 optimizations
    setup_a100_optimizations(config)
    
    # Create synthetic data
    logger.info("Creating synthetic training data...")
    cnn_data, additional_features, labels = create_synthetic_data(config, num_samples=100)
    
    # Create data loader
    data_loader = RobustDataLoader(
        cnn_data, additional_features, labels,
        batch_size=config.batch_size,
        min_batch_size=config.min_batch_size
    )
    
    # Initialize trainer
    trainer = A100Trainer(config)
    
    try:
        # Train model
        history = trainer.train(data_loader, num_epochs=5)  # Short training for testing
        
        logger.info("A100 CNN-DNN Training completed successfully!")
        
        # Print summary
        successful_epochs = [h for h in history if h['successful_batches'] > 0]
        if successful_epochs:
            final_loss = successful_epochs[-1]['loss']
            final_acc = successful_epochs[-1]['accuracy']
            total_failures = successful_epochs[-1]['total_failures']
            
            logger.info(f"Final Loss: {final_loss:.4f}")
            logger.info(f"Final Accuracy: {final_acc:.4f}")
            logger.info(f"Total Failures: {total_failures}")
        else:
            logger.error("No successful epochs completed")
    
    except Exception as e:
        logger.error(f"A100 CNN-DNN Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
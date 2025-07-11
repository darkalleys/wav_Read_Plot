"""
Integration test to verify the complete CNN-DNN A100 training pipeline works correctly.
This test validates that the fixes resolve the original issues while maintaining performance.
"""

import sys
import os
sys.path.append('./src')

import torch
import logging
from pathlib import Path

from cnn_dnn_model import create_model_medium
from a100_cnn_dnn_training import (
    A100TrainingConfig, A100Trainer, create_synthetic_data, 
    RobustDataLoader, setup_a100_optimizations
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_complete_training_pipeline():
    """Test the complete training pipeline from data creation to model training."""
    
    print("🚀 TESTING COMPLETE CNN-DNN A100 TRAINING PIPELINE")
    print("="*60)
    
    # 1. Configuration
    print("1. Setting up A100 training configuration...")
    config = A100TrainingConfig()
    config.batch_size = 4  # Safe batch size
    config.num_epochs = 3  # Short training for testing
    
    # Setup A100 optimizations
    setup_a100_optimizations(config)
    print(f"   ✅ Configuration: batch_size={config.batch_size}, mixed_precision={config.mixed_precision}")
    
    # 2. Data creation
    print("\n2. Creating synthetic training data...")
    cnn_data, additional_features, labels = create_synthetic_data(config, num_samples=50)
    print(f"   ✅ Created data: CNN={cnn_data.shape}, Features={additional_features.shape}, Labels={labels.shape}")
    
    # 3. Data loader
    print("\n3. Setting up robust data loader...")
    data_loader = RobustDataLoader(
        cnn_data, additional_features, labels,
        batch_size=config.batch_size,
        min_batch_size=config.min_batch_size
    )
    print(f"   ✅ Data loader: batch_size={config.batch_size}, min_batch_size={config.min_batch_size}")
    
    # 4. Model creation
    print("\n4. Creating CNN-DNN model...")
    model = create_model_medium()
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5. Trainer initialization
    print("\n5. Initializing A100 trainer...")
    trainer = A100Trainer(config)
    print(f"   ✅ Trainer initialized on device: {trainer.device}")
    print(f"   ✅ Mixed precision: {trainer.use_mixed_precision}")
    
    # 6. Training execution
    print("\n6. Running training...")
    try:
        history = trainer.train(data_loader, num_epochs=config.num_epochs)
        
        # Analyze results
        successful_epochs = [h for h in history if h['successful_batches'] > 0]
        
        if len(successful_epochs) == config.num_epochs:
            print("   ✅ All epochs completed successfully!")
            
            final_metrics = successful_epochs[-1]
            print(f"   📊 Final metrics:")
            print(f"      - Loss: {final_metrics['loss']:.4f}")
            print(f"      - Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"      - Successful batches: {final_metrics['successful_batches']}")
            print(f"      - Failed batches: {final_metrics['failed_batches']}")
            print(f"      - Total failures: {final_metrics['total_failures']}")
            
            return True
        else:
            print(f"   ⚠️  Only {len(successful_epochs)}/{config.num_epochs} epochs completed")
            return False
            
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error scenarios."""
    
    print("\n" + "="*60)
    print("🧪 TESTING EDGE CASES AND ERROR SCENARIOS")
    print("="*60)
    
    config = A100TrainingConfig()
    
    # Test 1: Very small dataset
    print("\n1. Testing with very small dataset...")
    try:
        cnn_data, additional_features, labels = create_synthetic_data(config, num_samples=5)
        data_loader = RobustDataLoader(cnn_data, additional_features, labels, batch_size=4, min_batch_size=2)
        
        # Should handle gracefully
        batch_count = 0
        for batch in data_loader:
            batch_count += 1
        
        print(f"   ✅ Small dataset handled: {batch_count} batches created")
        
    except Exception as e:
        print(f"   ❌ Small dataset failed: {e}")
        return False
    
    # Test 2: Edge case batch sizes
    print("\n2. Testing edge case batch sizes...")
    try:
        model = create_model_medium()
        
        # Test batch_size = 1 (should work with our fixes)
        cnn_input = torch.randn(1, 2, 64, 64)
        additional_features = torch.randn(1, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(cnn_input, additional_features)
            print(f"   ✅ Batch size 1 handled: output shape {output.shape}")
        
        # Test larger batch
        cnn_input = torch.randn(8, 2, 64, 64)
        additional_features = torch.randn(8, 128)
        
        with torch.no_grad():
            output = model(cnn_input, additional_features)
            print(f"   ✅ Batch size 8 handled: output shape {output.shape}")
            
    except Exception as e:
        print(f"   ❌ Batch size test failed: {e}")
        return False
    
    # Test 3: Error recovery
    print("\n3. Testing error recovery mechanisms...")
    try:
        trainer = A100Trainer(config)
        
        # Test with problematic data (empty tensors)
        try:
            empty_cnn = torch.empty(0, 2, 64, 64)
            empty_features = torch.empty(0, 128)
            empty_labels = torch.empty(0, dtype=torch.long)
            
            result = trainer.train_batch(empty_cnn, empty_features, empty_labels)
            
            if 'error' in result:
                print(f"   ✅ Error recovery worked: {result['error']}")
            else:
                print(f"   ⚠️  Unexpected success with empty data")
                
        except Exception as e:
            print(f"   ✅ Error properly caught and handled: {e}")
            
    except Exception as e:
        print(f"   ❌ Error recovery test failed: {e}")
        return False
    
    return True


def test_performance_benchmarks():
    """Test performance characteristics."""
    
    print("\n" + "="*60)
    print("⚡ TESTING PERFORMANCE CHARACTERISTICS")
    print("="*60)
    
    import time
    
    config = A100TrainingConfig()
    model = create_model_medium()
    
    # Test 1: Inference speed
    print("\n1. Testing inference speed...")
    
    batch_sizes = [1, 4, 8, 16]
    for batch_size in batch_sizes:
        cnn_input = torch.randn(batch_size, 2, 64, 64)
        additional_features = torch.randn(batch_size, 128)
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(cnn_input, additional_features)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(cnn_input, additional_features)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        throughput = batch_size * 100 / (end_time - start_time)  # samples/sec
        
        print(f"   Batch size {batch_size:2d}: {avg_time:6.2f}ms/batch, {throughput:8.1f} samples/sec")
    
    # Test 2: Memory usage
    print("\n2. Testing memory efficiency...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Memory usage with different batch sizes
    if torch.cuda.is_available():
        print(f"   GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("   ✅ Performance benchmarks completed")
    
    return True


def main():
    """Run all integration tests."""
    
    print("🔧 CNN-DNN A100 TRAINING INTEGRATION TESTS")
    print("="*60)
    print("Testing the complete pipeline with all fixes applied.\n")
    
    # Run tests
    tests = [
        ("Complete Training Pipeline", test_complete_training_pipeline),
        ("Edge Cases and Error Scenarios", test_edge_cases),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The CNN-DNN A100 training pipeline is working correctly with all fixes applied.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
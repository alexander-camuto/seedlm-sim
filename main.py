import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from seedlm import lsfr_compression


# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(512, 16)
        self.fc2 = nn.Linear(16, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example 1: Compress an entire model
def example_compress_model():
    # Create a model
    model = SimpleModel()
    
     # Test inference
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        # Test inference
        original_output = model(x)
    
    
    # Calculate original model size
    original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original model parameters: {original_params}")
    
    # Compress the model with parameters matching the paper
    # K=8 (LFSR length), P=4 (projection dimension), bits=4 (quantization bits)
    compressed_model = lsfr_compression.compress_model(model, K=8, P=4, bits=4)
    
    # Calculate compression ratio
    ratio = lsfr_compression.calculate_compression_ratio(model, compressed_model)
    print(f"Compression ratio: {ratio:.4f}")
    
   
    with torch.no_grad():
        output = compressed_model(x)
    print(f"Output shape: {output.shape}")
    
    # Check if the output is similar to the original model
    diff = torch.abs(original_output - output).max()
    print(f"Max difference between original and compressed model output: {diff.item():.6f}")

    return compressed_model


# Example 5: Demonstrate LFSR sequence generation and normalization
def example_demonstrate_lfsr():
    print("\nDemonstrating LFSR sequence generation:")
    
    # Define a 4-bit LFSR with polynomial x^4 + x + 1 (coefficients [1,1,0,0])
    lfsr1 = lsfr_compression.LFSR(seed=5, coefficients=[1, 1, 0, 0], nbits=4)
    
    # Generate and print a sequence of 16 bits
    print("LFSR with polynomial x^4 + x + 1 (coefficients [1,1,0,0]), seed=5:")
    bits_sequence = []
    for _ in range(16):
        bit = lfsr1.next_bit()
        bits_sequence.append(bit)
    print(f"Bit sequence: {bits_sequence}")
    
    # Try a different polynomial: x^4 + x^3 + 1 (coefficients [1,0,1,0])
    lfsr2 = lsfr_compression.LFSR(seed=5, coefficients=[1, 0, 1, 0], nbits=4)
    
    # Generate and print a sequence of 16 bits
    print("\nLFSR with polynomial x^4 + x^3 + 1 (coefficients [1,0,1,0]), seed=5:")
    bits_sequence = []
    for _ in range(16):
        bit = lfsr2.next_bit()
        bits_sequence.append(bit)
    print(f"Bit sequence: {bits_sequence}")
    
    # Demonstrate normalization formula from the paper
    lfsr3 = lsfr_compression.LFSR(seed=7, coefficients=[1, 1, 0, 0], nbits=4)
    
    print("\nDemonstrating normalization of 4-bit LFSR values:")
    print("Raw integers and their normalized values:")
    for _ in range(8):
        raw_int = lfsr3.next_int()
        # Reset LFSR to the same state to get the same value
        temp_state = lfsr3.state
        lfsr3.state = temp_state >> 4  # Restore state before next_int
        normalized = lfsr3.next_float()
        # Manually calculate normalization for verification
        mid_point = 2**(4 - 1)  # 2^(K-1)
        norm_factor = 1.0 / (mid_point - 1)
        manual_norm = norm_factor * (raw_int - mid_point)
        print(f"Raw: {raw_int}, Normalized: {normalized:.6f}, Manual: {manual_norm:.6f}")
    
    # Generate a small random matrix using LFSR
    lfsr4 = lsfr_compression.LFSR(seed=7, coefficients=[1, 1, 0, 0], nbits=4)
    matrix = lfsr4.generate_matrix((3, 3))
    print("\nRandom 3x3 matrix generated with LFSR (seed=7):")
    print(matrix)
    print(f"Matrix min: {matrix.min().item():.6f}, max: {matrix.max().item():.6f}")
    
    # Demonstrate sequence reproducibility with same seed
    lfsr5a = lsfr_compression.LFSR(seed=9, coefficients=[1, 1, 0, 0], nbits=4)
    lfsr5b = lsfr_compression.LFSR(seed=9, coefficients=[1, 1, 0, 0], nbits=4)
    
    print("\nDemonstrating reproducibility with same seed=9:")
    matrix1 = lfsr5a.generate_matrix((2, 2))
    matrix2 = lfsr5b.generate_matrix((2, 2))
    print("Matrix 1:\n", matrix1)
    print("Matrix 2:\n", matrix2)
    print(f"Matrices are identical: {torch.all(matrix1 == matrix2).item()}")
    
    # Show different K values and their effects on normalization
    print("\nEffect of different K values on normalization range:")
    for k in [4, 8, 16]:
        lfsr_k = lsfr_compression.LFSR(seed=12, coefficients=None, nbits=k)
        samples = [lfsr_k.next_float() for _ in range(100)]
        min_val = min(samples)
        max_val = max(samples)
        print(f"K={k}: min={min_val:.6f}, max={max_val:.6f}, range={max_val-min_val:.6f}")

# Run the examples
if __name__ == "__main__":
    
    print("\nExample 1: Demonstrating LFSR sequence generation")
    example_demonstrate_lfsr()
    
    print("Example 2: Compressing an entire model")
    compressed_model = example_compress_model()
    
    
    
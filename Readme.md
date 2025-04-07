# SeedLM: LFSR-Based Weight Compression simulation using PyTorch

This repository implements the LFSR-based neural network weight compression technique as described in ["SeedLM: LFSR-Based Weight Compression for Large Language Models"](https://arxiv.org/pdf/2410.10714).

Rather than using hardware LFSR registers, we simulate them in software to attempt to reproduce the exact behavior described in the original research.

It provides a simple, modular approach for compressing PyTorch models using Linear Feedback Shift Register (LFSR) techniques.

I am merely reproducing this research -- all credits due to original authors ! 

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy

### Setting up a Virtual Environment

1. Clone this repository:
```bash
git clone https://github.com/alexander-camuto/seedlm-sim.git
cd seedlm-sim
```

2. Create a virtual environment:
```bash
# Using venv
python -m venv venv

# Activate the environment
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from seedlm.lfsr_compression import compress_model, compress_layer

# Load your existing PyTorch model
model = YourModel()

# Compress the entire model
compressed_model = compress_model(model, K=8, P=4, bits=4)

# Use the compressed model for inference
output = compressed_model(input_data)

# Alternatively, compress specific layers
model.layer1 = compress_layer(model.layer1, K=8, P=4, bits=4)
```

### Parameters

- `K`: LFSR length parameter (determines number of matrices to test)
- `P`: Projection dimension (controls compression ratio)
- `bits`: Number of bits for quantization (default is 4 as in the paper)
- `coefficients`: Feedback polynomial coefficients for LFSR

### Examples

See the `main.py` file for example usage:

```bash
# Run the example
python main.py
```

## Implementation Details

The implementation follows the algorithm described in the paper:

1. For a given weight matrix, generate N = 2^K - 1 random matrices using LFSR
2. Project the weight onto each matrix and quantize the projections
3. Choose the seed that minimizes reconstruction error
4. Store only the seed, quantized projection, and shared exponent

During inference, weights are regenerated on-the-fly using the stored seed and projection.

## Memory Savings

The memory reduction depends on the original weight size (C), projection dimension (P), and quantization bits (b):

- Original size: C × 32 bits (for float32)
- Compressed size: P × b bits + overhead (seed and exponent)

For example, a 1024×1024 weight matrix (4MB) can be compressed to just a few KB, depending on the projection dimension.




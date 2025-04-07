import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class LFSRCompressedParameter(nn.Parameter):
    """
    A compressed parameter using LFSR projection method.
    """
    def __new__(cls, data, seed, quantized_projection, exponent, coefficients=None, K=8, bits=4):
        # Create a new Parameter instance properly
        instance = super(LFSRCompressedParameter, cls).__new__(cls, data, requires_grad=False)
        return instance
    
    def __init__(self, data, seed, quantized_projection, exponent, coefficients=None, K=8, bits=4):
        """
        Initialize a compressed parameter.
        
        Args:
            data: Original weight tensor
            seed: Optimal seed for LFSR
            quantized_projection: Quantized projection vector
            exponent: Shared exponent for dequantization
            coefficients: Feedback polynomial coefficients
            K: LFSR length parameter
            bits: Number of bits used for quantization
        """
        # Don't call super().__init__() as it's already called in __new__
        
        self.original_shape = data.shape
        self.seed = seed
        self.quantized_projection = quantized_projection  # This is a regular tensor, not a parameter
        self.exponent = exponent
        self.coefficients = coefficients
        self.K = K
        self.bits = bits
        
        # Cache for the random matrix to avoid regenerating it each time
        self._cached_matrix = None
        
    def regenerate(self) -> torch.Tensor:
        """Regenerate the weights using the stored seed and projection."""
        # Use cached matrix if available, otherwise generate it
        if self._cached_matrix is None:
            # Create LFSR with the stored seed and enable caching for efficiency
            lfsr = LFSR(seed=self.seed, coefficients=self.coefficients, nbits=self.K, cache_states=True)
            
            # Generate the random matrix
            C = np.prod(self.original_shape)
            P = len(self.quantized_projection)
            self._cached_matrix = lfsr.generate_matrix((C, P))
        
        # Dequantize the projection (quantized_projection is now a tensor, not a Parameter)
        t_hat = dequantize_vector(self.quantized_projection, self.exponent, self.bits)
        
        # Reconstruct the weight vector
        w_reconstructed = torch.matmul(self._cached_matrix, t_hat).reshape(self.original_shape)
        
        return w_reconstructed

class LFSR:
    """
    Linear Feedback Shift Register implementation for random number generation
    following the mathematical definition from the paper.
    
    Includes state caching for efficient matrix generation.
    """
    def __init__(self, seed: int, coefficients: List[int] = None, nbits: int = 16, cache_states: bool = True):
        """
        Initialize the LFSR.
        
        Args:
            seed: Initial state of the register
            coefficients: Binary coefficients (α₀, α₁, ..., α_{K-1}) defining the feedback polynomial.
                          If None, uses a default configuration.
            nbits: Number of bits in the register (K in the paper)
            cache_states: Whether to cache all possible states for efficient matrix generation
        """
        self.state = seed
        self.nbits = nbits
        self.mask = (1 << nbits) - 1  # Mask to keep only nbits
        self.cache_states = cache_states
        
        # If coefficients are not provided, use a default configuration
        # based on primitive polynomials which guarantee maximum period
        if coefficients is None:
            # Default coefficients for common register sizes
            # These correspond to primitive polynomials
            default_coeffs = {
                4: [1, 1, 0, 0],          # x⁴ + x + 1
                8: [1, 0, 1, 1, 0, 0, 0],  # x⁸ + x⁴ + x³ + x + 1
                16: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # x¹⁶ + x¹⁵ + x + 1
            }
            
            # Use default or fall back to the simplest case for other sizes
            self.coefficients = default_coeffs.get(nbits, [1] + [0] * (nbits-2) + [1])
        else:
            self.coefficients = coefficients
            
        # Ensure coefficients array has correct length
        if len(self.coefficients) < nbits:
            self.coefficients = self.coefficients + [0] * (nbits - len(self.coefficients))
        elif len(self.coefficients) > nbits:
            self.coefficients = self.coefficients[:nbits]
        
        # Cache of states for efficient matrix generation
        self.state_cache = {}
        
        # If caching is enabled, generate and cache all possible states
        if self.cache_states:
            self._generate_state_cache()
            
    def _generate_state_cache(self):
        """
        Generate and cache all possible states of the LFSR.
        As described in the paper, this allows for efficient matrix generation.
        """
        # Maximum number of states to cache (2^K - 1)
        max_states = (1 << self.nbits) - 1
        
        # Remember original state
        original_state = self.state
        
        # Reset to seed state
        curr_state = self.state
        
        # Generate and cache all states
        self.state_cache = {}
        
        # Store the initial seed state
        self.state_cache[0] = curr_state
        
        # Generate the sequence of states
        for i in range(1, max_states):
            # Calculate new bit using the formula from the paper
            new_bit = 0
            
            # Initialize registers with the current state
            registers = [(curr_state >> j) & 1 for j in range(self.nbits-1, -1, -1)]
            
            # Calculate x_{n+1} = ∑_{i=0}^{K-1} α_i · x_{n+i-K+1} mod 2
            for j in range(self.nbits):
                # Only include terms where coefficient is 1
                if self.coefficients[j] == 1:
                    new_bit ^= registers[j]  # XOR operation for mod 2 addition
            
            # Update state: shift left and insert new bit at the rightmost position
            curr_state = ((curr_state << 1) | new_bit) & self.mask
            
            # Store the state
            self.state_cache[i] = curr_state
        
        # Reset the state back to original
        self.state = original_state
        
    def next_bit(self) -> int:
        """
        Generate the next bit in the sequence according to the paper's formula:
        x_{n+1} = ∑_{i=0}^{K-1} α_i · x_{n+i-K+1} mod 2
        """
        # Calculate new bit using the formula from the paper
        new_bit = 0
        
        # Initialize registers with the current state
        registers = [(self.state >> i) & 1 for i in range(self.nbits-1, -1, -1)]
        
        # Calculate x_{n+1} = ∑_{i=0}^{K-1} α_i · x_{n+i-K+1} mod 2
        for i in range(self.nbits):
            # Only include terms where coefficient is 1
            if self.coefficients[i] == 1:
                new_bit ^= registers[i]  # XOR operation for mod 2 addition
                
        # Update state: shift left and insert new bit at the rightmost position
        self.state = ((self.state << 1) | new_bit) & self.mask
        
        return new_bit
    
    def next_float(self) -> float:
        """Generate a random float between -1 and 1."""
        value = 0
        for _ in range(self.nbits):
            value = (value << 1) | self.next_bit()
        
        # Convert to float between -1 and 1
        return (value / (2**self.nbits - 1)) * 2 - 1
        
    def next_int(self) -> int:
        """
        Generate the next integer value from the LFSR.
        This produces a raw integer value before normalization.
        """
        value = 0
        for _ in range(self.nbits):
            value = (value << 1) | self.next_bit()
        return value
        
    def next_float(self) -> float:
        """
        Generate a random float between -1 and 1 using the paper's normalization formula:
        U(s) = (1 / (2^(K-1) - 1)) * (V(s) - 2^(K-1) * 1)
        
        Where V(s) is the raw integer from the LFSR, and K is the number of bits.
        """
        # Generate raw integer value V(s)
        raw_value = self.next_int()
        
        # Apply the normalization formula from the paper
        mid_point = 2**(self.nbits - 1)
        normalization_factor = 1.0 / (mid_point - 1)
        
        # U(s) = (1 / (2^(K-1) - 1)) * (V(s) - 2^(K-1) * 1)
        normalized_value = normalization_factor * (raw_value - mid_point)
        
        return normalized_value
        
    def generate_matrix(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate a random matrix of the specified shape using the LFSR.
        Each element is a float between -1 and 1, normalized according to the paper's formula.
        
        Args:
            shape: The shape of the matrix to generate
            
        Returns:
            A torch tensor with random values derived from the LFSR
        """
        size = np.prod(shape)
        values = [self.next_float() for _ in range(size)]
        return torch.tensor(values, dtype=torch.float32).reshape(shape)


def quantize_vector(vector: torch.Tensor, bits: int = 4) -> Tuple[torch.Tensor, int]:
    """
    Quantize a vector using n-bit integers and a shared exponent.
    
    Args:
        vector: Vector to quantize
        bits: Number of bits to use for quantization
        
    Returns:
        Tuple of (quantized_vector, exponent)
    """
    # Find max absolute value to determine exponent
    max_abs = torch.max(torch.abs(vector))
    
    if max_abs == 0:
        return torch.zeros_like(vector), 0
    
    # Calculate exponent (power of 2)
    exponent = torch.floor(torch.log2(max_abs)).int().item()
    
    # Calculate scaling factor
    scale = 2.0 ** exponent
    
    # Scale the vector
    scaled = vector / scale
    
    # Calculate the maximum value representable with the given bits
    max_int = (2 ** (bits - 1)) - 1  # Assuming signed integers
    
    # Clamp and quantize
    quantized = torch.round(torch.clamp(scaled * max_int, -max_int, max_int))
    
    return quantized, exponent


def dequantize_vector(quantized: torch.Tensor, exponent: int, bits: int = 4) -> torch.Tensor:
    """
    Dequantize a vector using the shared exponent.
    
    Args:
        quantized: Quantized vector
        exponent: Shared exponent
        bits: Number of bits used for quantization
        
    Returns:
        Dequantized vector
    """
    max_int = (2 ** (bits - 1)) - 1
    scale = 2.0 ** exponent
    return (quantized / max_int) * scale


def find_optimal_projection(weight: torch.Tensor, K: int = 8, P: int = 4, bits: int = 4, coefficients: List[int] = None):
    """
    Find the optimal random matrix and projection for compressing weights.
    Uses LFSR with state caching for efficient matrix generation.
    
    Args:
        weight: Weight tensor to compress
        K: LFSR length parameter (determines number of matrices to test)
        P: Projection dimension
        bits: Number of bits for quantization
        coefficients: Feedback polynomial coefficients (if None, defaults will be used)
        
    Returns:
        Tuple of (best_seed, quantized_projection, exponent, best_matrix)
    """
    # Reshape weight to vector
    original_shape = weight.shape
    w = weight.reshape(-1)
    C = len(w)
    
    # Generate N = 2^K - 1 random matrices
    N = (2 ** K) - 1
    
    best_error = float('inf')
    best_seed = 0
    best_quantized = None
    best_exponent = 0
    best_matrix = None
    
    # Try all possible seeds using cached LFSR approach
    for j in range(1, N + 1):  # From 1 to N as per the paper
        # Create LFSR with seed j and enable caching
        lfsr = LFSR(seed=j, coefficients=coefficients, nbits=K, cache_states=True)
        
        # Generate random matrix U(s_j) of size C×P using cached states
        U_j = lfsr.generate_matrix((C, P))
        
        # Project weight vector onto the matrix
        t_j = torch.matmul(U_j.T, w)  # This is t_j = U(s_j)^T * w
        
        # Quantize t_j
        quantized_t_j, exponent_j = quantize_vector(t_j, bits=bits)
        
        # Dequantize for error calculation
        dequantized_t_j = dequantize_vector(quantized_t_j, exponent_j, bits=bits)
        
        # Reconstruct and compute error
        reconstructed_w = torch.matmul(U_j, dequantized_t_j)
        error = torch.norm(w - reconstructed_w, p=2) ** 2
        
        # Update best if this has lower error
        if error < best_error:
            best_error = error
            best_seed = j
            best_quantized = quantized_t_j
            best_exponent = exponent_j
            best_matrix = U_j
    
    return best_seed, best_quantized, best_exponent, best_matrix



class LFSRCompressedLayer(nn.Module):
    """
    A wrapper for any PyTorch layer that compresses weights using LFSR projection.
    """
    def __init__(self, 
                original_layer: nn.Module, 
                K: int = 8,
                P: int = 4,
                bits: int = 4,
                coefficients: List[int] = None):
        """
        Initialize a compressed layer.
        
        Args:
            original_layer: The original PyTorch layer
            K: LFSR length parameter
            P: Projection dimension
            bits: Number of bits for quantization
            coefficients: Feedback polynomial coefficients (if None, defaults will be used)
        """
        super().__init__()
        self.original_layer = original_layer
        self.K = K
        self.P = P
        self.bits = bits
        self.coefficients = coefficients
        
        # Compress weights
        self._compress_weights()
        
    def _compress_weights(self):
        """Compress the weights of the layer using LFSR projection."""
        for name, param in list(self.original_layer.named_parameters()):
            # Skip bias and other non-weight parameters
            if 'weight' not in name or not param.requires_grad:
                continue
                
            # Find optimal projection
            best_seed, quantized_projection, exponent, _ = find_optimal_projection(
                param.data, 
                K=self.K,
                P=self.P,
                bits=self.bits,
                coefficients=self.coefficients
            )
            
            # Create compressed parameter
            compressed_param = LFSRCompressedParameter(
                param.data,
                best_seed,
                quantized_projection,
                exponent,
                self.coefficients,
                self.K,
                self.bits
            )
            
            # Replace the original parameter in the layer
            for p_name, _ in self.original_layer.named_parameters():
                if p_name == name:
                    param_path = p_name.split('.')
                    module = self.original_layer
                    for part in param_path[:-1]:
                        module = getattr(module, part)
                    setattr(module, param_path[-1], compressed_param)
    
    def forward(self, *args, **kwargs):
        """Forward pass: regenerate weights and call the original forward method."""
        # Regenerate weights for all compressed parameters
        for name, param in self.original_layer.named_parameters():
            if isinstance(param, LFSRCompressedParameter):
                # Regenerate the weights
                regenerated_weights = param.regenerate()
                # Replace the data in the parameter
                param.data = regenerated_weights
                
        # Call the original forward method
        return self.original_layer(*args, **kwargs)
    
    def extra_repr(self) -> str:
        """Extra representation information."""
        return f'K={self.K}, P={self.P}, bits={self.bits}'


class LFSRCompressedModel(nn.Module):
    """
    A model with LFSR-compressed layers.
    """
    def __init__(self, 
                original_model: nn.Module, 
                K: int = 8,
                P: int = 4,
                bits: int = 4,
                coefficients: List[int] = None,
                compress_all: bool = False,
                layer_types: List[type] = None):
        """
        Initialize a compressed model.
        
        Args:
            original_model: The original PyTorch model
            K: LFSR length parameter
            P: Projection dimension
            bits: Number of bits for quantization
            coefficients: Feedback polynomial coefficients
            compress_all: Whether to compress all layers or just specific types
            layer_types: Types of layers to compress (if compress_all is False)
        """
        super().__init__()
        
        self.original_model = original_model
        self.K = K
        self.P = P
        self.bits = bits
        self.coefficients = coefficients
        
        if layer_types is None:
            # Default: compress only Linear and Conv2d layers
            layer_types = [nn.Linear, nn.Conv2d]
        
        # Replace layers with compressed versions
        self._compress_layers(compress_all, layer_types)
        
    def _compress_layers(self, compress_all: bool, layer_types: List[type]):
        """
        Replace layers with compressed versions.
        
        Args:
            compress_all: Whether to compress all layers
            layer_types: Types of layers to compress
        """
        # Function to recursively replace layers
        def replace_layers(module, prefix=''):
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                
                print(f" --- processing layer: {full_name}")
                
                # Check if this child should be compressed
                should_compress = compress_all or any(isinstance(child, lt) for lt in layer_types)
                
                if should_compress and hasattr(child, 'weight'):
                    # Replace with compressed layer
                    compressed_layer = LFSRCompressedLayer(
                        original_layer=child,
                        K=self.K,
                        P=self.P,
                        bits=self.bits,
                        coefficients=self.coefficients
                    )
                    setattr(module, name, compressed_layer)
                else:
                    # Recursively process child's children
                    replace_layers(child, full_name)
        
        # Start the recursive replacement
        replace_layers(self.original_model)
    
    def forward(self, *args, **kwargs):
        """Forward pass using the original model's forward method."""
        return self.original_model(*args, **kwargs)
        
    def extra_repr(self) -> str:
        """Extra representation information."""
        return f'K={self.K}, P={self.P}, bits={self.bits}'

# Utility functions for compression
def compress_model(model, K=8, P=4, bits=4, coefficients=None):
    """
    Utility function to compress a model using LFSR projection.
    
    Args:
        model: PyTorch model to compress
        K: LFSR length parameter
        P: Projection dimension
        bits: Number of bits for quantization
        coefficients: Feedback polynomial coefficients
        
    Returns:
        Compressed model
    """
    return LFSRCompressedModel(model, K=K, P=P, bits=bits, coefficients=coefficients)

def compress_layer(layer, K=8, P=4, bits=4, coefficients=None):
    """
    Utility function to compress a single layer using LFSR projection.
    
    Args:
        layer: PyTorch layer to compress
        K: LFSR length parameter
        P: Projection dimension
        bits: Number of bits for quantization
        coefficients: Feedback polynomial coefficients
        
    Returns:
        Compressed layer
    """
    return LFSRCompressedLayer(layer, K=K, P=P, bits=bits, coefficients=coefficients)

def calculate_compression_ratio(original_model, compressed_model):
    """
    Calculate the compression ratio achieved.
    
    Args:
        original_model: Original PyTorch model
        compressed_model: Compressed model
        
    Returns:
        Compression ratio (compressed size / original size)
    """
    # Count original parameters
    original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    
    # Count parameters in compressed model
    compressed_params = 0
    for name, param in compressed_model.named_parameters():
        if param.requires_grad:
            compressed_params += param.numel()
    
    # Add overhead for seeds, exponents, etc.
    for name, module in compressed_model.named_modules():
        if isinstance(module, LFSRCompressedLayer):
            for param_name, param in module.original_layer.named_parameters():
                if isinstance(param, LFSRCompressedParameter):
                    # Add overhead: 1 for seed, 1 for exponent
                    compressed_params += 2
    
    return compressed_params / original_params
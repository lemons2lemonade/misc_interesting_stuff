import pytest
import torch
import math
import numpy as np
import time
import logging
import os
import sys
from typing import Tuple, Optional
from torch.autograd import gradcheck

# Adjust paths to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import the flash attention modules - use try/except to handle import errors gracefully
try:
    from flashattention.flash_attention_v23 import (
        flash_attention,
        flash_attention_with_kv_cache,
        reorder_kv_cache,
        KVCache,
        initialize_kv_cache,
        update_kv_cache,
        incremental_attention,
        single_token_attention,
        sliding_window_attention
    )
except ImportError as e:
    print(f"WARNING: Could not import flash_attention_v3 modules: {e}")
    print("Some tests may be skipped.")

    # Define placeholder KVCache class to prevent syntax errors
    class KVCache:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for testing
BATCH_SIZE = 2
NUM_HEADS = 4
HEAD_DIM = 64
MAX_SEQ_LEN = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skip tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tests require CUDA"
)

# Skip tests if flash attention modules are not available
requires_flash_attention = pytest.mark.skipif(
    'flash_attention' not in globals(),
    reason="Flash Attention modules not available"
)

# Utility function to compare tensors with a tolerance
def assert_close(a, b, rtol=1e-3, atol=1e-3):
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"Max diff: {(a - b).abs().max().item()}"

def create_random_attention_inputs(
    batch_size: int = BATCH_SIZE,
    num_heads: int = NUM_HEADS,
    seq_len_q: int = 16,
    seq_len_kv: int = 16,
    head_dim: int = HEAD_DIM,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float16,
    requires_grad: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random attention inputs for testing"""
    torch.manual_seed(42)  # For reproducibility
    
    q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, 
                   device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, 
                   device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, 
                   device=device, dtype=dtype, requires_grad=requires_grad)
    
    return q, k, v

def compute_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False
) -> torch.Tensor:
    """Compute attention using PyTorch's native operations"""
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_kv, _, _ = k.shape
    
    # Reshape for batch matrix multiplication
    q_flat = q.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_len_q, head_dim]
    k_flat = k.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_len_kv, head_dim]
    v_flat = v.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_len_kv, head_dim]
    
    # Scale query
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Compute attention scores
    attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * softmax_scale
    
    # Apply causal mask if needed
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv, device=q.device), 
            diagonal=1
        ).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Apply explicit attention mask if provided
    if attn_mask is not None:
        attn_scores.masked_fill_(~attn_mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v_flat)
    
    # Reshape back to original format
    return output.transpose(1, 2).contiguous()  # [batch_size, seq_len_q, num_heads, head_dim]

@pytest.fixture
def attention_inputs():
    """Create inputs for attention tests"""
    batch_size, seq_len, num_heads, head_dim = BATCH_SIZE, 16, NUM_HEADS, HEAD_DIM
    q, k, v = create_random_attention_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        head_dim=head_dim
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    return {
        "q": q, "k": k, "v": v, 
        "softmax_scale": softmax_scale,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim
    }

@requires_cuda
@requires_flash_attention
class TestFlashAttention:
    """Tests for the basic Flash Attention functionality"""
    
    def test_flash_attention_output_shape(self, attention_inputs):
        """Test that FlashAttention returns the correct output shape."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Test with non-causal attention
        output = flash_attention(
            q, k, v, 
            softmax_scale=attention_inputs["softmax_scale"],
            causal=False
        )
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        
        # Test with causal attention
        output = flash_attention(
            q, k, v, 
            softmax_scale=attention_inputs["softmax_scale"],
            causal=True
        )
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    
    def test_causal_mask(self, attention_inputs):
        """Test that causal attention properly masks future tokens."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        
        # Run with causal masking
        causal_output = flash_attention(q, k, v, softmax_scale=softmax_scale, causal=True)
        
        # Run with non-causal masking for comparison
        non_causal_output = flash_attention(q, k, v, softmax_scale=softmax_scale, causal=False)
        
        # Results should be different
        assert not torch.allclose(causal_output, non_causal_output, rtol=1e-2, atol=1e-2)
        
        # Compare with reference implementation
        ref_causal_output = compute_reference_attention(q, k, v, softmax_scale=softmax_scale, causal=True)
        
        # Check approximate correctness (not exact due to different computation patterns)
        # but should be relatively close
        assert_close(causal_output, ref_causal_output, rtol=1e-2, atol=1e-2)
        
        # Early positions should be more similar than later positions in causal vs non-causal
        # (since early positions see fewer future tokens in both cases)
        early_diff = (non_causal_output[:, :2] - causal_output[:, :2]).abs().mean()
        late_diff = (non_causal_output[:, -2:] - causal_output[:, -2:]).abs().mean()
        
        assert early_diff < late_diff, "Early positions should have smaller differences than later positions"
    
    @pytest.mark.parametrize("seq_len_q,seq_len_kv", [(16, 16), (8, 16), (16, 8)])
    def test_variable_sequence_lengths(self, seq_len_q, seq_len_kv):
        """Test with different sequence lengths for queries and keys/values."""
        batch_size, num_heads, head_dim = BATCH_SIZE, NUM_HEADS, HEAD_DIM
        
        q, k, v = create_random_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            head_dim=head_dim
        )
        
        # Run the flash attention
        output = flash_attention(q, k, v)
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len_q, num_heads, head_dim)
        
        # Verify against reference implementation
        ref_output = compute_reference_attention(q, k, v)
        assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through flash attention."""
        batch_size, num_heads, seq_len, head_dim = BATCH_SIZE, NUM_HEADS, 16, HEAD_DIM
        
        # Create inputs with requires_grad=True
        q, k, v = create_random_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            head_dim=head_dim,
            requires_grad=True
        )
        
        # Forward pass
        output = flash_attention(q, k, v)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        # Check that gradients are computed
        assert q.grad is not None, "No gradients for query"
        assert k.grad is not None, "No gradients for key"
        assert v.grad is not None, "No gradients for value"
        
        # Check gradient shapes
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

@requires_cuda
@requires_flash_attention
class TestKVCache:
    """Tests for KV cache creation and usage"""
    
    def test_kv_cache_creation(self, attention_inputs):
        """Test KV cache creation and usage."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        
        # Initial forward pass with KV cache creation
        output, kv_cache = flash_attention_with_kv_cache(
            q, k, v,
            kv_cache=None,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Check KV cache shapes
        assert kv_cache.keys.shape == (batch_size, seq_len, num_heads, head_dim)
        assert kv_cache.values.shape == (batch_size, seq_len, num_heads, head_dim)
        
        # Verify that output matches regular flash attention
        regular_output = flash_attention(q, k, v, softmax_scale=softmax_scale, causal=True)
        assert_close(output, regular_output)
    
    def test_kv_cache_reuse(self, attention_inputs):
        """Test reusing KV cache for incremental decoding."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        
        # Step 1: Generate cache with initial sequence
        initial_q = q[:, :seq_len-1]
        initial_k = k[:, :seq_len-1]
        initial_v = v[:, :seq_len-1]
        
        _, kv_cache = flash_attention_with_kv_cache(
            initial_q, initial_k, initial_v,
            kv_cache=None,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Step 2: Generate single new token with cache
        new_q = q[:, seq_len-1:seq_len]
        new_k = k[:, seq_len-1:seq_len]
        new_v = v[:, seq_len-1:seq_len]
        
        incremental_output, updated_kv_cache = flash_attention_with_kv_cache(
            new_q, new_k, new_v,
            kv_cache=kv_cache,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Step 3: Verify results match full sequence processing
        full_output = flash_attention(q, k, v, softmax_scale=softmax_scale, causal=True)
        assert_close(incremental_output, full_output[:, seq_len-1:seq_len])
        
        # Verify updated cache shapes
        assert updated_kv_cache.keys.shape == (batch_size, seq_len, num_heads, head_dim)
        assert updated_kv_cache.values.shape == (batch_size, seq_len, num_heads, head_dim)
    
    def test_initialize_kv_cache(self):
        """Test that initializing KV cache works correctly"""
        cache = initialize_kv_cache(BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        
        # Check properties
        assert hasattr(cache, 'keys'), "KVCache should have keys attribute"
        assert hasattr(cache, 'values'), "KVCache should have values attribute"
        
        # Check shapes and device
        assert cache.keys.shape[0] == BATCH_SIZE
        assert cache.values.shape[0] == BATCH_SIZE
        assert cache.keys.device.type == "cuda"
        assert cache.values.device.type == "cuda"
    
    def test_update_kv_cache(self):
        """Test that updating KV cache works correctly"""
        # Initialize cache
        cache = initialize_kv_cache(BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        
        # Create some random k and v tensors
        k = torch.randn(BATCH_SIZE, 10, NUM_HEADS, HEAD_DIM, device=DEVICE)
        v = torch.randn(BATCH_SIZE, 10, NUM_HEADS, HEAD_DIM, device=DEVICE)
        
        # Update cache
        update_kv_cache(cache, k, v)
        
        # Add more entries
        k2 = torch.randn(BATCH_SIZE, 5, NUM_HEADS, HEAD_DIM, device=DEVICE)
        v2 = torch.randn(BATCH_SIZE, 5, NUM_HEADS, HEAD_DIM, device=DEVICE)
        
        update_kv_cache(cache, k2, v2)
        
        # Use the updated cache
        q_new = torch.randn(BATCH_SIZE, 1, NUM_HEADS, HEAD_DIM, device=DEVICE)
        output = incremental_attention(q_new, cache, causal=True)
        
        # Check output shape
        assert output.shape == (BATCH_SIZE, 1, NUM_HEADS, HEAD_DIM)

@requires_cuda
@requires_flash_attention
class TestBeamSearch:
    """Tests for beam search with KV caching"""
    
    def test_beam_search_kv_cache(self, attention_inputs):
        """Test beam search reordering of KV cache."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        beam_size = 3
        
        # Step 1: Create initial KV cache
        _, kv_cache = flash_attention_with_kv_cache(
            q, k, v,
            kv_cache=None,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Step 2: Expand cache for beam search
        expanded_keys = kv_cache.keys.repeat_interleave(beam_size, dim=0)
        expanded_values = kv_cache.values.repeat_interleave(beam_size, dim=0)
        expanded_cache = KVCache(expanded_keys, expanded_values)
        
        # Check expanded cache shapes
        assert expanded_cache.keys.shape == (batch_size * beam_size, seq_len, num_heads, head_dim)
        assert expanded_cache.values.shape == (batch_size * beam_size, seq_len, num_heads, head_dim)
        
        # Step 3: Create a beam index tensor
        # For example, to select [beam 1, beam 0, beam 2] for first batch element
        # and [beam 2, beam 1, beam 0] for second batch element
        beam_idx = torch.tensor([
            1, 0, 2,  # First batch element
            5, 4, 3   # Second batch element
        ], device="cuda")
        
        # Save a copy of the original cache
        original_keys = expanded_cache.keys.clone()
        original_values = expanded_cache.values.clone()
        
        # Step 4: Apply beam reordering
        reorder_kv_cache(expanded_cache, beam_idx)
        
        # Step 5: Verify results - each new position should have data from the specified beam
        for i, beam_id in enumerate(beam_idx):
            # Verify keys and values were properly reordered
            assert_close(
                expanded_cache.keys[i], 
                original_keys[beam_id]
            )
            assert_close(
                expanded_cache.values[i], 
                original_values[beam_id]
            )
    
    def test_beam_search_incremental(self, attention_inputs):
        """Test beam search with incremental decoding."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        beam_size = 2
        
        # Initial forward pass to build cache
        output, kv_cache = flash_attention_with_kv_cache(
            q, k, v,
            kv_cache=None,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Expand cache for beam search
        expanded_keys = kv_cache.keys.repeat_interleave(beam_size, dim=0)
        expanded_values = kv_cache.values.repeat_interleave(beam_size, dim=0)
        kv_cache = KVCache(expanded_keys, expanded_values)
        
        # New token for incremental decoding
        new_q = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        new_k = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        new_v = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        
        # First incremental step
        output1, kv_cache = flash_attention_with_kv_cache(
            new_q, new_k, new_v,
            kv_cache=kv_cache,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Create beam indices for reordering
        beam_idx = torch.tensor([
            1, 0,  # First batch element
            3, 2   # Second batch element
        ], device="cuda")
        
        # Apply beam reordering
        reorder_kv_cache(kv_cache, beam_idx)
        
        # Continue with another token
        new_q2 = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        new_k2 = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        new_v2 = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, 
                            device="cuda", dtype=torch.float16)
        
        # Second incremental step
        output2, kv_cache = flash_attention_with_kv_cache(
            new_q2, new_k2, new_v2,
            kv_cache=kv_cache,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Ensure shapes are correct
        assert output2.shape == (batch_size * beam_size, 1, num_heads, head_dim)
        assert kv_cache.keys.shape == (batch_size * beam_size, seq_len + 2, num_heads, head_dim)
        assert kv_cache.values.shape == (batch_size * beam_size, seq_len + 2, num_heads, head_dim)
    
    def test_invalid_beam_indices(self, attention_inputs):
        """Test error handling for invalid beam indices."""
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = attention_inputs["softmax_scale"]
        beam_size = 2
        
        # Create KV cache
        _, kv_cache = flash_attention_with_kv_cache(
            q, k, v,
            kv_cache=None,
            softmax_scale=softmax_scale,
            causal=True
        )
        
        # Expand for beam search
        expanded_keys = kv_cache.keys.repeat_interleave(beam_size, dim=0)
        expanded_values = kv_cache.values.repeat_interleave(beam_size, dim=0)
        kv_cache = KVCache(expanded_keys, expanded_values)
        
        # Test with invalid beam indices shape (2D tensor)
        invalid_beam_idx = torch.ones((batch_size, beam_size), device="cuda", dtype=torch.long)
        with pytest.raises(AssertionError):
            reorder_kv_cache(kv_cache, invalid_beam_idx)
        
        # Test with invalid beam indices size (too few elements)
        invalid_beam_idx = torch.ones(batch_size, device="cuda", dtype=torch.long)
        with pytest.raises(AssertionError):
            reorder_kv_cache(kv_cache, invalid_beam_idx)

@requires_cuda
@requires_flash_attention
class TestSingleTokenOptimization:
    """Tests for the single-token optimization"""
    
    def test_single_token_correctness(self):
        """Test that single-token attention produces correct results"""
        # Create random inputs with single-token query
        q, k, v = create_random_attention_inputs(
            seq_len_q=1, 
            seq_len_kv=100
        )
        
        # Initialize KV cache and update it
        cache = initialize_kv_cache(BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        
        # Reshape for KV cache update
        k_reshaped = k.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_len_kv, head_dim]
        v_reshaped = v.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_len_kv, head_dim]
        update_kv_cache(cache, k_reshaped, v_reshaped)
        
        # Reshape q for single token attention
        q_reshaped = q.transpose(1, 2).contiguous()  # [batch_size, num_heads, 1, head_dim]
        
        # Compute with single-token optimization
        output_opt = single_token_attention(q_reshaped, cache)
        
        # Compute with general incremental attention
        output_gen = incremental_attention(q_reshaped, cache)
        
        # Results should be identical
        assert torch.allclose(output_opt, output_gen, rtol=1e-3, atol=1e-3), \
            f"Max difference: {(output_opt - output_gen).abs().max().item()}"
        
        # Compute reference output
        ref_output = compute_reference_attention(q, k, v)
        
        # Reshape for comparison
        output_opt_reshaped = output_opt.transpose(1, 2).contiguous()  # [batch_size, 1, num_heads, head_dim]
        
        # Check approximate correctness (not exact due to different computation patterns)
        assert torch.allclose(output_opt_reshaped, ref_output, rtol=1e-2, atol=1e-2), \
            f"Max difference: {(output_opt_reshaped - ref_output).abs().max().item()}"
    
    def test_single_token_performance(self):
        """Test that single-token optimization improves performance"""
        # Create random inputs with larger dimensions for more accurate timing
        batch_size = 4
        num_heads = 16
        head_dim = 64
        seq_len_kv = 1024
        
        # Create inputs
        q = torch.randn(batch_size, 1, num_heads, head_dim, device=DEVICE)
        k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, device=DEVICE)
        v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, device=DEVICE)
        
        # Reshape for KV cache
        k_reshaped = k.transpose(1, 2).contiguous()
        v_reshaped = v.transpose(1, 2).contiguous()
        q_reshaped = q.transpose(1, 2).contiguous()
        
        # Initialize KV cache and update it
        cache = initialize_kv_cache(batch_size, num_heads, MAX_SEQ_LEN, head_dim)
        update_kv_cache(cache, k_reshaped, v_reshaped)
        
        # Warm up
        for _ in range(5):
            single_token_attention(q_reshaped, cache)
            incremental_attention(q_reshaped, cache)
        
        # Time single-token optimization (10 iterations)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            single_token_attention(q_reshaped, cache)
        torch.cuda.synchronize()
        time_opt = time.time() - start
        
        # Time general incremental attention (10 iterations)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            incremental_attention(q_reshaped, cache)
        torch.cuda.synchronize()
        time_gen = time.time() - start
        
        # Log the timing results
        logger.info(f"Single-token optimization time: {time_opt * 1000:.2f} ms")
        logger.info(f"General incremental attention time: {time_gen * 1000:.2f} ms")
        logger.info(f"Speedup factor: {time_gen / time_opt:.2f}x")
        
        # The optimization should generally provide some speedup,
        # but we don't assert on the exact value since it can vary by hardware
        # and system load. Just logging is sufficient for this test.

@requires_cuda
@requires_flash_attention
class TestSlidingWindowAttention:
    """Tests for the sliding window attention"""
    
    def test_sliding_window_correctness(self):
        """Test that sliding window attention produces correct results for window_size = seq_len"""
        # Create random inputs
        q, k, v = create_random_attention_inputs(
            seq_len_q=10, 
            seq_len_kv=100
        )
        
        # Reshape for KV cache
        k_reshaped = k.transpose(1, 2).contiguous()
        v_reshaped = v.transpose(1, 2).contiguous()
        q_reshaped = q.transpose(1, 2).contiguous()
        
        # Initialize KV cache and update it
        cache = initialize_kv_cache(BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        update_kv_cache(cache, k_reshaped, v_reshaped)
        
        # Compute with sliding window (full window)
        output_sw = sliding_window_attention(q_reshaped, cache, window_size=100)
        
        # Compute with general incremental attention
        output_gen = incremental_attention(q_reshaped, cache)
        
        # Results should be identical with full window
        assert torch.allclose(output_sw, output_gen, rtol=1e-3, atol=1e-3), \
            f"Max difference: {(output_sw - output_gen).abs().max().item()}"
    
    def test_sliding_window_masking(self):
        """Test that sliding window attention correctly masks out tokens outside the window"""
        # Create random inputs with a longer sequence
        batch_size = BATCH_SIZE
        num_heads = NUM_HEADS
        head_dim = HEAD_DIM
        seq_len_kv = 200
        
        q = torch.randn(batch_size, 1, num_heads, head_dim, device=DEVICE)
        k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, device=DEVICE)
        v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, device=DEVICE)
        
        # Make v distinctive per position to clearly see which positions contribute
        for i in range(seq_len_kv):
            v[:, i] = i / seq_len_kv
        
        # Reshape for KV cache
        k_reshaped = k.transpose(1, 2).contiguous()
        v_reshaped = v.transpose(1, 2).contiguous()
        q_reshaped = q.transpose(1, 2).contiguous()
        
        # Initialize KV cache and update it
        cache = initialize_kv_cache(batch_size, num_heads, MAX_SEQ_LEN, head_dim)
        update_kv_cache(cache, k_reshaped, v_reshaped)
        
        # Compute with different window sizes
        output_full = sliding_window_attention(q_reshaped, cache, window_size=seq_len_kv)
        output_window50 = sliding_window_attention(q_reshaped, cache, window_size=50)
        
        # Results should be different
        assert not torch.allclose(output_full, output_window50, rtol=1e-3, atol=1e-3)
        
        # Reshape outputs for comparison
        output_full_reshaped = output_full.transpose(1, 2).contiguous()
        output_window50_reshaped = output_window50.transpose(1, 2).contiguous()
        
        # The window-50 output should be closer to attending only to the last 50 positions
        # We test this by comparing means of the outputs
        assert output_window50_reshaped.mean() > 0.5, \
            "Window-50 output should be biased toward later tokens (which have higher values)"
        assert output_full_reshaped.mean() < output_window50_reshaped.mean(), \
            "Full window should have a lower mean than window-50 (which focuses on higher valued later tokens)"

@requires_cuda
@requires_flash_attention
class TestPerformance:
    """Performance benchmarks for Flash Attention"""
    
    @pytest.mark.benchmark
    def test_flash_vs_native_attention(self):
        """Benchmark Flash Attention against PyTorch's native attention."""
        # Use larger dimensions for more accurate benchmarking
        batch_size = 8
        num_heads = 16 
        seq_len = 1024
        head_dim = 64
        
        # Create inputs
        q, k, v = create_random_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            head_dim=head_dim
        )
        
        # Define the PyTorch native attention function
        def native_attention():
            # Reshape inputs
            q_flat = q.transpose(1, 2).contiguous()
            k_flat = k.transpose(1, 2).contiguous()
            v_flat = v.transpose(1, 2).contiguous()
            
            # Scale query
            scale = 1.0 / (head_dim ** 0.5)
            
            # Compute attention scores
            scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply attention weights to values
            output = torch.matmul(attn_weights, v_flat)
            
            # Return transposed output to match flash attention format
            return output.transpose(1, 2).contiguous()
        
        # Define the flash attention function
        def flash_attn():
            return flash_attention(q, k, v)
        
        # Warmup
        for _ in range(5):
            native_attention()
            flash_attn()
        
        # Benchmark native attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            native_attention()
        torch.cuda.synchronize()
        native_time = time.time() - start
        
        # Benchmark flash attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            flash_attn()
        torch.cuda.synchronize()
        flash_time = time.time() - start
        
        # Log results
        logger.info(f"Native attention time: {native_time * 1000:.2f} ms")
        logger.info(f"Flash attention time: {flash_time * 1000:.2f} ms")
        logger.info(f"Speedup factor: {native_time / flash_time:.2f}x")
        
        # Assertion is not strictly necessary for a benchmark,
        # but we can add a soft assertion that flash attention is faster
        assert flash_time < native_time, "Flash attention should be faster than native attention"
    
    @pytest.mark.benchmark
    def test_kv_cache_vs_full_recompute(self):
        """Benchmark KV caching against full recomputation."""
        # Parameters
        batch_size = 4
        num_heads = 16
        head_dim = 64
        context_len = 512
        generation_steps = 10
        
        # Create inputs for context
        q_ctx, k_ctx, v_ctx = create_random_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=context_len,
            seq_len_kv=context_len,
            head_dim=head_dim
        )
        
        # Initialize KV cache
        cache = initialize_kv_cache(batch_size, num_heads, 2048, head_dim)
        
        # Update cache with context
        k_ctx_reshaped = k_ctx.transpose(1, 2).contiguous()
        v_ctx_reshaped = v_ctx.transpose(1, 2).contiguous()
        update_kv_cache(cache, k_ctx_reshaped, v_ctx_reshaped)
        
        # Benchmark full recomputation
        torch.cuda.synchronize()
        start = time.time()
        
        # Current full sequence
        q_full = q_ctx
        k_full = k_ctx
        v_full = v_ctx
        
        for i in range(generation_steps):
            # New token to generate
            q_new, k_new, v_new = create_random_attention_inputs(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len_q=1,
                seq_len_kv=1,
                head_dim=head_dim
            )
            
            # Concatenate to full sequence
            q_full = torch.cat([q_full, q_new], dim=1)
            k_full = torch.cat([k_full, k_new], dim=1)
            v_full = torch.cat([v_full, v_new], dim=1)
            
            # Recompute full attention
            _ = flash_attention(q_full, k_full, v_full, causal=True)
        
        torch.cuda.synchronize()
        full_time = time.time() - start
        
        # Benchmark KV cache approach
        torch.cuda.synchronize()
        start = time.time()
        
        # Reset cache
        cache = initialize_kv_cache(batch_size, num_heads, 2048, head_dim)
        update_kv_cache(cache, k_ctx_reshaped, v_ctx_reshaped)
        
        for i in range(generation_steps):
            # New token to generate
            q_new, k_new, v_new = create_random_attention_inputs(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len_q=1,
                seq_len_kv=1,
                head_dim=head_dim
            )
            
            # Reshape for KV cache
            k_new_reshaped = k_new.transpose(1, 2).contiguous()
            v_new_reshaped = v_new.transpose(1, 2).contiguous()
            q_new_reshaped = q_new.transpose(1, 2).contiguous()
            
            # Use incremental attention
            _ = incremental_attention(q_new_reshaped, cache, causal=True)
            
            # Update cache
            update_kv_cache(cache, k_new_reshaped, v_new_reshaped)
        
        torch.cuda.synchronize()
        cache_time = time.time() - start
        
        # Log results
        logger.info(f"Full recomputation time: {full_time * 1000:.2f} ms")
        logger.info(f"KV cache approach time: {cache_time * 1000:.2f} ms")
        logger.info(f"Speedup factor: {full_time / cache_time:.2f}x")
        
        # KV cache should be faster than full recomputation
        assert cache_time < full_time, "KV cache should be faster than full recomputation"


if __name__ == "__main__":
    # Set up basic configuration for running tests directly
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    pytest.main(["-xvs", __file__])
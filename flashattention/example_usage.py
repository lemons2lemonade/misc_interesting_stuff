from flashattention.flash_attention_v3 import reorder_kv_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time
import math
import os

# Load the C++ extension
script_dir = os.path.dirname(os.path.abspath(__file__))
flash_attn_cpp = load(
    name="flash_attention_v3",
    sources=[os.path.join(script_dir, "flash_attention_partial_v23.cpp")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

# Flash Attention implementation as a nn.Module
class FlashAttention(nn.Module):
    """
    Flash Attention V3 implementation as a PyTorch module.
    
    This module provides a drop-in replacement for standard attention
    mechanisms with improved performance due to optimized CUDA kernels.
    """
    def __init__(self, softmax_scale=None, causal=False):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.causal = causal
    
    def forward(self, q, k, v, attn_mask=None):
        """
        Forward pass for Flash Attention
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            attn_mask: Optional boolean mask of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        
        Returns:
            output: Attention output of shape [batch_size, num_heads, seq_len_q, head_dim]
        """
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("FlashAttention requires CUDA tensors")
        
        # Compute softmax scale if not provided
        if self.softmax_scale is None:
            softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        else:
            softmax_scale = self.softmax_scale
        
        # Use our custom C++ implementation
        return FlashAttentionAutograd.apply(q, k, v, attn_mask, softmax_scale, self.causal)
    
    def forward_with_kv_cache(self, q, k_cache, v_cache, attn_mask=None):
        """
        Forward pass using pre-computed key-value cache for efficient inference.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            k_cache: Cached key tensor [batch_size, num_heads, seq_len_k, head_dim]
            v_cache: Cached value tensor [batch_size, num_heads, seq_len_k, head_dim]
            attn_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
            
        Returns:
            Output tensor [batch_size, num_heads, seq_len_q, head_dim]
        """
        if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda:
            raise ValueError("FlashAttention requires CUDA tensors")
        
        # Scale query
        if self.softmax_scale is None:
            softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        else:
            softmax_scale = self.softmax_scale
            
        # Use our custom C++ implementation with KV cache
        return FlashAttentionKVCacheAutograd.apply(q, k_cache, v_cache, attn_mask, softmax_scale, self.causal)
    
    def incremental_decode(self, q, kv_cache, attn_mask=None):
        """
        Optimized forward pass with KV cache for efficient autoregressive decoding.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            kv_cache: A KVCache object containing the cached keys and values
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, num_heads, seq_len_q, head_dim]
        """
        if not q.is_cuda:
            raise ValueError("FlashAttention requires CUDA tensors")
        
        # Compute softmax scale if not provided
        if self.softmax_scale is None:
            softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        else:
            softmax_scale = self.softmax_scale
        
        # Use optimized single-token path if applicable (common case in decoding)
        if q.size(2) == 1:
            return flash_attn_cpp.single_token_forward(
                q, kv_cache, attn_mask, softmax_scale, self.causal
            )
        
        # Call the general incremental forward function for multi-token case
        return flash_attn_cpp.incremental_forward(
            q, kv_cache, attn_mask, softmax_scale, self.causal
        )
        
    def sliding_window_decode(self, q, kv_cache, window_size=1024, attn_mask=None):
        """
        Sliding window attention for efficiently processing very long sequences.
        Only attends to the most recent `window_size` tokens in the KV cache.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            kv_cache: A KVCache object containing the cached keys and values
            window_size: Size of the sliding window (number of tokens to attend to)
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, num_heads, seq_len_q, head_dim]
        """
        if not q.is_cuda:
            raise ValueError("FlashAttention requires CUDA tensors")
        
        # Compute softmax scale if not provided
        if self.softmax_scale is None:
            softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        else:
            softmax_scale = self.softmax_scale
        
        return flash_attn_cpp.sliding_window_forward(
            q, kv_cache, attn_mask, window_size, softmax_scale, self.causal
        )

# Custom autograd function for Flash Attention
class FlashAttentionAutograd(torch.autograd.Function):
    """
    Custom autograd function for Flash Attention to enable explicit backward pass.
    """
    @staticmethod
    def forward(ctx, q, k, v, attn_mask, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        
        # Run the forward CUDA kernel
        output, logsumexp, max_score = flash_attn_cpp.forward(
            q, k, v, attn_mask, softmax_scale, causal
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, output, logsumexp, max_score)
        ctx.attn_mask = attn_mask
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, logsumexp, max_score = ctx.saved_tensors
        
        # Run the backward CUDA kernel
        dq, dk, dv = flash_attn_cpp.backward(
            grad_output, q, k, v, output, logsumexp, max_score,
            ctx.attn_mask, ctx.softmax_scale, ctx.causal
        )
        
        return dq, dk, dv, None, None, None

# Custom autograd function for Flash Attention with KV Cache
class FlashAttentionKVCacheAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, attn_mask, softmax_scale, causal):
        # Forward pass with KV cache
        output, logsumexp, max_score = flash_attn_cpp.forward_with_kv_cache(
            q, k_cache, v_cache, attn_mask, softmax_scale, causal
        )
        
        # Save inputs and intermediate results (needed for backward pass)
        ctx.save_for_backward(q, k_cache, v_cache, output, logsumexp, max_score)
        ctx.attn_mask = attn_mask
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        q, k_cache, v_cache, output, logsumexp, max_score = ctx.saved_tensors
        
        # For simplicity, just use the normal backward for now
        # In a full implementation, you might want a specialized backward
        # particularly for the KV cache scenario
        dq, dk, dv = flash_attn_cpp.backward(
            grad_output, q, k_cache, v_cache, output, logsumexp, max_score,
            ctx.attn_mask, ctx.softmax_scale, ctx.causal
        )
        
        return dq, dk, dv, None, None, None

# Functional interface
def flash_attention(q, k, v, attn_mask=None, softmax_scale=None, causal=False):
    """
    Functional interface to flash attention
    
    Args:
        q, k, v: Query, key, value tensors [batch_size, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
        softmax_scale: Scaling factor (if None, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, num_heads, seq_len, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    return FlashAttentionAutograd.apply(q, k, v, attn_mask, softmax_scale, causal)

# Functional interface with KV cache
def flash_attention_with_kv_cache(q, k_cache, v_cache, attn_mask=None, softmax_scale=None, causal=False):
    """
    Functional interface to flash attention with key-value cache
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k_cache: Cached key tensor [batch_size, num_heads, seq_len_k, head_dim]
        v_cache: Cached value tensor [batch_size, num_heads, seq_len_k, head_dim]
        attn_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
        softmax_scale: Scaling factor (if None, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    return FlashAttentionKVCacheAutograd.apply(q, k_cache, v_cache, attn_mask, softmax_scale, causal)

# Helper functions for optimized KV caching
def initialize_kv_cache(batch_size, num_heads, max_seq_len, head_dim, device="cuda"):
    """
    Initialize a new KV cache for incremental decoding
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length the cache can hold
        head_dim: Head dimension
        device: Device to create cache on
        
    Returns:
        A KVCache object
    """
    options = torch.empty(0, device=device).options()
    return flash_attn_cpp.initialize_kv_cache(
        batch_size, num_heads, max_seq_len, head_dim, options
    )

def update_kv_cache(kv_cache, new_keys, new_values):
    """
    Update the KV cache with new keys and values
    
    Args:
        kv_cache: The KVCache object to update
        new_keys: New keys to add [batch_size, num_heads, new_seq_len, head_dim]
        new_values: New values to add [batch_size, num_heads, new_seq_len, head_dim]
    """
    flash_attn_cpp.update_kv_cache(kv_cache, new_keys, new_values)
    
def incremental_attention(q, kv_cache, attn_mask=None, softmax_scale=None, causal=False):
    """
    Functional interface to optimized incremental attention with KV cache
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        kv_cache: KVCache object containing cached keys and values
        attn_mask: Optional attention mask
        softmax_scale: Scaling factor (if None, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    # Use single-token optimization if applicable
    if q.size(2) == 1:
        return single_token_attention(q, kv_cache, attn_mask, softmax_scale, causal)
    
    return flash_attn_cpp.incremental_forward(
        q, kv_cache, attn_mask, softmax_scale, causal
    )

def single_token_attention(q, kv_cache, attn_mask=None, softmax_scale=None, causal=False):
    """
    Highly optimized attention for single-token generation, the most common case in inference
    
    Args:
        q: Query tensor [batch_size, num_heads, 1, head_dim]
        kv_cache: KVCache object containing cached keys and values
        attn_mask: Optional attention mask
        softmax_scale: Scaling factor (if None, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, num_heads, 1, head_dim]
    """
    assert q.size(2) == 1, "This function is optimized for single-token generation"
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    return flash_attn_cpp.single_token_forward(
        q, kv_cache, attn_mask, softmax_scale, causal
    )

def sliding_window_attention(q, kv_cache, window_size=1024, attn_mask=None, softmax_scale=None, causal=False):
    """
    Sliding window attention for efficiently processing very long sequences
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        kv_cache: KVCache object containing cached keys and values
        window_size: Size of the sliding window (number of tokens to attend to)
        attn_mask: Optional attention mask
        softmax_scale: Scaling factor (if None, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    return flash_attn_cpp.sliding_window_forward(
        q, kv_cache, attn_mask, window_size, softmax_scale, causal
    )

# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 4
    num_heads = 8
    seq_len_q = 1024
    seq_len_kv = 1024
    head_dim = 64
    
    # Create sample inputs
    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device='cuda')
    
    # Optional: Create causal mask
    causal = True
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Method 1: Using the module
    flash_attn_module = FlashAttention(causal=causal)
    output_module = flash_attn_module(q, k, v)
    
    print(f"Output shape from module: {output_module.shape}")
    
    # Method 2: Using the functional interface
    output_func = flash_attention(q, k, v, causal=causal)
    
    print(f"Output shape from function: {output_func.shape}")
    
    # Verify gradient flow
    output_func.sum().backward()
    
    print(f"Gradients successfully computed: q.grad={q.grad is not None}, k.grad={k.grad is not None}, v.grad={v.grad is not None}")
    
    # Benchmark against PyTorch's native attention
    def benchmark(fn, name, repeat=10):
        # Warmup
        for _ in range(3):
            fn()
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(repeat):
            fn()
        
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"{name}: {(end - start) / repeat * 1000:.2f} ms per iteration")
    
    # Flash Attention
    def run_flash():
        with torch.no_grad():
            flash_attention(q, k, v, causal=causal)
    
    # PyTorch native attention
    def run_pytorch():
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            if causal:
                mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device='cuda'), diagonal=1).bool()
                scores.masked_fill_(mask, -float('inf'))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
    
    print("\nBenchmarking:")
    benchmark(run_flash, "Flash Attention V3")
    benchmark(run_pytorch, "PyTorch Native Attention")
    
    # === KV Caching Example ===
    print("\n=== KV Caching Example ===")
    
    # Create input tensors
    batch_size = 2
    num_heads = 4
    full_seq_len = 512
    head_dim = 64
    
    # Initial sequence
    q1 = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    k1 = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    v1 = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Initial forward pass to build cache
    attn = FlashAttention(causal=True)
    flash_out1 = attn(q1, k1, v1)
    
    # Next token
    q2 = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Concatenate to build key-value cache
    k_cache = torch.cat([k1, torch.randn(batch_size, num_heads, 4, head_dim, device="cuda")], dim=2)
    v_cache = torch.cat([v1, torch.randn(batch_size, num_heads, 4, head_dim, device="cuda")], dim=2)
    
    # Use cached version
    flash_out2 = attn.forward_with_kv_cache(q2, k_cache, v_cache)
    
    print(f"Output with KV cache shape: {flash_out2.shape}")
    
    # Benchmark KV caching vs full computation
    print("\n=== KV Cache Benchmark ===")
    
    # Simulate growing sequence
    seq_len = 1024
    context_len = 900
    
    # Create inputs
    q_prefix = torch.randn(batch_size, num_heads, context_len, head_dim, device="cuda")
    k_prefix = torch.randn(batch_size, num_heads, context_len, head_dim, device="cuda")
    v_prefix = torch.randn(batch_size, num_heads, context_len, head_dim, device="cuda")
    
    q_next = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Warm-up
    _ = attn(q_prefix, k_prefix, v_prefix)
    _ = attn.forward_with_kv_cache(q_next, k_prefix, v_prefix)
    
    # Time standard approach (recomputing everything)
    torch.cuda.synchronize()
    start_time = time.time()
    
    q_full = torch.cat([q_prefix, q_next], dim=2)
    k_full = torch.cat([k_prefix, torch.zeros(batch_size, num_heads, 1, head_dim, device="cuda")], dim=2)
    v_full = torch.cat([v_prefix, torch.zeros(batch_size, num_heads, 1, head_dim, device="cuda")], dim=2)
    full_out = attn(q_full, k_full, v_full)
    last_token_output_full = full_out[:, :, -1:, :]
    
    torch.cuda.synchronize()
    full_time = time.time() - start_time
    
    # Time KV cache approach
    torch.cuda.synchronize()
    start_time = time.time()
    
    cached_out = attn.forward_with_kv_cache(q_next, k_prefix, v_prefix)
    
    torch.cuda.synchronize()
    cache_time = time.time() - start_time
    
    print(f"Full recomputation time: {full_time * 1000:.2f} ms")
    print(f"KV cache computation time: {cache_time * 1000:.2f} ms")
    print(f"Speedup factor: {full_time / cache_time:.2f}x")
    
    # Verify outputs are the same
    if torch.allclose(last_token_output_full, cached_out, rtol=1e-3, atol=1e-4):
        print("✓ Outputs match between full computation and KV cache!")
    else:
        print("✗ Outputs don't match between full computation and KV cache.")
        print(f"Max absolute difference: {(last_token_output_full - cached_out).abs().max().item()}")

    # Memory usage comparison
    torch.cuda.reset_peak_memory_stats()
    
    # Full computation memory usage
    _ = attn(q_full, k_full, v_full)
    full_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    torch.cuda.reset_peak_memory_stats()
    
    # KV cache memory usage
    _ = attn.forward_with_kv_cache(q_next, k_prefix, v_prefix)
    cache_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"Full computation peak memory: {full_memory:.2f} MB")
    print(f"KV cache peak memory: {cache_memory:.2f} MB")
    print(f"Memory reduction: {(full_memory - cache_memory) / full_memory * 100:.2f}%")

    # === Optimized KV Cache Example ===
    print("\n=== Optimized KV Cache Example ===")
    
    # Initialize optimized KV cache for a maximum sequence length
    max_seq_len = 1024
    kv_cache = initialize_kv_cache(batch_size, num_heads, max_seq_len, head_dim)
    print(f"Initialized KV cache with capacity for {max_seq_len} tokens")
    
    # Generate first token
    q_first = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    k_first = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    v_first = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Update the cache with first token's KV
    update_kv_cache(kv_cache, k_first, v_first)
    print(f"Updated KV cache with first token, current length: {kv_cache.current_length}")
    
    # Compute attention for first token
    output_first = incremental_attention(q_first, kv_cache, causal=True)
    print(f"Output shape for first token: {output_first.shape}")
    
    # Simulate generating next token
    q_next = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    k_next = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    v_next = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Update cache with next token
    update_kv_cache(kv_cache, k_next, v_next)
    print(f"Updated KV cache with next token, current length: {kv_cache.current_length}")
    
    # Compute attention for next token (uses entire cache)
    output_next = incremental_attention(q_next, kv_cache, causal=True)
    print(f"Output shape for next token: {output_next.shape}")
    
    # Benchmark optimized KV cache vs regular KV cache approach
    print("\n=== Optimized KV Cache Benchmark ===")
    
    # Simulate growing sequence
    seq_len = 1024
    prefix_len = 900
    generation_steps = 10
    
    # Create prefix tensors
    q_prefix = torch.randn(batch_size, num_heads, prefix_len, head_dim, device="cuda")
    k_prefix = torch.randn(batch_size, num_heads, prefix_len, head_dim, device="cuda")
    v_prefix = torch.randn(batch_size, num_heads, prefix_len, head_dim, device="cuda")
    
    # Initialize optimized cache for benchmark
    opt_cache = initialize_kv_cache(batch_size, num_heads, prefix_len + generation_steps, head_dim)
    
    # Update cache with prefix
    update_kv_cache(opt_cache, k_prefix, v_prefix)
    
    # Query for next token
    q_gen = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    k_gen = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    v_gen = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    
    # Warm-up
    for _ in range(3):
        # Regular KV cache approach
        _ = attn.forward_with_kv_cache(q_gen, 
                                       torch.cat([k_prefix, k_gen], dim=2)[:, :, :prefix_len+1], 
                                       torch.cat([v_prefix, v_gen], dim=2)[:, :, :prefix_len+1])
        
        # Optimized way
        _ = attn.incremental_decode(q_gen, opt_cache)
    
    # Time regular KV cache approach
    torch.cuda.synchronize()
    start_time = time.time()
    
    k_cache_full = k_prefix.clone()
    v_cache_full = v_prefix.clone()
    
    for i in range(generation_steps):
        # Generate new token
        q_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        k_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        v_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        
        # Append to full cache
        k_cache_full = torch.cat([k_cache_full, k_step], dim=2)
        v_cache_full = torch.cat([v_cache_full, v_step], dim=2)
        
        # Run with regular KV cache approach
        _ = attn.forward_with_kv_cache(q_step, k_cache_full, v_cache_full)
    
    torch.cuda.synchronize()
    regular_cache_time = time.time() - start_time
    
    # Reset for fair comparison
    torch.cuda.empty_cache()
    opt_cache = initialize_kv_cache(batch_size, num_heads, prefix_len + generation_steps, head_dim)
    update_kv_cache(opt_cache, k_prefix, v_prefix)
    
    # Time optimized KV cache approach
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(generation_steps):
        # Generate new token
        q_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        k_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        v_step = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
        
        # Run optimized approach
        _ = attn.incremental_decode(q_step, opt_cache)
        
        # Update cache
        update_kv_cache(opt_cache, k_step, v_step)
    
    torch.cuda.synchronize()
    opt_cache_time = time.time() - start_time
    
    print(f"Regular KV cache time: {regular_cache_time * 1000:.2f} ms")
    print(f"Optimized KV cache time: {opt_cache_time * 1000:.2f} ms")
    print(f"Speedup factor: {regular_cache_time / opt_cache_time:.2f}x")
    
    # Memory usage comparison
    torch.cuda.reset_peak_memory_stats()
    
    # Regular KV cache memory usage
    k_cache_bench = torch.cat([k_prefix, k_gen], dim=2)
    v_cache_bench = torch.cat([v_prefix, v_gen], dim=2)
    _ = attn.forward_with_kv_cache(q_gen, k_cache_bench, v_cache_bench)
    regular_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    torch.cuda.reset_peak_memory_stats()
    
    # Optimized KV cache memory usage
    _ = attn.incremental_decode(q_gen, opt_cache)
    opt_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"Regular KV cache peak memory: {regular_mem:.2f} MB")
    print(f"Optimized KV cache peak memory: {opt_mem:.2f} MB")
    print(f"Memory reduction: {(regular_mem - opt_mem) / regular_mem * 100:.2f}%")

    # Example of beam search with FlashAttention
    def beam_search_example():
        print("\n===== Flash Attention with Beam Search =====")
        
        # Parameters
        batch_size = 2
        seq_len = 10
        num_heads = 4
        head_dim = 64
        beam_size = 3
        
        # Initialize model
        flash_attn = FlashAttention(
            head_dim=head_dim,
            num_heads=num_heads,
            softmax_scale=1.0 / math.sqrt(head_dim),
            attention_dropout=0.0,
            causal=True,
            sliding_window=None,
            return_attn_weights=False
        )
        
        # Initial forward pass to build cache
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        
        # Create KV cache during initial forward pass
        output, kv_cache = flash_attention_with_kv_cache(
            q, k, v,
            kv_cache=None,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True
        )
        
        print(f"Initial KV cache shape - Keys: {kv_cache.keys.shape}, Values: {kv_cache.values.shape}")
        
        # Expand cache for beam search
        expanded_keys = kv_cache.keys.repeat_interleave(beam_size, dim=0)
        expanded_values = kv_cache.values.repeat_interleave(beam_size, dim=0)
        kv_cache = KVCache(expanded_keys, expanded_values)
        print(f"Expanded KV cache shape - Keys: {kv_cache.keys.shape}, Values: {kv_cache.values.shape}")
        
        # Simulation of beam search iterations
        for step in range(3):
            print(f"\nBeam search step {step+1}")
            
            # Generate new token
            new_q = torch.randn(batch_size * beam_size, 1, num_heads, head_dim, device="cuda", dtype=torch.float16)
            
            # Forward pass with KV cache
            output, kv_cache = flash_attention_with_kv_cache(new_q, k, v, kv_cache=kv_cache)
            
            # Beam selection indices
            beam_idx = torch.tensor([1, 0, 2, 5, 3, 4], device="cuda")
            
            # Reorder the KV cache based on beam selection
            reorder_kv_cache(kv_cache, beam_idx)
            
            print(f"KV cache shape after step {step+1} - Keys: {kv_cache.keys.shape}, Values: {kv_cache.values.shape}")
            
            # In a real scenario, we would now select the corresponding output tokens
            # based on beam_idx and continue the generation process
        
        print("\nBeam search completed successfully")

    benchmark_kv_cache()
    beam_search_example()
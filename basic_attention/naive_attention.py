import torch
import torch.nn.functional as F
from flash_attention_v2_python import MultiHeadAttention as FlashMultiHeadAttention

class NaiveAttention(torch.nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v, padding_mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask should be (batch_size, seq_len) with False for padding tokens
            # Expand to (batch_size, 1, 1, seq_len) for broadcasting
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            # Mask should have True for padding positions
            padding_mask = ~padding_mask.to(torch.bool)
            # Apply to scores
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

def compare_gradients(naive_grads, flash_grads, rtol=1e-5, atol=1e-8):
    all_close = True
    error_message = ""
    for i, (naive, flash) in enumerate(zip(naive_grads, flash_grads)):
        if not torch.allclose(naive, flash, rtol=rtol, atol=atol):
            all_close = False
            max_diff = torch.max(torch.abs(naive - flash))
            max_rel_diff = torch.max(torch.abs(naive - flash) / (torch.abs(naive) + 1e-8))
            error_message += f"Input {i}: Max absolute difference: {max_diff:.2e}, Max relative difference: {max_rel_diff:.2e}\n"
    return all_close, error_message

def test_backward_pass(batch_size, num_heads, seq_len, head_dim, causal=False, use_padding_mask=False):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True, device='cuda')

    # Create padding mask if needed
    padding_mask = None
    if use_padding_mask:
        # Create random padding mask where ~20% of positions are padding
        padding_mask = torch.rand(batch_size, seq_len, device='cuda') > 0.2

    # Naive attention
    naive_attn = NaiveAttention(causal=causal).cuda()
    naive_output = naive_attn(q, k, v, padding_mask)
    naive_loss = naive_output.sum()
    naive_loss.backward()
    naive_grads = [tensor.grad.clone() for tensor in (q, k, v)]

    # Reset gradients
    for tensor in (q, k, v):
        tensor.grad.zero_()

    # FlashAttention
    flash_attn = FlashMultiHeadAttention(head_dim * num_heads, num_heads, causal=causal).cuda()
    flash_output = flash_attn(q, k, v, padding_mask)
    flash_loss = flash_output.sum()
    flash_loss.backward()
    flash_grads = [tensor.grad.clone() for tensor in (q, k, v)]

    all_close, error_message = compare_gradients(naive_grads, flash_grads)
    return all_close, error_message

def run_backward_pass_tests():
    test_cases = [
        # (batch_size, num_heads, seq_len, head_dim, causal, use_padding_mask)
        (1, 1, 1, 32, False, False),  # Minimal case
        (2, 4, 100, 64, False, False),  # Standard case
        (1, 8, 1024, 64, False, False),  # Long sequence
        (8, 16, 256, 32, False, False),  # Many heads
        (2, 4, 100, 64, True, False),  # Causal attention
        (1, 8, 1024, 64, True, False),  # Long sequence with causal attention
        # Add test cases with padding masks
        (2, 4, 100, 64, False, True),   # Standard case with padding
        (2, 4, 100, 64, True, True),    # Causal attention with padding
        (1, 8, 512, 64, False, True),   # Longer sequence with padding
    ]

    for i, (batch_size, num_heads, seq_len, head_dim, causal, use_padding_mask) in enumerate(test_cases):
        print(f"Running test case {i + 1}:")
        print(f"batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}, causal={causal}, use_padding_mask={use_padding_mask}")
        
        all_close, error_message = test_backward_pass(batch_size, num_heads, seq_len, head_dim, causal, use_padding_mask)
        
        if all_close:
            print("Test passed: Gradients match closely.")
        else:
            print("Test failed: Gradient mismatch detected.")
            print(error_message)
        print()

if __name__ == "__main__":
    run_backward_pass_tests()

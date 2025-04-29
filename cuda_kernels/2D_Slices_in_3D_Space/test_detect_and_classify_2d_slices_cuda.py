# import torch
# import pytest
# from torch.autograd import gradcheck
# import detect_and_classify_2d_slices_cuda

# class TwoDSliceDetectionClassification(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, positions, softmax_scale, num_classes):
#         output, class_logits, bounding_boxes, l, m = detect_and_classify_2d_slices_cuda.forward(
#             input, positions, softmax_scale, num_classes)
#         ctx.save_for_backward(input, positions, output, class_logits, bounding_boxes, l, m)
#         ctx.softmax_scale = softmax_scale
#         ctx.num_classes = num_classes
#         return output, class_logits, bounding_boxes

#     @staticmethod
#     def backward(ctx, grad_output, grad_class_logits, grad_bounding_boxes):
#         input, positions, output, class_logits, bounding_boxes, l, m = ctx.saved_tensors
#         grad_input, grad_positions = detect_and_classify_2d_slices_cuda.backward(
#             input, positions, output, class_logits, bounding_boxes, l, m,
#             grad_output, grad_class_logits, grad_bounding_boxes, ctx.softmax_scale)
#         return grad_input, grad_positions, None, None

# @pytest.mark.parametrize("batch_size", [1, 2, 4])
# @pytest.mark.parametrize("num_heads", [1, 4, 8])
# @pytest.mark.parametrize("num_slices", [1, 3, 5])
# @pytest.mark.parametrize("seq_len", [16, 32, 64])
# @pytest.mark.parametrize("head_dim", [32, 64])
# @pytest.mark.parametrize("num_classes", [10, 20])
# @pytest.mark.parametrize("softmax_scale", [0.1, 1.0, 10.0])
# def test_forward(batch_size, num_heads, num_slices, seq_len, head_dim, num_classes, softmax_scale):
#     input = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', dtype=torch.float32)
#     positions = torch.randn(batch_size, num_slices, device='cuda', dtype=torch.float32)
    
#     output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, softmax_scale, num_classes)
    
#     assert output.shape == input.shape
#     assert class_logits.shape == (batch_size, num_slices, num_classes)
#     assert bounding_boxes.shape == (batch_size, num_slices, 4)
    
#     # Check if output is normalized
#     assert torch.allclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1)), atol=1e-6)
    
#     # Check if class_logits and bounding_boxes are within reasonable ranges
#     assert class_logits.min() >= 0 and class_logits.max() <= num_classes
#     assert bounding_boxes.min() >= 0 and bounding_boxes.max() <= 1

# @pytest.mark.parametrize("batch_size", [1, 2])
# @pytest.mark.parametrize("num_heads", [1, 4])
# @pytest.mark.parametrize("num_slices", [1, 3])
# @pytest.mark.parametrize("seq_len", [16, 32])
# @pytest.mark.parametrize("head_dim", [32, 64])
# @pytest.mark.parametrize("num_classes", [10, 20])
# @pytest.mark.parametrize("softmax_scale", [0.1, 1.0, 10.0])
# def test_backward(batch_size, num_heads, num_slices, seq_len, head_dim, num_classes, softmax_scale):
#     input = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', dtype=torch.double, requires_grad=True)
#     positions = torch.randn(batch_size, num_slices, device='cuda', dtype=torch.double, requires_grad=True)
    
#     assert gradcheck(TwoDSliceDetectionClassification.apply, (input, positions, softmax_scale, num_classes), eps=1e-6, atol=1e-4)

# def test_empty_input():
#     input = torch.empty(0, 4, 3, 16, 32, device='cuda', dtype=torch.float32)
#     positions = torch.empty(0, 3, device='cuda', dtype=torch.float32)
    
#     output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, 1.0, 10)
    
#     assert output.shape == (0, 4, 3, 16, 32)
#     assert class_logits.shape == (0, 3, 10)
#     assert bounding_boxes.shape == (0, 3, 4)

# def test_large_input():
#     input = torch.randn(8, 16, 20, 256, 64, device='cuda', dtype=torch.float32)
#     positions = torch.randn(8, 20, device='cuda', dtype=torch.float32)
    
#     output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, 1.0, 100)
    
#     assert output.shape == input.shape
#     assert class_logits.shape == (8, 20, 100)
#     assert bounding_boxes.shape == (8, 20, 4)

# def test_numerical_stability():
#     input = torch.randn(2, 4, 3, 16, 32, device='cuda', dtype=torch.float32)
#     positions = torch.randn(2, 3, device='cuda', dtype=torch.float32)
    
#     # Test with different softmax_scale values
#     for scale in [1e-5, 1.0, 1e5]:
#         output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, scale, 10)
#         assert not torch.isnan(output).any()
#         assert not torch.isinf(output).any()
#         assert not torch.isnan(class_logits).any()
#         assert not torch.isinf(class_logits).any()
#         assert not torch.isnan(bounding_boxes).any()
#         assert not torch.isinf(bounding_boxes).any()

# if __name__ == "__main__":
#     pytest.main([__file__])
import time
import torch
import pytest
from torch.autograd import gradcheck
import detect_and_classify_2d_slices_cuda

class TwoDSliceDetectionClassification(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, positions, softmax_scale, num_classes):
        output, class_logits, bounding_boxes, l, m = detect_and_classify_2d_slices_cuda.forward(
            input, positions, softmax_scale, num_classes)
        ctx.save_for_backward(input, positions, output, class_logits, bounding_boxes, l, m)
        ctx.softmax_scale = softmax_scale
        ctx.num_classes = num_classes
        return output, class_logits, bounding_boxes

    @staticmethod
    def backward(ctx, grad_output, grad_class_logits, grad_bounding_boxes):
        input, positions, output, class_logits, bounding_boxes, l, m = ctx.saved_tensors
        grad_input, grad_positions = detect_and_classify_2d_slices_cuda.backward(
            input, positions, output, class_logits, bounding_boxes, l, m,
            grad_output, grad_class_logits, grad_bounding_boxes, ctx.softmax_scale)
        return grad_input, grad_positions, None, None

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("num_slices", [1, 3, 5])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("num_classes", [10, 20])
@pytest.mark.parametrize("softmax_scale", [0.1, 1.0, 10.0])
def test_forward(batch_size, num_heads, num_slices, seq_len, head_dim, num_classes, softmax_scale):
    input = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', dtype=torch.float32)
    positions = torch.randn(batch_size, num_slices, device='cuda', dtype=torch.float32)
    
    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, softmax_scale, num_classes)
    
    assert output.shape == input.shape
    assert class_logits.shape == (batch_size, num_slices, num_classes)
    assert bounding_boxes.shape == (batch_size, num_slices, 4)
    
    # Check if output is normalized
    assert torch.allclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1)), atol=1e-6)
    
    # Check if class_logits and bounding_boxes are within reasonable ranges
    assert class_logits.min() >= 0 and class_logits.max() <= num_classes
    assert bounding_boxes.min() >= 0 and bounding_boxes.max() <= 1

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("num_slices", [1, 3])
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("num_classes", [10, 20])
@pytest.mark.parametrize("softmax_scale", [0.1, 1.0, 10.0])
def test_backward(batch_size, num_heads, num_slices, seq_len, head_dim, num_classes, softmax_scale):
    input = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', dtype=torch.double, requires_grad=True)
    positions = torch.randn(batch_size, num_slices, device='cuda', dtype=torch.double, requires_grad=True)
    
    assert gradcheck(TwoDSliceDetectionClassification.apply, (input, positions, softmax_scale, num_classes), eps=1e-6, atol=1e-4)

def test_empty_input():
    input = torch.empty(0, 4, 3, 16, 32, device='cuda', dtype=torch.float32)
    positions = torch.empty(0, 3, device='cuda', dtype=torch.float32)
    
    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, 1.0, 10)
    
    assert output.shape == (0, 4, 3, 16, 32)
    assert class_logits.shape == (0, 3, 10)
    assert bounding_boxes.shape == (0, 3, 4)

def test_large_input():
    input = torch.randn(8, 16, 20, 256, 64, device='cuda', dtype=torch.float32)
    positions = torch.randn(8, 20, device='cuda', dtype=torch.float32)
    
    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, 1.0, 100)
    
    assert output.shape == input.shape
    assert class_logits.shape == (8, 20, 100)
    assert bounding_boxes.shape == (8, 20, 4)

def test_numerical_stability():
    input = torch.randn(2, 4, 3, 16, 32, device='cuda', dtype=torch.float32)
    positions = torch.randn(2, 3, device='cuda', dtype=torch.float32)
    
    # Test with different softmax_scale values
    for scale in [1e-5, 1.0, 1e5]:
        output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input, positions, scale, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(class_logits).any()
        assert not torch.isinf(class_logits).any()
        assert not torch.isnan(bounding_boxes).any()
        assert not torch.isinf(bounding_boxes).any()

def test_end_to_end():
    batch_size, num_heads, num_slices, seq_len, head_dim = 2, 4, 3, 16, 64
    num_classes = 10
    softmax_scale = 0.1

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', requires_grad=True)
    positions = torch.randn(batch_size, num_slices, device='cuda', requires_grad=True)

    # Run forward and backward passes
    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)
    loss = output.sum() + class_logits.sum() + bounding_boxes.sum()
    loss.backward()

    # Check that gradients are computed
    assert input_tensor.grad is not None
    assert positions.grad is not None
    assert not torch.isnan(input_tensor.grad).any()
    assert not torch.isnan(positions.grad).any()

def test_gradient_flow():
    batch_size, num_heads, num_slices, seq_len, head_dim = 2, 4, 3, 16, 64
    num_classes = 10
    softmax_scale = 0.1

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', requires_grad=True)
    positions = torch.randn(batch_size, num_slices, device='cuda', requires_grad=True)

    initial_input = input_tensor.clone()
    initial_positions = positions.clone()

    optimizer = torch.optim.Adam([input_tensor, positions], lr=0.01)

    for _ in range(5):  # Run a few optimization steps
        optimizer.zero_grad()
        output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)
        loss = output.sum() + class_logits.sum() + bounding_boxes.sum()
        loss.backward()
        optimizer.step()

    # Check that the parameters have been updated
    assert not torch.allclose(input_tensor, initial_input)
    assert not torch.allclose(positions, initial_positions)



def test_performance():
    batch_size, num_heads, num_slices, seq_len, head_dim = 32, 8, 10, 128, 64
    num_classes = 100
    softmax_scale = 1.0

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda')
    positions = torch.randn(batch_size, num_slices, device='cuda')

    # Warm-up
    for _ in range(10):
        TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)

    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Average execution time: {avg_time * 1000:.2f} ms")

def test_input_consistency():
    batch_size, num_heads, num_slices, seq_len, head_dim = 2, 4, 3, 16, 64
    num_classes = 10
    softmax_scale = 0.1

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda')
    positions = torch.randn(batch_size, num_slices, device='cuda')

    output1, class_logits1, bounding_boxes1 = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)
    output2, class_logits2, bounding_boxes2 = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)

    assert torch.allclose(output1, output2)
    assert torch.allclose(class_logits1, class_logits2)
    assert torch.allclose(bounding_boxes1, bounding_boxes2)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_support(dtype):
    batch_size, num_heads, num_slices, seq_len, head_dim = 2, 4, 3, 16, 64
    num_classes = 10
    softmax_scale = 0.1

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda', dtype=dtype)
    positions = torch.randn(batch_size, num_slices, device='cuda', dtype=dtype)

    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)

    assert output.dtype == dtype
    assert class_logits.dtype == dtype
    assert bounding_boxes.dtype == dtype

def test_gpu_memory_usage():
    batch_size, num_heads, num_slices, seq_len, head_dim = 32, 8, 10, 128, 64
    num_classes = 100
    softmax_scale = 1.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    input_tensor = torch.randn(batch_size, num_heads, num_slices, seq_len, head_dim, device='cuda')
    positions = torch.randn(batch_size, num_slices, device='cuda')

    initial_memory = torch.cuda.memory_allocated()

    output, class_logits, bounding_boxes = TwoDSliceDetectionClassification.apply(input_tensor, positions, softmax_scale, num_classes)

    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory usage: {(final_memory - initial_memory) / 1024**2:.2f} MB")
    print(f"Peak memory usage: {peak_memory / 1024**2:.2f} MB")

    # Ensure memory usage is reasonable (adjust threshold as needed)
    assert peak_memory - initial_memory < 1024**3  # Less than 1 GB increase

def test_error_handling():
    with pytest.raises(RuntimeError):
        # Mismatched batch sizes
        input_tensor = torch.randn(2, 4, 3, 16, 64, device='cuda')
        positions = torch.randn(3, 3, device='cuda')
        TwoDSliceDetectionClassification.apply(input_tensor, positions, 1.0, 10)

    with pytest.raises(RuntimeError):
        # Invalid number of classes
        input_tensor = torch.randn(2, 4, 3, 16, 64, device='cuda')
        positions = torch.randn(2, 3, device='cuda')
        TwoDSliceDetectionClassification.apply(input_tensor, positions, 1.0, 0)


if __name__ == "__main__":
    pytest.main([__file__])
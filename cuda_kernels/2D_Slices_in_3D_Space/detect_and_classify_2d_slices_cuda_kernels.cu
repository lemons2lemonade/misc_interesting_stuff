#include <torch/extension.h>
#include <vector>

// Declare the CUDA functions (implemented in the .cu file)
std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_forward(
    torch::Tensor& input,
    torch::Tensor& positions,
    float softmax_scale,
    int num_classes);

std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_backward(
    torch::Tensor& input,
    torch::Tensor& positions,
    torch::Tensor& output,
    torch::Tensor& class_logits,
    torch::Tensor& bounding_boxes,
    torch::Tensor& doutput,
    torch::Tensor& dclass_logits,
    torch::Tensor& dbounding_boxes,
    float softmax_scale);

// C++ interface for forward pass
std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_forward(
    torch::Tensor& input,
    torch::Tensor& positions,
    float softmax_scale,
    int num_classes) {
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");
    TORCH_CHECK(positions.dim() == 2, "positions must be a 2D tensor");

    // Call the CUDA function
    return detect_and_classify_two_dimensional_slices_cuda_forward(input, positions, softmax_scale, num_classes);
}

// C++ interface for backward pass
std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_backward(
    torch::Tensor& input,
    torch::Tensor& positions,
    torch::Tensor& output,
    torch::Tensor& class_logits,
    torch::Tensor& bounding_boxes,
    torch::Tensor& doutput,
    torch::Tensor& dclass_logits,
    torch::Tensor& dbounding_boxes,
    float softmax_scale) {
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(class_logits.is_cuda(), "class_logits must be a CUDA tensor");
    TORCH_CHECK(bounding_boxes.is_cuda(), "bounding_boxes must be a CUDA tensor");
    TORCH_CHECK(doutput.is_cuda(), "doutput must be a CUDA tensor");
    TORCH_CHECK(dclass_logits.is_cuda(), "dclass_logits must be a CUDA tensor");
    TORCH_CHECK(dbounding_boxes.is_cuda(), "dbounding_boxes must be a CUDA tensor");

    // Call the CUDA function
    return detect_and_classify_two_dimensional_slices_cuda_backward(
        input, positions, output, class_logits, bounding_boxes, 
        doutput, dclass_logits, dbounding_boxes, softmax_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &detect_and_classify_two_dimensional_slices_forward, "2D Slice Detection and Classification forward (CUDA)");
    m.def("backward", &detect_and_classify_two_dimensional_slices_backward, "2D Slice Detection and Classification backward (CUDA)");
}
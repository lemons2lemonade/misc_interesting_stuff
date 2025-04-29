#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>

using namespace torch::indexing;


// // Utility function for positional encoding
// __device__ float positional_encoding(float position, int dim, int max_dim) {
//     return dim % 2 == 0 ? sinf(position / powf(10000, dim / float(max_dim))) 
//                         : cosf(position / powf(10000, (dim - 1) / float(max_dim)));
// }

// template <typename scalar_t>
// __global__ void detect_and_classify_two_dimensional_slices_forward_kernel(
//     const scalar_t* input,
//     const float* positions,
//     scalar_t* output,
//     scalar_t* class_logits,
//     scalar_t* bounding_boxes,
//     scalar_t* l,
//     scalar_t* m,
//     int batch_size,
//     int num_heads,
//     int num_slices,
//     int seq_len,
//     int head_dim,
//     int num_classes,
//     float softmax_scale) {
    
//     extern __shared__ char shared_memory[];
//     scalar_t* s_input = (scalar_t*)shared_memory;
//     scalar_t* s_attention_scores = (scalar_t*)&s_input[seq_len * head_dim];
//     float* s_pos = (float*)&s_attention_scores[seq_len];
//     __shared__ scalar_t s_max_val;
//     __shared__ scalar_t s_sum_exp;

//     int tidx = threadIdx.x;
//     int bidx = blockIdx.x;
//     int batch_idx = bidx / (num_heads * num_slices);
//     int head_idx = (bidx / num_slices) % num_heads;
//     int slice_idx = bidx % num_slices;
    
//     // Load input and position into shared memory
//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         for (int j = 0; j < head_dim; ++j) {
//             s_input[i * head_dim + j] = input[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
//         }
//     }
//     if (tidx == 0) {
//         s_pos[0] = positions[batch_idx * num_slices + slice_idx];
//         s_max_val = -INFINITY;
//         s_sum_exp = 0;
//     }
//     __syncthreads();

//     // First pass: Compute attention scores and find max
//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         scalar_t attention_score = 0;
//         for (int j = 0; j < head_dim; ++j) {
//             attention_score += s_input[i * head_dim + j] * positional_encoding(s_pos[0], j, head_dim);
//         }
//         attention_score *= softmax_scale;
//         s_attention_scores[i] = attention_score;
//         atomicMax(&s_max_val, attention_score);
//     }
//     __syncthreads();

//     // Second pass: Compute sum of exp and normalized attention
//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         scalar_t exp_attention = __expf(s_attention_scores[i] - s_max_val);
//         atomicAdd(&s_sum_exp, exp_attention);
        
//         for (int j = 0; j < head_dim; ++j) {
//             atomicAdd(&output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
//                       exp_attention * s_input[i * head_dim + j]);
//         }
//     }
//     __syncthreads();

//     // Normalize output and compute class logits and bounding boxes
//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         for (int j = 0; j < head_dim; ++j) {
//             scalar_t normalized_output = output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j] / s_sum_exp;
//             output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j] = normalized_output;
            
//             // Compute class logits
//             for (int c = 0; c < num_classes; ++c) {
//                 atomicAdd(&class_logits[(batch_idx * num_slices + slice_idx) * num_classes + c],
//                           normalized_output * (c + 1));  // Simple class computation, adjust as needed
//             }
            
//             // Compute bounding boxes (assuming 4 values for 2D bounding box)
//             for (int b = 0; b < 4; ++b) {
//                 atomicAdd(&bounding_boxes[(batch_idx * num_slices + slice_idx) * 4 + b],
//                           normalized_output * (b + 1));  // Simple box computation, adjust as needed
//             }
//         }
//     }

//     // Store l and m for backward pass
//     if (tidx == 0) {
//         l[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)] = s_sum_exp;
//         m[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)] = s_max_val;
//     }
// }

// std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_forward(
//     torch::Tensor& input,
//     torch::Tensor& positions,
//     float softmax_scale,
//     int num_classes) {
    
//     auto batch_size = input.size(0);
//     auto num_heads = input.size(1);
//     auto num_slices = input.size(2);
//     auto seq_len = input.size(3);
//     auto head_dim = input.size(4);

//     auto output = torch::zeros_like(input);
//     auto class_logits = torch::zeros({batch_size, num_slices, num_classes}, input.options());
//     auto bounding_boxes = torch::zeros({batch_size, num_slices, 4}, input.options());
//     auto l = torch::zeros({batch_size, num_heads, num_slices}, input.options());
//     auto m = torch::zeros({batch_size, num_heads, num_slices}, input.options());

//     const int threads = 256;
//     const int blocks = batch_size * num_heads * num_slices;
    
//     // Modified shared memory size calculation
//     const int shared_memory_size = (seq_len * (head_dim + 1) * sizeof(float)) + sizeof(float);

//     AT_DISPATCH_FLOATING_TYPES(input.type(), "detect_and_classify_forward_cuda", ([&] {
//         detect_and_classify_two_dimensional_slices_forward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
//             input.data_ptr<scalar_t>(),
//             positions.data_ptr<float>(),
//             output.data_ptr<scalar_t>(),
//             class_logits.data_ptr<scalar_t>(),
//             bounding_boxes.data_ptr<scalar_t>(),
//             l.data_ptr<scalar_t>(),
//             m.data_ptr<scalar_t>(),
//             batch_size,
//             num_heads,
//             num_slices,
//             seq_len,
//             head_dim,
//             num_classes,
//             softmax_scale
//         );
//     }));

//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(error));
//     }

//     return {output, class_logits, bounding_boxes, l, m};
// }

// template <typename scalar_t>
// __global__ void detect_and_classify_two_dimensional_slices_backward_kernel(
//     const scalar_t* input,
//     const float* positions,
//     const scalar_t* output,
//     const scalar_t* class_logits,
//     const scalar_t* bounding_boxes,
//     const scalar_t* l,
//     const scalar_t* m,
//     const scalar_t* doutput,
//     const scalar_t* dclass_logits,
//     const scalar_t* dbounding_boxes,
//     scalar_t* dinput,
//     float* dpositions,
//     int batch_size,
//     int num_heads,
//     int num_slices,
//     int seq_len,
//     int head_dim,
//     int num_classes,
//     float softmax_scale) {
    
//     extern __shared__ char shared_memory[];
//     scalar_t* s_input = (scalar_t*)shared_memory;
//     scalar_t* s_output = (scalar_t*)&s_input[seq_len * head_dim];
//     scalar_t* s_doutput = (scalar_t*)&s_output[seq_len * head_dim];
//     float* s_pos = (float*)&s_doutput[seq_len * head_dim];
//     __shared__ scalar_t s_max_val;
//     __shared__ scalar_t s_sum_exp;

//     int tidx = threadIdx.x;
//     int bidx = blockIdx.x;
//     int batch_idx = bidx / (num_heads * num_slices);
//     int head_idx = (bidx / num_slices) % num_heads;
//     int slice_idx = bidx % num_slices;
    
//     // Load data into shared memory
//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         for (int j = 0; j < head_dim; ++j) {
//             s_input[i * head_dim + j] = input[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
//             s_output[i * head_dim + j] = output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
//             s_doutput[i * head_dim + j] = doutput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
//         }
//     }
//     if (tidx == 0) {
//         s_pos[0] = positions[batch_idx * num_slices + slice_idx];
//         s_max_val = m[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)];
//         s_sum_exp = l[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)];
//     }
//     __syncthreads();

//     scalar_t thread_dinput[64];  // Assume max head_dim is 64
//     scalar_t thread_dpos = 0;

//     for (int i = tidx; i < seq_len; i += blockDim.x) {
//         scalar_t attention_score = 0;
//         for (int j = 0; j < head_dim; ++j) {
//             attention_score += s_input[i * head_dim + j] * positional_encoding(s_pos[0], j, head_dim);
//         }
//         attention_score *= softmax_scale;
        
//         scalar_t attention_weight = __expf(attention_score - s_max_val) / s_sum_exp;
        
//         for (int j = 0; j < head_dim; ++j) {
//             scalar_t pos_encoding = positional_encoding(s_pos[0], j, head_dim);
//             scalar_t grad_term = s_doutput[i * head_dim + j] * (s_output[i * head_dim + j] - attention_weight * s_input[i * head_dim + j]);
            
//             thread_dinput[j] += softmax_scale * pos_encoding * grad_term;
//             thread_dpos += softmax_scale * attention_weight * s_input[i * head_dim + j] * grad_term * 
//                            (j % 2 == 0 ? -sinf(s_pos[0]) : cosf(s_pos[0])) / powf(10000, j / float(head_dim));
//         }
//     }

//     // Accumulate gradients
//     for (int j = 0; j < head_dim; ++j) {
//         atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + tidx) * head_dim + j],
//                   thread_dinput[j]);
//     }
//     atomicAdd(&dpositions[batch_idx * num_slices + slice_idx], thread_dpos);

//     // Gradients for class logits and bounding boxes
//     if (tidx < num_classes) {
//         for (int i = 0; i < seq_len; ++i) {
//             for (int j = 0; j < head_dim; ++j) {
//                 atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
//                           dclass_logits[(batch_idx * num_slices + slice_idx) * num_classes + tidx] * (tidx + 1));
//             }
//         }
//     }
//     if (tidx < 4) {  // 4 for 2D bounding box
//         for (int i = 0; i < seq_len; ++i) {
//             for (int j = 0; j < head_dim; ++j) {
//                 atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
//                           dbounding_boxes[(batch_idx * num_slices + slice_idx) * 4 + tidx] * (tidx + 1));
//             }
//         }
//     }
// }
// std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_backward(
//     torch::Tensor& input,
//     torch::Tensor& positions,
//     torch::Tensor& output,
//     torch::Tensor& class_logits,
//     torch::Tensor& bounding_boxes,
//     torch::Tensor& l,
//     torch::Tensor& m,
//     torch::Tensor& doutput,
//     torch::Tensor& dclass_logits,
//     torch::Tensor& dbounding_boxes,
//     float softmax_scale) {
    
//     auto batch_size = input.size(0);
//     auto num_heads = input.size(1);
//     auto num_slices = input.size(2);
//     auto seq_len = input.size(3);
//     auto head_dim = input.size(4);
//     auto num_classes = class_logits.size(2);

//     auto dinput = torch::zeros_like(input);
//     auto dpositions = torch::zeros_like(positions);

//     const int threads = 256;
//     const int blocks = batch_size * num_heads * num_slices;
//     const int shared_memory_size = (3 * seq_len * head_dim * sizeof(float)) + sizeof(float);

//     AT_DISPATCH_FLOATING_TYPES(input.type(), "detect_and_classify_backward_cuda", ([&] {
//         detect_and_classify_two_dimensional_slices_backward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
//             input.data_ptr<scalar_t>(),
//             positions.data_ptr<float>(),
//             output.data_ptr<scalar_t>(),
//             class_logits.data_ptr<scalar_t>(),
//             bounding_boxes.data_ptr<scalar_t>(),
//             l.data_ptr<scalar_t>(),
//             m.data_ptr<scalar_t>(),
//             doutput.data_ptr<scalar_t>(),
//             dclass_logits.data_ptr<scalar_t>(),
//             dbounding_boxes.data_ptr<scalar_t>(),
//             dinput.data_ptr<scalar_t>(),
//             dpositions.data_ptr<float>(),
//             batch_size,
//             num_heads,
//             num_slices,
//             seq_len,
//             head_dim,
//             num_classes,
//             softmax_scale
//         );
//     }));

//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(error));
//     }

//     return {dinput, dpositions};
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &detect_and_classify_two_dimensional_slices_cuda_forward, 
//           "2D Slice Detection and Classification forward pass (CUDA)",
//           py::arg("input"), py::arg("positions"), py::arg("softmax_scale"), py::arg("num_classes"));
//     m.def("backward", &detect_and_classify_two_dimensional_slices_cuda_backward, 
//           "2D Slice Detection and Classification backward pass (CUDA)",
//           py::arg("input"), py::arg("positions"), py::arg("output"), py::arg("class_logits"), 
//           py::arg("bounding_boxes"), py::arg("l"), py::arg("m"), py::arg("doutput"), py::arg("dclass_logits"), 
//           py::arg("dbounding_boxes"), py::arg("softmax_scale"));
// }

// Utility function for positional encoding
__device__ float positional_encoding(float position, int dim, int max_dim) {
    return dim % 2 == 0 ? sinf(position / powf(10000, dim / float(max_dim))) 
                        : cosf(position / powf(10000, (dim - 1) / float(max_dim)));
}

template <typename scalar_t>
__global__ void detect_and_classify_two_dimensional_slices_forward_kernel(
    const scalar_t* input,
    const float* positions,
    scalar_t* output,
    scalar_t* class_logits,
    scalar_t* bounding_boxes,
    scalar_t* l,
    scalar_t* m,
    int batch_size,
    int num_heads,
    int num_slices,
    int seq_len,
    int head_dim,
    int num_classes,
    float softmax_scale) {
    
    extern __shared__ char shared_memory[];
    scalar_t* s_input = (scalar_t*)shared_memory;
    scalar_t* s_attention_scores = (scalar_t*)&s_input[seq_len * head_dim];
    float* s_pos = (float*)&s_attention_scores[seq_len];
    __shared__ scalar_t s_max_val;
    __shared__ scalar_t s_sum_exp;

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int batch_idx = bidx / (num_heads * num_slices);
    int head_idx = (bidx / num_slices) % num_heads;
    int slice_idx = bidx % num_slices;
    
    // Load input and position into shared memory
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        for (int j = 0; j < head_dim; ++j) {
            s_input[i * head_dim + j] = input[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
        }
    }
    if (tidx == 0) {
        s_pos[0] = positions[batch_idx * num_slices + slice_idx];
        s_max_val = -INFINITY;
        s_sum_exp = 0;
    }
    __syncthreads();

    // First pass: Compute attention scores and find max
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        scalar_t attention_score = 0;
        for (int j = 0; j < head_dim; ++j) {
            attention_score += s_input[i * head_dim + j] * positional_encoding(s_pos[0], j, head_dim);
        }
        attention_score *= softmax_scale;
        s_attention_scores[i] = attention_score;
        atomicMax(&s_max_val, attention_score);
    }
    __syncthreads();

    // Second pass: Compute sum of exp and normalized attention
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        scalar_t exp_attention = __expf(s_attention_scores[i] - s_max_val);
        atomicAdd(&s_sum_exp, exp_attention);
        
        for (int j = 0; j < head_dim; ++j) {
            atomicAdd(&output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
                      exp_attention * s_input[i * head_dim + j]);
        }
    }
    __syncthreads();

    // Normalize output and compute class logits and bounding boxes
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        for (int j = 0; j < head_dim; ++j) {
            scalar_t normalized_output = output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j] / s_sum_exp;
            output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j] = normalized_output;
            
            // Compute class logits
            for (int c = 0; c < num_classes; ++c) {
                atomicAdd(&class_logits[(batch_idx * num_slices + slice_idx) * num_classes + c],
                          normalized_output * (c + 1));  // Simple class computation, adjust as needed
            }
            
            // Compute bounding boxes (assuming 4 values for 2D bounding box)
            for (int b = 0; b < 4; ++b) {
                atomicAdd(&bounding_boxes[(batch_idx * num_slices + slice_idx) * 4 + b],
                          normalized_output * (b + 1));  // Simple box computation, adjust as needed
            }
        }
    }

    // Store l and m for backward pass
    if (tidx == 0) {
        l[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)] = s_sum_exp;
        m[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)] = s_max_val;
    }
}

template <typename scalar_t>
__global__ void detect_and_classify_two_dimensional_slices_backward_kernel(
    const scalar_t* input,
    const float* positions,
    const scalar_t* output,
    const scalar_t* class_logits,
    const scalar_t* bounding_boxes,
    const scalar_t* l,
    const scalar_t* m,
    const scalar_t* doutput,
    const scalar_t* dclass_logits,
    const scalar_t* dbounding_boxes,
    scalar_t* dinput,
    float* dpositions,
    int batch_size,
    int num_heads,
    int num_slices,
    int seq_len,
    int head_dim,
    int num_classes,
    float softmax_scale) {
    
    extern __shared__ char shared_memory[];
    scalar_t* s_input = (scalar_t*)shared_memory;
    scalar_t* s_output = (scalar_t*)&s_input[seq_len * head_dim];
    scalar_t* s_doutput = (scalar_t*)&s_output[seq_len * head_dim];
    float* s_pos = (float*)&s_doutput[seq_len * head_dim];
    __shared__ scalar_t s_max_val;
    __shared__ scalar_t s_sum_exp;

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int batch_idx = bidx / (num_heads * num_slices);
    int head_idx = (bidx / num_slices) % num_heads;
    int slice_idx = bidx % num_slices;
    
    // Load data into shared memory
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        for (int j = 0; j < head_dim; ++j) {
            s_input[i * head_dim + j] = input[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
            s_output[i * head_dim + j] = output[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
            s_doutput[i * head_dim + j] = doutput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j];
        }
    }
    if (tidx == 0) {
        s_pos[0] = positions[batch_idx * num_slices + slice_idx];
        s_max_val = m[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)];
        s_sum_exp = l[((batch_idx * num_heads + head_idx) * num_slices + slice_idx)];
    }
    __syncthreads();

    scalar_t thread_dinput[64];  // Assume max head_dim is 64
    scalar_t thread_dpos = 0;

    for (int i = tidx; i < seq_len; i += blockDim.x) {
        scalar_t attention_score = 0;
        for (int j = 0; j < head_dim; ++j) {
            attention_score += s_input[i * head_dim + j] * positional_encoding(s_pos[0], j, head_dim);
        }
        attention_score *= softmax_scale;
        
        scalar_t attention_weight = __expf(attention_score - s_max_val) / s_sum_exp;
        
        for (int j = 0; j < head_dim; ++j) {
            scalar_t pos_encoding = positional_encoding(s_pos[0], j, head_dim);
            scalar_t grad_term = s_doutput[i * head_dim + j] * (s_output[i * head_dim + j] - attention_weight * s_input[i * head_dim + j]);
            
            thread_dinput[j] += softmax_scale * pos_encoding * grad_term;
            thread_dpos += softmax_scale * attention_weight * s_input[i * head_dim + j] * grad_term * 
                           (j % 2 == 0 ? -sinf(s_pos[0]) : cosf(s_pos[0])) / powf(10000, j / float(head_dim));
        }
    }

    // Accumulate gradients
    for (int j = 0; j < head_dim; ++j) {
        atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + tidx) * head_dim + j],
                  thread_dinput[j]);
    }
    atomicAdd(&dpositions[batch_idx * num_slices + slice_idx], thread_dpos);

    // Gradients for class logits and bounding boxes
    if (tidx < num_classes) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
                          dclass_logits[(batch_idx * num_slices + slice_idx) * num_classes + tidx] * (tidx + 1));
            }
        }
    }
    if (tidx < 4) {  // 4 for 2D bounding box
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                atomicAdd(&dinput[(((batch_idx * num_heads + head_idx) * num_slices + slice_idx) * seq_len + i) * head_dim + j],
                          dbounding_boxes[(batch_idx * num_slices + slice_idx) * 4 + tidx] * (tidx + 1));
            }
        }
    }
}

std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_forward(
    torch::Tensor& input,
    torch::Tensor& positions,
    float softmax_scale,
    int num_classes) {
    
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto num_slices = input.size(2);
    auto seq_len = input.size(3);
    auto head_dim = input.size(4);

    auto output = torch::zeros_like(input);
    auto class_logits = torch::zeros({batch_size, num_slices, num_classes}, input.options());
    auto bounding_boxes = torch::zeros({batch_size, num_slices, 4}, input.options());
    auto l = torch::zeros({batch_size, num_heads, num_slices}, input.options());
    auto m = torch::zeros({batch_size, num_heads, num_slices}, input.options());

    const int threads = 256;
    const int blocks = batch_size * num_heads * num_slices;
    
    const int shared_memory_size = (seq_len * (head_dim + 1) * sizeof(float)) + sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "detect_and_classify_forward_cuda", ([&] {
        detect_and_classify_two_dimensional_slices_forward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            positions.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            class_logits.data_ptr<scalar_t>(),
            bounding_boxes.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            num_slices,
            seq_len,
            head_dim,
            num_classes,
            softmax_scale
        );
    }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    return {output, class_logits, bounding_boxes, l, m};
}

std::vector<torch::Tensor> detect_and_classify_two_dimensional_slices_cuda_backward(
    torch::Tensor& input,
    torch::Tensor& positions,
    torch::Tensor& output,
    torch::Tensor& class_logits,
    torch::Tensor& bounding_boxes,
    torch::Tensor& l,
    torch::Tensor& m,
    torch::Tensor& doutput,
    torch::Tensor& dclass_logits,
    torch::Tensor& dbounding_boxes,
    float softmax_scale) {
    
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto num_slices = input.size(2);
    auto seq_len = input.size(3);
    auto head_dim = input.size(4);
    auto num_classes = class_logits.size(2);

    auto dinput = torch::zeros_like(input);
    auto dpositions = torch::zeros_like(positions);

    const int threads = 256;
    const int blocks = batch_size * num_heads * num_slices;
    const int shared_memory_size = (3 * seq_len * head_dim * sizeof(float)) + sizeof(float);

AT_DISPATCH_FLOATING_TYPES(input.type(), "detect_and_classify_backward_cuda", ([&] {
        detect_and_classify_two_dimensional_slices_backward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            positions.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            class_logits.data_ptr<scalar_t>(),
            bounding_boxes.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            doutput.data_ptr<scalar_t>(),
            dclass_logits.data_ptr<scalar_t>(),
            dbounding_boxes.data_ptr<scalar_t>(),
            dinput.data_ptr<scalar_t>(),
            dpositions.data_ptr<float>(),
            batch_size,
            num_heads,
            num_slices,
            seq_len,
            head_dim,
            num_classes,
            softmax_scale
        );
    }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    return {dinput, dpositions};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &detect_and_classify_two_dimensional_slices_cuda_forward, 
          "2D Slice Detection and Classification forward pass (CUDA)",
          py::arg("input"), py::arg("positions"), py::arg("softmax_scale"), py::arg("num_classes"));
    m.def("backward", &detect_and_classify_two_dimensional_slices_cuda_backward, 
          "2D Slice Detection and Classification backward pass (CUDA)",
          py::arg("input"), py::arg("positions"), py::arg("output"), py::arg("class_logits"), 
          py::arg("bounding_boxes"), py::arg("l"), py::arg("m"), py::arg("doutput"), py::arg("dclass_logits"), 
          py::arg("dbounding_boxes"), py::arg("softmax_scale"));
}



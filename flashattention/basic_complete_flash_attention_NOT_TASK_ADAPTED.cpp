#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA kernel declarations
template <typename scalar_t>
__global__ void flash_attention_forward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* o,
    scalar_t* l,
    scalar_t* m,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    float softmax_scale,
    bool causal);

template <typename scalar_t>
__global__ void flash_attention_backward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* o,
    const scalar_t* l,
    const scalar_t* m,
    const scalar_t* dout,
    scalar_t* dq,
    scalar_t* dk,
    scalar_t* dv,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    float softmax_scale,
    bool causal);

// C++ interface
std::vector<torch::Tensor> flash_attention_forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    float softmax_scale,
    bool causal) {
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len_q = q.size(2);
    auto head_dim_q = q.size(3);
    auto seq_len_kv = k.size(2);
    auto head_dim_k = k.size(3);
    auto head_dim_v = v.size(3);

    auto o = torch::zeros({batch_size, num_heads, seq_len_q, head_dim_v}, q.options());
    auto l = torch::zeros({batch_size, num_heads, seq_len_q}, q.options());
    auto m = torch::full({batch_size, num_heads, seq_len_q}, -std::numeric_limits<float>::infinity(), q.options());

    const int threads = 256;
    const int blocks = (batch_size * num_heads * seq_len_q + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_forward", ([&] {
        flash_attention_forward_kernel<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_kv,
            head_dim_q,
            head_dim_k,
            head_dim_v,
            softmax_scale,
            causal
        );
    }));

    return {o, l, m};
}

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& o,
    torch::Tensor& l,
    torch::Tensor& m,
    torch::Tensor& dout,
    float softmax_scale,
    bool causal) {
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len_q = q.size(2);
    auto head_dim_q = q.size(3);
    auto seq_len_kv = k.size(2);
    auto head_dim_k = k.size(3);
    auto head_dim_v = v.size(3);

    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const int threads = 256;
    const int blocks = (batch_size * num_heads * seq_len_q + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_backward", ([&] {
        flash_attention_backward_kernel<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            dout.data_ptr<scalar_t>(),
            dq.data_ptr<scalar_t>(),
            dk.data_ptr<scalar_t>(),
            dv.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_kv,
            head_dim_q,
            head_dim_k,
            head_dim_v,
            softmax_scale,
            causal
        );
    }));

    return {dq, dk, dv};
}

// CUDA kernel implementations
template <typename scalar_t>
__global__ void flash_attention_forward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* o,
    scalar_t* l,
    scalar_t* m,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    float softmax_scale,
    bool causal) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int lane = tidx % 32;
    int warp = tidx / 32;

    // Constants for tiling
    constexpr int TILE_SIZE = 64;
    
    // Shared memory
    __shared__ scalar_t s_k[TILE_SIZE][64];
    __shared__ scalar_t s_v[TILE_SIZE][64];
    
    int batch_idx = bidx / (num_heads * seq_len_q);
    int head_idx = (bidx / seq_len_q) % num_heads;
    int seq_idx = bidx % seq_len_q;
    
    scalar_t thread_max = -INFINITY;
    scalar_t thread_sum = 0;

    // Main loop over tiles
    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, seq_len_kv);
        
        // Load K and V into shared memory
        for (int i = tidx; i < (tile_end - tile_start) * max(head_dim_k, head_dim_v); i += blockDim.x) {
            int col = i % max(head_dim_k, head_dim_v);
            int row = i / max(head_dim_k, head_dim_v);
            if (col < head_dim_k) {
                s_k[row][col] = k[batch_idx * num_heads * seq_len_kv * head_dim_k + 
                                  head_idx * seq_len_kv * head_dim_k + 
                                  (tile_start + row) * head_dim_k + col];
            }
            if (col < head_dim_v) {
                s_v[row][col] = v[batch_idx * num_heads * seq_len_kv * head_dim_v + 
                                  head_idx * seq_len_kv * head_dim_v + 
                                  (tile_start + row) * head_dim_v + col];
            }
        }
        __syncthreads();
        
        // Compute attention scores and update output
        for (int i = 0; i < tile_end - tile_start; ++i) {
            if (causal && tile_start + i > seq_idx) break;
            
            scalar_t qk_sum = 0;
            #pragma unroll
            for (int j = 0; j < min(head_dim_q, head_dim_k); ++j) {
                qk_sum += q[batch_idx * num_heads * seq_len_q * head_dim_q + 
                            head_idx * seq_len_q * head_dim_q + 
                            seq_idx * head_dim_q + j] * s_k[i][j];
            }
            qk_sum *= softmax_scale;
            
            scalar_t exp_sum = __expf(qk_sum - thread_max);
            scalar_t new_max = max(thread_max, qk_sum);
            scalar_t exp_rescale = __expf(thread_max - new_max);
            
            thread_sum = thread_sum * exp_rescale + exp_sum;
            thread_max = new_max;
            
            scalar_t attention_weight = exp_sum / thread_sum;
            
            #pragma unroll
            for (int j = 0; j < head_dim_v; ++j) {
                atomicAdd(&o[batch_idx * num_heads * seq_len_q * head_dim_v + 
                            head_idx * seq_len_q * head_dim_v + 
                            seq_idx * head_dim_v + j], 
                          attention_weight * s_v[i][j]);
            }
        }
        __syncthreads();
    }
    
    // Store final l and m values
    l[batch_idx * num_heads * seq_len_q + head_idx * seq_len_q + seq_idx] = thread_sum;
    m[batch_idx * num_heads * seq_len_q + head_idx * seq_len_q + seq_idx] = thread_max;
}

template <typename scalar_t>
__global__ void flash_attention_backward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* o,
    const scalar_t* l,
    const scalar_t* m,
    const scalar_t* dout,
    scalar_t* dq,
    scalar_t* dk,
    scalar_t* dv,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    float softmax_scale,
    bool causal) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int lane = tidx % 32;
    int warp = tidx / 32;

    // Constants for tiling
    constexpr int TILE_SIZE = 64;
    
    // Shared memory
    __shared__ scalar_t s_k[TILE_SIZE][64];
    __shared__ scalar_t s_v[TILE_SIZE][64];
    __shared__ scalar_t s_do[TILE_SIZE][64];
    
    int batch_idx = bidx / (num_heads * seq_len_q);
    int head_idx = (bidx / seq_len_q) % num_heads;
    int seq_idx = bidx % seq_len_q;
    
    scalar_t thread_sum = 0;

    // Main loop over tiles
    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, seq_len_kv);
        
        // Load K, V, and dO into shared memory
        for (int i = tidx; i < (tile_end - tile_start) * max(head_dim_k, head_dim_v); i += blockDim.x) {
            int col = i % max(head_dim_k, head_dim_v);
            int row = i / max(head_dim_k, head_dim_v);
            if (col < head_dim_k) {
                s_k[row][col] = k[batch_idx * num_heads * seq_len_kv * head_dim_k + 
                                  head_idx * seq_len_kv * head_dim_k + 
                                  (tile_start + row) * head_dim_k + col];
            }
            if (col < head_dim_v) {
                s_v[row][col] = v[batch_idx * num_heads * seq_len_kv * head_dim_v + 
                                  head_idx * seq_len_kv * head_dim_v + 
                                  (tile_start + row) * head_dim_v + col];
            }
        }
        for (int i = tidx; i < head_dim_v; i += blockDim.x) {
            s_do[0][i] = dout[batch_idx * num_heads * seq_len_q * head_dim_v + 
                              head_idx * seq_len_q * head_dim_v + 
                              seq_idx * head_dim_v + i];
        }
        __syncthreads();
        
        // Compute gradients
        for (int i = 0; i < tile_end - tile_start; ++i) {
            if (causal && tile_start + i > seq_idx) break;
            
            scalar_t qk_sum = 0;
            #pragma unroll
            for (int j = 0; j < min(head_dim_q, head_dim_k); ++j) {
                qk_sum += q[batch_idx * num_heads * seq_len_q * head_dim_q + 
                            head_idx * seq_len_q * head_dim_q + 
                            seq_idx * head_dim_q + j] * s_k[i][j];
            }
            qk_sum *= softmax_scale;
            
            scalar_t attention_weight = __expf(qk_sum - m[batch_idx * num_heads * seq_len_q + head_idx * seq_len_q + seq_idx]) / 
                                        l[batch_idx * num_heads * seq_len_q + head_idx * seq_len_q + seq_idx];
            
            // Gradient w.r.t. q
            #pragma unroll
            for (int j = 0; j < head_dim_q; ++j) {
                scalar_t dq_sum = 0;
                for (int k = 0; k < head_dim_v; ++k) {
                    dq_sum += s_do[0][k] * (s_v[i][k] - o[batch_idx * num_heads * seq_len_q * head_dim_v + 
                                                          head_idx * seq_len_q * head_dim_v + 
                                                          seq_idx * head_dim_v + k]);
                }
                atomicAdd(&dq[batch_idx * num_heads * seq_len_q * head_dim_q + 
                              head_idx * seq_len_q * head_dim_q + 
                              seq_idx * head_dim_q + j],
                          softmax_scale * attention_weight * s_k[i][j] * dq_sum);
            }
            
            // Gradient w.r.t. k
            #pragma unroll
            for (int j = 0; j < head_dim_k; ++j) {
                scalar_t dk_sum = 0;
                for (int k = 0; k < head_dim_v; ++k) {
                    dk_sum += s_do[0][k] * (s_v[i][k] - o[batch_idx * num_heads * seq_len_q * head_dim_v + 
                                                          head_idx * seq_len_q * head_dim_v + 
                                                          seq_idx * head_dim_v + k]);
                }
                atomicAdd(&dk[batch_idx * num_heads * seq_len_kv * head_dim_k + 
                              head_idx * seq_len_kv * head_dim_k + 
                              (tile_start + i) * head_dim_k + j],
                          softmax_scale * attention_weight * 
                          q[batch_idx * num_heads * seq_len_q * head_dim_q + 
                            head_idx * seq_len_q * head_dim_q + 
                            seq_idx * head_dim_q + j] * dk_sum);
            }
            
            // Gradient w.r.t. v
            #pragma unroll
            for (int j = 0; j < head_dim_v; ++j) {
                atomicAdd(&dv[batch_idx * num_heads * seq_len_kv * head_dim_v + 
                              head_idx * seq_len_kv * head_dim_v + 
                              (tile_start + i) * head_dim_v + j],
                          attention_weight * s_do[0][j]);
            }
        }
        __syncthreads();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward (CUDA)");
    m.def("backward", &flash_attention_backward, "FlashAttention backward (CUDA)");
}                    
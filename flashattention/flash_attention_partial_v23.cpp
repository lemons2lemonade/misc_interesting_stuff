// Flash Attention Partial Implementation (V2/V3 features)
// This file implements a subset of optimizations from Flash Attention V2/V3

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <memory>

// Constants for optimized Flash Attention
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Add CUDA error checking macro
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(0)

// Add kernel launch error checking
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::string error_msg = "CUDA error: "; \
        error_msg += cudaGetErrorString(err_); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

// ===============================
// 0. KV CACHE STRUCTURE & MANAGEMENT
// ===============================

struct KVCache {
    torch::Tensor keys;    // [batch_size, num_heads, max_seq_len, head_dim]
    torch::Tensor values;  // [batch_size, num_heads, max_seq_len, head_dim]
    int current_length;    // Current filled length in the cache
    
    // Add explicit destructor
    ~KVCache() {
        // Ensure any manual resources are freed
        // PyTorch tensors will clean themselves up automatically
    }
};

// Function to initialize a new KV cache
std::unique_ptr<KVCache> initialize_kv_cache(
    int batch_size, 
    int num_heads, 
    int max_seq_len, 
    int head_dim, 
    torch::TensorOptions options) {
    
    auto keys = torch::zeros({batch_size, num_heads, max_seq_len, head_dim}, options);
    auto values = torch::zeros({batch_size, num_heads, max_seq_len, head_dim}, options);
    return std::make_unique<KVCache>(keys, values, 0);
}

// Function to update the KV cache with new keys and values
void update_kv_cache(
    KVCache& cache, 
    const torch::Tensor& new_keys, 
    const torch::Tensor& new_values) {
    
    // Make sure any pending operations that use the cache are complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update cache (original implementation)
    int new_tokens = new_keys.size(2);
    int new_end = cache.current_length + new_tokens;
    
    // Cache update operations
    cache.keys.index({"...", torch::indexing::Slice(cache.current_length, new_end)}) = new_keys;
    cache.values.index({"...", torch::indexing::Slice(cache.current_length, new_end)}) = new_values;
    
    // Update length
    cache.current_length = new_end;
    
    // Ensure updates are complete before returning
    CUDA_CHECK(cudaGetLastError());
}

// Add after the KVCache struct definition:
void reorder_kv_cache(
    KVCache& cache,
    const torch::Tensor& beam_idx) {
    
    // Check that cache is valid
    TORCH_CHECK(cache.keys.defined() && cache.values.defined(), 
               "KV cache contains undefined tensors");
    TORCH_CHECK(cache.keys.sizes() == cache.values.sizes(),
               "Keys and values must have the same dimensions");
    
    // Validate inputs
    TORCH_CHECK(beam_idx.device().is_cuda(), "beam_idx must be a CUDA tensor");
    TORCH_CHECK(beam_idx.dim() == 1, "beam_idx must be a 1D tensor");
    TORCH_CHECK(beam_idx.size(0) == cache.keys.size(0), 
               "beam_idx size must match batch dimension of KV cache");
    
    auto batch_size = cache.keys.size(0);
    auto num_heads = cache.keys.size(1);
    auto seq_len = cache.current_length;
    auto head_dim = cache.keys.size(3);
    
    // Create new buffers to hold the reordered cache
    auto options = cache.keys.options();
    auto new_keys = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    auto new_values = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    
    // On CPU: Get beam indices to reorder the batches
    auto beam_idx_cpu = beam_idx.to(torch::kCPU);
    auto beam_idx_ptr = beam_idx_cpu.data_ptr<int64_t>();
    
    // Reorder the keys and values based on beam indices
    for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int64_t src_idx = beam_idx_ptr[batch_idx];
        
        // Validate source index
        TORCH_CHECK(src_idx >= 0 && src_idx < batch_size, 
                   "Invalid beam index: ", src_idx, " at position ", batch_idx);
        
        // Copy keys and values from source to target
        new_keys[batch_idx] = cache.keys[src_idx];
        new_values[batch_idx] = cache.values[src_idx];
    }
    
    // Update the cache with the reordered buffers
    cache.keys.copy_(new_keys);
    cache.values.copy_(new_values);
}

// Consider memory pool reuse for frequent allocations/deallocations
// Add a resize method to efficiently handle growing caches
void resize_kv_cache(KVCache& cache, int new_size) {
    if (new_size > cache.keys.size(2)) {
        // Create larger tensors
        auto options = cache.keys.options();
        torch::Tensor new_keys = torch::zeros({cache.keys.size(0), cache.keys.size(1), 
                                              new_size, cache.keys.size(3)}, options);
        torch::Tensor new_values = torch::zeros({cache.values.size(0), cache.values.size(1), 
                                                new_size, cache.values.size(3)}, options);
        
        // Copy existing data
        new_keys.index({"...", torch::indexing::Slice(0, cache.current_length)}) = 
            cache.keys.index({"...", torch::indexing::Slice(0, cache.current_length)});
        new_values.index({"...", torch::indexing::Slice(0, cache.current_length)}) = 
            cache.values.index({"...", torch::indexing::Slice(0, cache.current_length)});
        
        // Replace with larger buffers
        cache.keys = new_keys;
        cache.values = new_values;
    }
}

// ===============================
// 1. KERNEL DECLARATIONS
// ===============================

template <typename scalar_t>
__global__ void flash_attention_v3_forward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* o,
    scalar_t* l,
    scalar_t* m,
    const bool* attention_mask,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool causal);

template <typename scalar_t>
__global__ void flash_attention_v3_backward_kernel(
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
    const bool* attention_mask,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool causal);

// New kernel for incremental attention during decoding
template <typename scalar_t>
__global__ void flash_attention_incremental_kernel(
    const scalar_t* q,                // [batch, heads, query_len, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, query_len, dim]
    const int batch_size,
    const int num_heads,
    const int query_len,              
    const int cache_len,              
    const int head_dim,
    const float softmax_scale,
    const bool causal);

// Specialized kernel for single-token generation (common case)
template <typename scalar_t>
__global__ void flash_attention_single_token_kernel(
    const scalar_t* q,                // [batch, heads, 1, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, 1, dim]
    const int batch_size,
    const int num_heads,
    const int cache_len,              
    const int head_dim,
    const float softmax_scale,
    const bool causal);

// Sliding window attention kernel for very long sequences
template <typename scalar_t>
__global__ void flash_attention_sliding_window_kernel(
    const scalar_t* q,                // [batch, heads, query_len, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, query_len, dim]
    const int batch_size,
    const int num_heads,
    const int query_len,
    const int cache_len,              
    const int head_dim,
    const int window_size,           // Size of the sliding window
    const float softmax_scale,
    const bool causal);

// ===============================
// 2. C++ INTERFACE IMPLEMENTATION
// ===============================

std::vector<torch::Tensor> flash_attention_v3_forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor attention_mask,
    float softmax_scale,
    bool causal) {
    
    // Calculate optimal block size based on tensor dimensions and GPU architecture
    const int seq_len = q.size(2);
    const int head_dim = q.size(3);
    
    // Determine optimal tile size based on shared memory constraints
    const int tile_size = std::min(128, seq_len);
    
    // Dynamic shared memory allocation
    const int shared_mem_size = 2 * tile_size * head_dim * sizeof(float);
    
    // Check if shared memory size is within limits
    int dev_id;
    cudaGetDevice(&dev_id);
    int shared_mem_per_block;
    cudaDeviceGetAttribute(&shared_mem_per_block, 
                          cudaDevAttrMaxSharedMemoryPerBlock, 
                          dev_id);
    
    TORCH_CHECK(shared_mem_size <= shared_mem_per_block,
               "Required shared memory exceeds device limits");
    
    // Configure kernel dimensions
    dim3 grid(q.size(0), q.size(1), (seq_len + tile_size - 1) / tile_size);
    dim3 block(std::min(512, tile_size * head_dim / 32 * 32));  // Round to warp size
    
    // Launch the kernel with dynamic shared memory
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_forward", ([&] {
        flash_attention_v3_forward_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            attention_mask.data_ptr<bool>(),
            q.size(0),
            q.size(1),
            seq_len,
            k.size(2),
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Error check
    CUDA_CHECK(cudaGetLastError());

    return {o, l, m};
}

std::vector<torch::Tensor> flash_attention_v3_backward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& o,
    torch::Tensor& l,
    torch::Tensor& m,
    torch::Tensor& dout,
    torch::Tensor attention_mask,
    float softmax_scale,
    bool causal) {
    
    // Input validation
    TORCH_CHECK(q.dim() == 4, "q must be a 4D tensor");
    TORCH_CHECK(k.dim() == 4, "k must be a 4D tensor");
    TORCH_CHECK(v.dim() == 4, "v must be a 4D tensor");
    TORCH_CHECK(o.dim() == 4, "o must be a 4D tensor");
    TORCH_CHECK(dout.dim() == 4, "dout must be a 4D tensor");
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len_q = q.size(2);
    auto head_dim = q.size(3);
    auto seq_len_kv = k.size(2);
    
    // Validate sizes of all tensors
    // (Similar validation as forward pass)
    
    // Create output gradient tensors
    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    
    // Using 2D grid for better parallelism
    constexpr int BLOCK_SIZE_M = 64; // Number of queries per block
    const dim3 blocks(batch_size * num_heads, (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    // Get attention mask pointer (null if not provided)
    const bool* mask_ptr = nullptr;
    if (attention_mask.defined()) {
        // Validate mask dimensions
        mask_ptr = attention_mask.data_ptr<bool>();
    }
    
    // Launch kernel with appropriate type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_v3_backward", ([&] {
        flash_attention_v3_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
            mask_ptr,
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_kv,
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // Only if synchronization is needed

    return {dq, dk, dv};
}

// Add new KV caching forward function definition
std::vector<torch::Tensor> flash_attention_v3_forward_with_kv_cache(
    torch::Tensor& q,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor attention_mask,
    float softmax_scale,
    bool causal) {
    
    // Validate input tensors
    TORCH_CHECK(q.dim() == 4, "q must be a 4D tensor");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be a 4D tensor");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be a 4D tensor");
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len_q = q.size(2);
    auto head_dim = q.size(3);
    auto seq_len_kv = k_cache.size(2);
    
    // Validate that cached KV tensors match expected dimensions
    TORCH_CHECK(k_cache.size(0) == batch_size, "batch size of k_cache and q must match");
    TORCH_CHECK(k_cache.size(1) == num_heads, "num_heads of k_cache and q must match");
    TORCH_CHECK(k_cache.size(3) == head_dim, "head_dim of k_cache and q must match");
    TORCH_CHECK(v_cache.size(0) == batch_size, "batch size of v_cache and q must match");
    TORCH_CHECK(v_cache.size(1) == num_heads, "num_heads of v_cache and q must match");
    TORCH_CHECK(v_cache.size(2) == seq_len_kv, "seq_len of v_cache and k_cache must match");
    TORCH_CHECK(v_cache.size(3) == head_dim, "head_dim of v_cache and q must match");
    
    // Create output tensors
    auto o = torch::zeros({batch_size, num_heads, seq_len_q, head_dim}, q.options());
    auto l = torch::zeros({batch_size, num_heads, seq_len_q}, q.options());
    auto m = torch::full({batch_size, num_heads, seq_len_q}, -std::numeric_limits<float>::infinity(), q.options());
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    
    // Using 2D grid for better parallelism
    constexpr int BLOCK_SIZE_M = 64;
    const dim3 blocks(batch_size * num_heads, (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    // Get attention mask pointer
    const bool* mask_ptr = nullptr;
    if (attention_mask.defined()) {
        TORCH_CHECK(attention_mask.dim() == 4, "attention_mask must be a 4D tensor");
        TORCH_CHECK(attention_mask.size(0) == batch_size, "batch size of attention_mask and q must match");
        TORCH_CHECK(attention_mask.size(1) == num_heads, "num_heads of attention_mask and q must match");
        TORCH_CHECK(attention_mask.size(2) == seq_len_q, "seq_len_q of attention_mask and q must match");
        TORCH_CHECK(attention_mask.size(3) == seq_len_kv, "seq_len_kv of attention_mask and k_cache must match");
        mask_ptr = attention_mask.data_ptr<bool>();
    }
    
    // Launch kernel
    // Note: We reuse the same kernel as it already handles the KV data properly
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_v3_forward_with_kv_cache", ([&] {
        flash_attention_v3_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            mask_ptr,
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_kv,
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // Only if synchronization is needed

    return {o, l, m};
}

// Forward function with key-value cache for efficient inference
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_with_kv_cache(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::optional<torch::Tensor> attn_mask,
    float softmax_scale,
    bool causal) {
    
    // Input validation
    // q: [batch_size, num_heads, seq_len_q, head_dim]
    // k_cache: [batch_size, num_heads, seq_len_k, head_dim]
    // v_cache: [batch_size, num_heads, seq_len_k, head_dim]
    TORCH_CHECK(q.dim() == 4, "Query tensor must be 4-dimensional");
    TORCH_CHECK(k_cache.dim() == 4, "Key cache tensor must be 4-dimensional");
    TORCH_CHECK(v_cache.dim() == 4, "Value cache tensor must be 4-dimensional");
    
    // Check device & layout
    TORCH_CHECK(q.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(k_cache.is_cuda(), "Key cache tensor must be on CUDA device");
    TORCH_CHECK(v_cache.is_cuda(), "Value cache tensor must be on CUDA device");
    
    TORCH_CHECK(q.size(0) == k_cache.size(0) && q.size(0) == v_cache.size(0), 
                "Batch sizes must match across tensors");
    TORCH_CHECK(q.size(1) == k_cache.size(1) && q.size(1) == v_cache.size(1), 
                "Number of heads must match across tensors");
    TORCH_CHECK(k_cache.size(2) == v_cache.size(2), 
                "Sequence lengths of key cache and value cache must match");
    TORCH_CHECK(q.size(3) == k_cache.size(3) && q.size(3) == v_cache.size(3), 
                "Head dimensions must match across tensors");
    
    // If attention mask is provided, check its shape
    if (attn_mask.has_value()) {
        auto mask = attn_mask.value();
        TORCH_CHECK(mask.dim() == 4, "Attention mask must be 4-dimensional");
        TORCH_CHECK(mask.size(0) == q.size(0) && mask.size(1) == q.size(1),
                    "Attention mask batch and heads dimensions must match query");
        TORCH_CHECK(mask.size(2) == q.size(2) && mask.size(3) == k_cache.size(2),
                    "Attention mask sequence dimensions must match query and key cache");
        TORCH_CHECK(mask.is_cuda(), "Attention mask must be on CUDA device");
        TORCH_CHECK(mask.scalar_type() == torch::ScalarType::Bool, 
                    "Attention mask must be a boolean tensor");
    }
    
    // Get dimensions
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len_q = q.size(2);
    int seq_len_k = k_cache.size(2);
    int head_dim = q.size(3);
    
    // Prepare output tensor and workspace tensors
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, head_dim}, 
                              q.options());
    
    // These tensors store intermediate values needed for backward pass
    auto logsumexp = torch::zeros({batch_size, num_heads, seq_len_q}, 
                                 q.options());
    auto max_score = torch::zeros({batch_size, num_heads, seq_len_q}, 
                                 q.options());
    
    // Get pointers to CUDA memory
    void* mask_ptr = nullptr;
    if (attn_mask.has_value()) {
        mask_ptr = attn_mask.value().data_ptr<bool>();
    }
    
    // Configure kernel launch parameters for efficient incremental inference
    const int block_size = 256;
    dim3 grid_dim(batch_size, num_heads, seq_len_q);  // One block per query position
    dim3 block_dim(block_size);
    
    // Launch the incremental kernel for efficient KV cache processing
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_incremental", ([&] {
        flash_attention_incremental_kernel<scalar_t><<<grid_dim, block_dim, head_dim * sizeof(scalar_t)>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return std::make_tuple(output, logsumexp, max_score);
}

// Optimized incremental attention function using KV cache
torch::Tensor flash_attention_incremental_forward(
    const torch::Tensor& query,
    const KVCache& kv_cache,
    torch::optional<torch::Tensor> mask,
    float softmax_scale,
    bool causal) {
    
    // Extract dimensions
    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int query_len = query.size(2);
    int head_dim = query.size(3);
    int cache_len = kv_cache.current_length;
    
    // Validation
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(kv_cache.keys.is_cuda(), "KV cache must be on CUDA device");
    TORCH_CHECK(query.scalar_type() == kv_cache.keys.scalar_type(), 
                "Query and KV cache must have the same data type");
    
    // Optional mask validation
    if (mask.has_value()) {
        auto mask_tensor = mask.value();
        TORCH_CHECK(mask_tensor.dim() == 4, "Mask must be 4-dimensional");
        TORCH_CHECK(mask_tensor.size(0) == batch_size && mask_tensor.size(1) == num_heads,
                  "Mask batch and heads dimensions must match query");
        TORCH_CHECK(mask_tensor.size(2) == query_len && mask_tensor.size(3) >= cache_len,
                  "Mask sequence dimensions must be compatible with query and cache");
    }
    
    // Create output tensor
    auto output = torch::zeros({batch_size, num_heads, query_len, head_dim}, query.options());
    
    // Launch kernel with optimized grid/block dimensions
    dim3 grid(batch_size, num_heads, query_len);
    dim3 block(256);  // Adjust based on SM occupancy and head_dim
    
    // Shared memory size (for query)
    size_t smem_size = head_dim * sizeof(float);
    
    void* mask_ptr = nullptr;
    if (mask.has_value()) {
        mask_ptr = mask.value().data_ptr<bool>();
    }
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "flash_attention_incremental_forward", ([&] {
        flash_attention_incremental_kernel<scalar_t><<<grid, block, smem_size>>>(
            query.data_ptr<scalar_t>(),
            kv_cache.keys.data_ptr<scalar_t>(),
            kv_cache.values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            query_len,
            cache_len,
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// Optimized function for single-token decoding (common case in inference)
torch::Tensor flash_attention_single_token_forward(
    const torch::Tensor& query,
    const KVCache& kv_cache,
    torch::optional<torch::Tensor> mask,
    float softmax_scale,
    bool causal) {
    
    // Validation
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(kv_cache.keys.is_cuda(), "KV cache must be on CUDA device");
    TORCH_CHECK(query.size(2) == 1, "This function is optimized for single-token generation (query_len must be 1)");
    
    // Extract dimensions
    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int head_dim = query.size(3);
    int cache_len = kv_cache.current_length;

    // Ensure data types match
    TORCH_CHECK(query.scalar_type() == kv_cache.keys.scalar_type(), 
                "Query and KV cache must have the same data type");
    
    // Create output tensor
    auto output = torch::zeros({batch_size, num_heads, 1, head_dim}, query.options());
    
    // Get mask pointer
    void* mask_ptr = nullptr;
    if (mask.has_value()) {
        auto mask_tensor = mask.value();
        TORCH_CHECK(mask_tensor.dim() == 4, "Mask must be 4-dimensional");
        TORCH_CHECK(mask_tensor.size(0) == batch_size && mask_tensor.size(1) == num_heads,
                  "Mask batch and heads dimensions must match query");
        TORCH_CHECK(mask_tensor.size(2) == 1 && mask_tensor.size(3) >= cache_len,
                  "Mask sequence dimensions must be compatible with query and cache");
        mask_ptr = mask_tensor.data_ptr<bool>();
    }
    
    // Larger block size for better occupancy with single token
    const int block_size = 256;
    dim3 block(block_size);
    
    // Optimized kernel launch configuration for single-token case
    dim3 grid(batch_size, num_heads, 1);  // One block per batch/head combination
    
    // Shared memory for query, keys, and values
    size_t smem_size = head_dim * sizeof(float) * 3;  // 3x for q, k, v
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "flash_attention_single_token", ([&] {
        flash_attention_single_token_kernel<scalar_t><<<grid, block, smem_size>>>(
            query.data_ptr<scalar_t>(),
            kv_cache.keys.data_ptr<scalar_t>(),
            kv_cache.values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            cache_len,
            head_dim,
            softmax_scale,
            causal
        );
    }));
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// Sliding window attention for very long sequences
torch::Tensor flash_attention_sliding_window_forward(
    const torch::Tensor& query,
    const KVCache& kv_cache,
    torch::optional<torch::Tensor> mask,
    int window_size,
    float softmax_scale,
    bool causal) {
    
    // Validation
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(kv_cache.keys.is_cuda(), "KV cache must be on CUDA device");
    TORCH_CHECK(query.scalar_type() == kv_cache.keys.scalar_type(), 
               "Query and KV cache must have the same data type");
    TORCH_CHECK(window_size > 0, "Window size must be positive");
    
    // Extract dimensions
    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int query_len = query.size(2);
    int head_dim = query.size(3);
    int cache_len = kv_cache.current_length;
    
    // Ensure window size doesn't exceed cache length
    window_size = std::min(window_size, cache_len);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, num_heads, query_len, head_dim}, query.options());
    
    // Configure kernel launch parameters
    dim3 grid(batch_size, num_heads, query_len);
    dim3 block(256);
    
    // Shared memory for query, keys, and values
    size_t smem_size = head_dim * sizeof(float) * 3;  // 3x for q, k, v
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "flash_attention_sliding_window", ([&] {
        flash_attention_sliding_window_kernel<scalar_t><<<grid, block, smem_size>>>(
            query.data_ptr<scalar_t>(),
            kv_cache.keys.data_ptr<scalar_t>(),
            kv_cache.values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            query_len,
            cache_len,
            head_dim,
            window_size,
            softmax_scale,
            causal
        );
    }));
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// ===============================
// 3. FORWARD KERNEL IMPLEMENTATION
// ===============================

template <typename scalar_t>
__global__ void flash_attention_v3_forward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* o,
    scalar_t* l,
    scalar_t* m,
    const bool* attention_mask,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool causal) {
    
    // Block tiling constants
    constexpr int BLOCK_SIZE_M = 64; // Queries per block
    constexpr int BLOCK_SIZE_N = 64; // Keys/values per block
    constexpr int BLOCK_SIZE_K = 32; // Head dimension chunk size for better register usage
    
    // Thread indexing
    int tidx = threadIdx.x;
    int lane_idx = tidx % WARP_SIZE;
    int warp_idx = tidx / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Block indexing for 2D grid
    int batch_head_idx = blockIdx.x;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    
    // Calculate starting sequence index for this thread block
    int seq_block_start = blockIdx.y * BLOCK_SIZE_M;
    
    // Shared memory for Q, K, V tiles
    extern __shared__ char shared_memory[];
    scalar_t* s_k = (scalar_t*)shared_memory;
    scalar_t* s_v = (scalar_t*)(s_k + BLOCK_SIZE_N * head_dim);
    
    // Each warp processes a subset of queries
    for (int seq_idx_offset = 0; seq_idx_offset < BLOCK_SIZE_M; seq_idx_offset += warps_per_block) {
        int seq_idx = seq_block_start + seq_idx_offset + warp_idx;
        
        // Skip if this query is out of bounds
        if (seq_idx >= seq_len_q) continue;
        
        // Load Q into registers for this query
        scalar_t q_regs[BLOCK_SIZE_K];
        #pragma unroll
        for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
            if (i < head_dim) {
                q_regs[i % BLOCK_SIZE_K] = q[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i];
            }
        }
        
        // Initialize accumulators for this query
        scalar_t o_regs[BLOCK_SIZE_K] = {0};
        scalar_t m_reg = -INFINITY;
        scalar_t l_reg = 0;
        
        // Process key-value blocks
        for (int kv_block_start = 0; kv_block_start < seq_len_kv; kv_block_start += BLOCK_SIZE_N) {
            // Load K and V blocks into shared memory with coalesced memory access
            for (int i = tidx; i < BLOCK_SIZE_N * head_dim; i += blockDim.x) {
                int kv_seq_idx = kv_block_start + (i / head_dim);
                int dim_idx = i % head_dim;
                
                if (kv_seq_idx < seq_len_kv) {
                    s_k[i] = k[((batch_idx * num_heads + head_idx) * seq_len_kv + kv_seq_idx) * head_dim + dim_idx];
                    s_v[i] = v[((batch_idx * num_heads + head_idx) * seq_len_kv + kv_seq_idx) * head_dim + dim_idx];
                } else {
                    // Zero-pad if needed
                    s_k[i] = 0;
                    s_v[i] = 0;
                }
            }
            __syncthreads();
            
            // Process keys and compute attention scores with tiling
            for (int kv_idx = 0; kv_idx < min(BLOCK_SIZE_N, seq_len_kv - kv_block_start); ++kv_idx) {
                // Apply causal masking if needed
                if (causal && (kv_block_start + kv_idx) > seq_idx) continue;
                
                // Apply attention mask if provided
                if (attention_mask != nullptr) {
                    bool mask_value = attention_mask[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * seq_len_kv + (kv_block_start + kv_idx)];
                    if (!mask_value) continue;
                }
                
                // Compute Q·K for this key
                scalar_t qk_score = 0;
                
                // Each lane computes part of the dot product
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    qk_score += q_regs[i % BLOCK_SIZE_K] * s_k[kv_idx * head_dim + i];
                }
                
                // Warp-level reduction using shuffle
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    qk_score += __shfl_down_sync(FULL_MASK, qk_score, offset);
                }
                
                // Broadcast score to all lanes in the warp
                qk_score = __shfl_sync(FULL_MASK, qk_score, 0);
                
                // Apply softmax scaling
                qk_score *= softmax_scale;
                
                // Compute softmax: first update max value
                scalar_t m_prev = m_reg;
                m_reg = max(m_reg, qk_score);
                
                // Update running sum with scaling
                scalar_t scale_factor = exp(m_prev - m_reg);
                l_reg = l_reg * scale_factor + exp(qk_score - m_reg);
                
                // Scale the output accumulator based on updated max
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    o_regs[i % BLOCK_SIZE_K] *= scale_factor;
                }
                
                // Compute and accumulate attention-weighted values
                scalar_t attention_weight = exp(qk_score - m_reg) / l_reg;
                
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    o_regs[i % BLOCK_SIZE_K] += attention_weight * s_v[kv_idx * head_dim + i];
                }
            }
            
            __syncthreads(); // Ensure all warps are done with shared memory
        }
        
        // Write output, l, and m to global memory
        for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
            o[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i] = o_regs[i % BLOCK_SIZE_K];
        }
        
        if (lane_idx == 0) {
            l[((batch_idx * num_heads + head_idx) * seq_len_q) + seq_idx] = l_reg;
            m[((batch_idx * num_heads + head_idx) * seq_len_q) + seq_idx] = m_reg;
        }
    }
}

// ===============================
// 4. BACKWARD KERNEL IMPLEMENTATION
// ===============================

template <typename scalar_t>
__global__ void flash_attention_v3_backward_kernel(
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
    const bool* attention_mask,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool causal) {
    
    // Block tiling constants
    constexpr int BLOCK_SIZE_M = 64; // Queries per block
    constexpr int BLOCK_SIZE_N = 64; // Keys/values per block
    constexpr int BLOCK_SIZE_K = 32; // Head dimension chunk size
    
    // Thread indexing
    int tidx = threadIdx.x;
    int lane_idx = tidx % WARP_SIZE;
    int warp_idx = tidx / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Block indexing for 2D grid
    int batch_head_idx = blockIdx.x;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    
    // Calculate starting sequence index for this thread block
    int seq_block_start = blockIdx.y * BLOCK_SIZE_M;
    
    // Shared memory for K, V, and gradients
    extern __shared__ char shared_memory[];
    scalar_t* s_k = (scalar_t*)shared_memory;
    scalar_t* s_v = (scalar_t*)(s_k + BLOCK_SIZE_N * head_dim);
    scalar_t* s_do = (scalar_t*)(s_v + BLOCK_SIZE_N * head_dim);
    
    // Each warp processes a subset of queries
    for (int seq_idx_offset = 0; seq_idx_offset < BLOCK_SIZE_M; seq_idx_offset += warps_per_block) {
        int seq_idx = seq_block_start + seq_idx_offset + warp_idx;
        
        // Skip if this query is out of bounds
        if (seq_idx >= seq_len_q) continue;
        
        // Load Q and dO into registers for this query
        scalar_t q_regs[BLOCK_SIZE_K];
        scalar_t do_regs[BLOCK_SIZE_K];
        scalar_t o_regs[BLOCK_SIZE_K];
        
        #pragma unroll
        for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
            if (i < head_dim) {
                q_regs[i % BLOCK_SIZE_K] = q[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i];
                do_regs[i % BLOCK_SIZE_K] = dout[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i];
                o_regs[i % BLOCK_SIZE_K] = o[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i];
            }
        }
        
        // Get softmax normalization terms
        scalar_t m_val = m[((batch_idx * num_heads + head_idx) * seq_len_q) + seq_idx];
        scalar_t l_val = l[((batch_idx * num_heads + head_idx) * seq_len_q) + seq_idx];
        
        // Compute dO dot O (needed for gradient computation)
        scalar_t do_o_sum = 0;
        for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
            do_o_sum += do_regs[i % BLOCK_SIZE_K] * o_regs[i % BLOCK_SIZE_K];
        }
        
        // Warp-level reduction of do_o_sum
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            do_o_sum += __shfl_down_sync(FULL_MASK, do_o_sum, offset);
        }
        
        // Broadcast to all lanes
        do_o_sum = __shfl_sync(FULL_MASK, do_o_sum, 0);
        
        // Initialize dQ accumulator for this query
        scalar_t dq_regs[BLOCK_SIZE_K] = {0};
        
        // Process key-value blocks
        for (int kv_block_start = 0; kv_block_start < seq_len_kv; kv_block_start += BLOCK_SIZE_N) {
            // Load K and V blocks into shared memory
            for (int i = tidx; i < BLOCK_SIZE_N * head_dim; i += blockDim.x) {
                int kv_seq_idx = kv_block_start + (i / head_dim);
                int dim_idx = i % head_dim;
                
                if (kv_seq_idx < seq_len_kv) {
                    s_k[i] = k[((batch_idx * num_heads + head_idx) * seq_len_kv + kv_seq_idx) * head_dim + dim_idx];
                    s_v[i] = v[((batch_idx * num_heads + head_idx) * seq_len_kv + kv_seq_idx) * head_dim + dim_idx];
                } else {
                    s_k[i] = 0;
                    s_v[i] = 0;
                }
            }
            
            // Pre-load dO into shared memory for efficient access
            for (int i = tidx; i < BLOCK_SIZE_M * head_dim; i += blockDim.x) {
                int q_seq_idx = seq_block_start + (i / head_dim);
                int dim_idx = i % head_dim;
                
                if (q_seq_idx < seq_len_q) {
                    s_do[i] = dout[((batch_idx * num_heads + head_idx) * seq_len_q + q_seq_idx) * head_dim + dim_idx];
                } else {
                    s_do[i] = 0;
                }
            }
            
            __syncthreads();
            
            // Process keys and compute gradients with tiling
            for (int kv_idx = 0; kv_idx < min(BLOCK_SIZE_N, seq_len_kv - kv_block_start); ++kv_idx) {
                int global_kv_idx = kv_block_start + kv_idx;
                
                // Apply causal masking if needed
                if (causal && global_kv_idx > seq_idx) continue;
                
                // Apply attention mask if provided
                if (attention_mask != nullptr) {
                    bool mask_value = attention_mask[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * seq_len_kv + global_kv_idx];
                    if (!mask_value) continue;
                }
                
                // Compute Q·K for this key
                scalar_t qk_score = 0;
                
                // Each lane computes part of the dot product
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    qk_score += q_regs[i % BLOCK_SIZE_K] * s_k[kv_idx * head_dim + i];
                }
                
                // Warp-level reduction
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    qk_score += __shfl_down_sync(FULL_MASK, qk_score, offset);
                }
                
                // Broadcast to all lanes
                qk_score = __shfl_sync(FULL_MASK, qk_score, 0);
                qk_score *= softmax_scale;
                
                // Compute attention weight with stable softmax
                scalar_t attention_weight = exp(qk_score - m_val) / l_val;
                
                // Compute dV contribution
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    scalar_t dv_val = attention_weight * do_regs[i % BLOCK_SIZE_K];
                    atomicAdd(&dv[((batch_idx * num_heads + head_idx) * seq_len_kv + global_kv_idx) * head_dim + i], dv_val);
                }
                
                // Compute dQ and dK contributions
                scalar_t d_softmax = 0;
                
                // Compute dO·V
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    d_softmax += do_regs[i % BLOCK_SIZE_K] * s_v[kv_idx * head_dim + i];
                }
                
                // Warp-level reduction
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    d_softmax += __shfl_down_sync(FULL_MASK, d_softmax, offset);
                }
                
                // Broadcast to all lanes
                d_softmax = __shfl_sync(FULL_MASK, d_softmax, 0);
                
                // Compute gradient through softmax
                scalar_t ds = attention_weight * (d_softmax - do_o_sum);
                
                // Compute dQ contribution
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    dq_regs[i % BLOCK_SIZE_K] += ds * softmax_scale * s_k[kv_idx * head_dim + i];
                }
                
                // Compute dK contribution
                for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
                    scalar_t dk_val = ds * softmax_scale * q_regs[i % BLOCK_SIZE_K];
                    atomicAdd(&dk[((batch_idx * num_heads + head_idx) * seq_len_kv + global_kv_idx) * head_dim + i], dk_val);
                }
            }
            
            __syncthreads();
        }
        
        // Write dQ to global memory
        for (int i = lane_idx; i < head_dim; i += WARP_SIZE) {
            dq[((batch_idx * num_heads + head_idx) * seq_len_q + seq_idx) * head_dim + i] = dq_regs[i % BLOCK_SIZE_K];
        }
    }
}

// ===============================
// 5. INCREMENTAL KERNEL IMPLEMENTATION
// ===============================

template <typename scalar_t>
__global__ void flash_attention_incremental_kernel(
    const scalar_t* q,                // [batch, heads, query_len, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, query_len, dim]
    const int batch_size,
    const int num_heads,
    const int query_len,              
    const int cache_len,              
    const int head_dim,
    const float softmax_scale,
    const bool causal) {
    
    // Get batch, head, and query position indices
    int b = blockIdx.x;     // Batch index
    int h = blockIdx.y;     // Head index
    int q_idx = blockIdx.z; // Query index
    
    // Thread index within the block
    int tid = threadIdx.x;
    
    // Shared memory for efficient computation
    extern __shared__ char shared_memory[];
    scalar_t* s_q = (scalar_t*)shared_memory;
    
    // For a single query, load it into shared memory
    // Each thread loads one or more elements depending on head_dim
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q[((b * num_heads + h) * query_len + q_idx) * head_dim + i];
    }
    
    __syncthreads();
    
    // Register for accumulating output values
    scalar_t output_regs[16] = {0};  // Assuming head_dim <= 16*blockDim.x
    
    // Local variables for softmax computation
    scalar_t max_val = -INFINITY;
    scalar_t sum_exp = 0.0f;
    
    // First pass: find max value for softmax stability
    for (int k_idx = 0; k_idx < cache_len; k_idx++) {
        // Apply causal masking if needed
        if (causal && k_idx >= q_idx + 1) {
            continue;
        }
        
        // Compute dot product between query and key
        scalar_t qk_sum = 0.0f;
        
        for (int i = tid; i < head_dim; i += blockDim.x) {
            qk_sum += (float)s_q[i] * (float)k_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
        }
        
        // Warp reduction to sum up the dot product
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
        }
        
        // First thread in the warp finds the max
        if (tid % 32 == 0) {
            qk_sum *= softmax_scale;
            max_val = max(max_val, qk_sum);
        }
    }
    
    // Broadcast max_val to all threads in the warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Second pass: compute softmax and accumulate weighted values
    for (int k_idx = 0; k_idx < cache_len; k_idx++) {
        // Apply causal masking if needed
        if (causal && k_idx >= q_idx + 1) {
            continue;
        }
        
        // Recompute dot product
        scalar_t qk_sum = 0.0f;
        
        for (int i = tid; i < head_dim; i += blockDim.x) {
            qk_sum += (float)s_q[i] * (float)k_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
        }
        
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
        }
        
        // First thread computes the softmax weight
        scalar_t weight = 0.0f;
        if (tid % 32 == 0) {
            qk_sum *= softmax_scale;
            weight = exp(qk_sum - max_val);
            sum_exp += weight;
        }
        
        // Broadcast weight to all threads in the warp
        weight = __shfl_sync(0xffffffff, weight, 0);
        
        // Accumulate weighted value in registers
        for (int i = tid; i < head_dim; i += blockDim.x) {
            int reg_idx = i / blockDim.x;
            output_regs[reg_idx] += weight * v_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
        }
    }
    
    // Broadcast sum_exp to all threads
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    
    // Write output with normalization
    for (int i = tid; i < head_dim; i += blockDim.x) {
        int reg_idx = i / blockDim.x;
        output[((b * num_heads + h) * query_len + q_idx) * head_dim + i] = output_regs[reg_idx] / sum_exp;
    }
}

// Specialized kernel for single-token generation (common case)
template <typename scalar_t>
__global__ void flash_attention_single_token_kernel(
    const scalar_t* q,                // [batch, heads, 1, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, 1, dim]
    const int batch_size,
    const int num_heads,
    const int cache_len,              
    const int head_dim,
    const float softmax_scale,
    const bool causal) {
    
    // Get batch, head, and query position indices
    int b = blockIdx.x;     // Batch index
    int h = blockIdx.y;     // Head index
    int q_idx = 0;          // Query index - always 0 for single token
    
    // Thread index within the block
    int tid = threadIdx.x;
    
    // Shared memory for efficient computation
    extern __shared__ char shared_memory[];
    scalar_t* s_q = (scalar_t*)shared_memory;
    
    // Additional shared memory for keys and values to reduce global memory access
    scalar_t* s_k = (scalar_t*)(s_q + head_dim);
    scalar_t* s_v = (scalar_t*)(s_k + head_dim);
    
    // For a single query, load it into shared memory
    // Each thread loads one or more elements depending on head_dim
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q[((b * num_heads + h) * 1 + q_idx) * head_dim + i];
    }
    
    __syncthreads();
    
    // Register for accumulating output values - use float for better precision internally
    float output_regs[16] = {0};  // Assuming head_dim <= 16*blockDim.x
    
    // Local variables for softmax computation
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // Pre-compute the base index for the batch and head to reduce computation in the loop
    int64_t batch_head_offset = (b * num_heads + h) * cache_len * head_dim;
    
    // Tile the computation with prefetching
    const int TILE_SIZE = 16;  // Process 16 keys/values at a time
    
    // First pass: find max value for softmax stability
    for (int k_block = 0; k_block < cache_len; k_block += TILE_SIZE) {
        int k_end = min(k_block + TILE_SIZE, cache_len);
        
        // Prefetch next block of keys into L1 cache
        if (k_block + TILE_SIZE < cache_len) {
            for (int i = tid; i < head_dim; i += blockDim.x) {
                __builtin_prefetch(&k_cache[batch_head_offset + (k_block + TILE_SIZE) * head_dim + i], 0, 1);
            }
        }
        
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            // Apply causal masking if needed
            if (causal && k_idx >= cache_len - 1) {
                continue;
            }
            
            // Compute dot product between query and key
            float qk_sum = 0.0f;
            
            for (int i = tid; i < head_dim; i += blockDim.x) {
                qk_sum += (float)s_q[i] * (float)k_cache[batch_head_offset + k_idx * head_dim + i];
            }
            
            // Warp reduction to sum up the dot product
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
            }
            
            // First thread in the warp finds the max
            if (tid % 32 == 0) {
                qk_sum *= softmax_scale;
                max_val = max(max_val, qk_sum);
            }
        }
    }
    
    // Broadcast max_val to all threads in the warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Second pass: compute softmax and accumulate weighted values
    for (int k_block = 0; k_block < cache_len; k_block += TILE_SIZE) {
        int k_end = min(k_block + TILE_SIZE, cache_len);
        
        // Prefetch values for current block
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            for (int i = tid; i < head_dim; i += blockDim.x) {
                if (k_idx < cache_len && i < head_dim) {
                    __builtin_prefetch(&v_cache[batch_head_offset + k_idx * head_dim + i], 0, 1);
                }
            }
        }
        
        // Process the block of keys
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            // Apply causal masking if needed
            if (causal && k_idx >= cache_len - 1) {
                continue;
            }
            
            // Load key into shared memory for repeated access
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_k[i] = k_cache[batch_head_offset + k_idx * head_dim + i];
            }
            
            // Load value into shared memory for better locality
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_v[i] = v_cache[batch_head_offset + k_idx * head_dim + i];
            }
            
            __syncthreads();
            
            // Compute dot product using shared memory
            float qk_sum = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                qk_sum += (float)s_q[i] * (float)s_k[i];
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
            }
            
            // First thread computes the softmax weight
            float weight = 0.0f;
            if (tid % 32 == 0) {
                qk_sum *= softmax_scale;
                weight = exp(qk_sum - max_val);
                sum_exp += weight;
            }
            
            // Broadcast weight to all threads in the warp
            weight = __shfl_sync(0xffffffff, weight, 0);
            
            // Accumulate weighted value in registers using shared memory
            for (int i = tid; i < head_dim; i += blockDim.x) {
                int reg_idx = i / blockDim.x;
                output_regs[reg_idx] += weight * (float)s_v[i];
            }
            
            __syncthreads();
        }
    }
    
    // Broadcast sum_exp to all threads
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    
    // Write output with normalization
    for (int i = tid; i < head_dim; i += blockDim.x) {
        int reg_idx = i / blockDim.x;
        output[((b * num_heads + h) * 1 + q_idx) * head_dim + i] = (scalar_t)(output_regs[reg_idx] / sum_exp);
    }
}

// Sliding window attention kernel for very long sequences
template <typename scalar_t>
__global__ void flash_attention_sliding_window_kernel(
    const scalar_t* q,                // [batch, heads, query_len, dim]
    const scalar_t* k_cache,          // [batch, heads, cache_len, dim]
    const scalar_t* v_cache,          // [batch, heads, cache_len, dim]
    scalar_t* output,                 // [batch, heads, query_len, dim]
    const int batch_size,
    const int num_heads,
    const int query_len,
    const int cache_len,              
    const int head_dim,
    const int window_size,           // Size of the sliding window
    const float softmax_scale,
    const bool causal) {
    
    // Get batch, head, and query position indices
    int b = blockIdx.x;     // Batch index
    int h = blockIdx.y;     // Head index
    int q_idx = blockIdx.z; // Query index
    
    // Thread index within the block
    int tid = threadIdx.x;
    
    // Shared memory for efficient computation
    extern __shared__ char shared_memory[];
    scalar_t* s_q = (scalar_t*)shared_memory;
    
    // For a single query, load it into shared memory
    // Each thread loads one or more elements depending on head_dim
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q[((b * num_heads + h) * query_len + q_idx) * head_dim + i];
    }
    
    __syncthreads();
    
    // Register for accumulating output values
    scalar_t output_regs[16] = {0};  // Assuming head_dim <= 16*blockDim.x
    
    // Local variables for softmax computation
    scalar_t max_val = -INFINITY;
    scalar_t sum_exp = 0.0f;
    
    // First pass: find max value for softmax stability
    for (int k_block = 0; k_block < cache_len; k_block += window_size) {
        int k_end = min(k_block + window_size, cache_len);
        
        // Prefetch next block of keys into L1 cache
        if (k_block + window_size < cache_len) {
            for (int i = tid; i < head_dim; i += blockDim.x) {
                __builtin_prefetch(&k_cache[((b * num_heads + h) * cache_len + k_block + window_size) * head_dim + i], 0, 1);
            }
        }
        
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            // Apply causal masking if needed
            if (causal && k_idx >= cache_len - window_size) {
                continue;
            }
            
            // Compute dot product between query and key
            scalar_t qk_sum = 0.0f;
            
            for (int i = tid; i < head_dim; i += blockDim.x) {
                qk_sum += (float)s_q[i] * (float)k_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
            }
            
            // Warp reduction to sum up the dot product
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
            }
            
            // First thread in the warp finds the max
            if (tid % 32 == 0) {
                qk_sum *= softmax_scale;
                max_val = max(max_val, qk_sum);
            }
        }
    }
    
    // Broadcast max_val to all threads in the warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Second pass: compute softmax and accumulate weighted values
    for (int k_block = 0; k_block < cache_len; k_block += window_size) {
        int k_end = min(k_block + window_size, cache_len);
        
        // Prefetch values for current block
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            for (int i = tid; i < head_dim; i += blockDim.x) {
                if (k_idx < cache_len && i < head_dim) {
                    __builtin_prefetch(&v_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i], 0, 1);
                }
            }
        }
        
        // Process the block of keys
        for (int k_idx = k_block; k_idx < k_end; k_idx++) {
            // Apply causal masking if needed
            if (causal && k_idx >= cache_len - window_size) {
                continue;
            }
            
            // Load key into shared memory for repeated access
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_k[i] = k_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
            }
            
            // Load value into shared memory for better locality
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_v[i] = v_cache[((b * num_heads + h) * cache_len + k_idx) * head_dim + i];
            }
            
            __syncthreads();
            
            // Compute dot product using shared memory
            scalar_t qk_sum = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                qk_sum += (float)s_q[i] * (float)s_k[i];
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
            }
            
            // First thread computes the softmax weight
            scalar_t weight = 0.0f;
            if (tid % 32 == 0) {
                qk_sum *= softmax_scale;
                weight = exp(qk_sum - max_val);
                sum_exp += weight;
            }
            
            // Broadcast weight to all threads in the warp
            weight = __shfl_sync(0xffffffff, weight, 0);
            
            // Accumulate weighted value in registers using shared memory
            for (int i = tid; i < head_dim; i += blockDim.x) {
                int reg_idx = i / blockDim.x;
                output_regs[reg_idx] += weight * (float)s_v[i];
            }
            
            __syncthreads();
        }
    }
    
    // Broadcast sum_exp to all threads
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    
    // Write output with normalization
    for (int i = tid; i < head_dim; i += blockDim.x) {
        int reg_idx = i / blockDim.x;
        output[((b * num_heads + h) * query_len + q_idx) * head_dim + i] = (scalar_t)(output_regs[reg_idx] / sum_exp);
    }
}

// ===============================
// 6. PYTHON MODULE DEFINITION
// ===============================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_v3_forward, "FlashAttention V3 forward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("attention_mask") = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);
    
    m.def("forward_with_kv_cache", &flash_attention_v3_forward_with_kv_cache, 
          "FlashAttention V3 forward with KV cache (CUDA)",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), 
          py::arg("attention_mask") = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);
    
    m.def("backward", &flash_attention_v3_backward, "FlashAttention V3 backward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("o"), py::arg("l"), py::arg("m"), py::arg("dout"),
          py::arg("attention_mask") = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);

    m.def("forward_with_kv_cache", &forward_with_kv_cache, "Flash Attention V3 forward function with KV cache");

    // Add KV cache structure
    py::class_<KVCache>(m, "KVCache")
        .def(py::init<>())
        .def_readwrite("keys", &KVCache::keys)
        .def_readwrite("values", &KVCache::values)
        .def_readwrite("current_length", &KVCache::current_length);
    
    // Add KV cache functions
    m.def("initialize_kv_cache", &initialize_kv_cache, "Initialize a new KV cache",
          py::arg("batch_size"), py::arg("num_heads"), py::arg("max_seq_len"), 
          py::arg("head_dim"), py::arg("options"));
    
    m.def("update_kv_cache", &update_kv_cache, "Update the KV cache with new keys and values",
          py::arg("cache"), py::arg("new_keys"), py::arg("new_values"));
    
    m.def("incremental_forward", &flash_attention_incremental_forward, 
          "Optimized incremental forward pass using KV cache",
          py::arg("query"), py::arg("kv_cache"), 
          py::arg("mask") = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);

    m.def("single_token_forward", &flash_attention_single_token_forward, 
          "Optimized single-token forward pass using KV cache",
          py::arg("query"), py::arg("kv_cache"), 
          py::arg("mask") = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);

    m.def("sliding_window_forward", &flash_attention_sliding_window_forward, 
          "Optimized sliding window forward pass using KV cache",
          py::arg("query"), py::arg("kv_cache"), 
          py::arg("mask") = py::none(),
          py::arg("window_size") = 1024,
          py::arg("softmax_scale") = 1.0f,
          py::arg("causal") = false);

    m.def("reorder_kv_cache", &reorder_kv_cache, "Reorders KV cache based on beam indices",
          py::arg("cache"), py::arg("beam_idx"));
} 
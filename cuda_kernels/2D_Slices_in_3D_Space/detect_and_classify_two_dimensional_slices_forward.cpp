template <typename scalar_t>
__global__ void _3d_flash_attention_backward_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* o,
    const scalar_t* l,
    const scalar_t* m,
    const scalar_t* dout,
    const scalar_t* dclass_logits,
    const scalar_t* dbounding_boxes,
    const float* positions,
    scalar_t* dq,
    scalar_t* dk,
    scalar_t* dv,
    float* dpositions,
    int batch_size,
    int num_heads,
    int num_slices,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    int num_classes,
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
    __shared__ float s_pos[1];
    
    int batch_idx = bidx / (num_heads * num_slices * seq_len_q);
    int head_idx = (bidx / (num_slices * seq_len_q)) % num_heads;
    int slice_idx = (bidx / seq_len_q) % num_slices;
    int seq_idx = bidx % seq_len_q;
    
    scalar_t thread_sum = 0;

    // Load position into shared memory
    if (tidx == 0) {
        s_pos[0] = positions[batch_idx * num_slices + slice_idx];
    }
    __syncthreads();

    // Main loop over tiles
    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, seq_len_kv);
        
        // Load K, V, and dO into shared memory
        for (int i = tidx; i < (tile_end - tile_start) * max(head_dim_k, head_dim_v); i += blockDim.x) {
            int col = i % max(head_dim_k, head_dim_v);
            int row = i / max(head_dim_k, head_dim_v);
            if (col < head_dim_k) {
                s_k[row][col] = k[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_k + 
                                  head_idx * num_slices * seq_len_kv * head_dim_k +
                                  slice_idx * seq_len_kv * head_dim_k +
                                  (tile_start + row) * head_dim_k + col];
            }
            if (col < head_dim_v) {
                s_v[row][col] = v[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_v + 
                                  head_idx * num_slices * seq_len_kv * head_dim_v +
                                  slice_idx * seq_len_kv * head_dim_v +
                                  (tile_start + row) * head_dim_v + col];
            }
        }
        for (int i = tidx; i < head_dim_v; i += blockDim.x) {
            s_do[0][i] = dout[batch_idx * num_heads * num_slices * seq_len_q * head_dim_v + 
                              head_idx * num_slices * seq_len_q * head_dim_v +
                              slice_idx * seq_len_q * head_dim_v +
                              seq_idx * head_dim_v + i];
        }
        __syncthreads();
        
        // Compute gradients
        for (int i = 0; i < tile_end - tile_start; ++i) {
            if (causal && tile_start + i > seq_idx) break;
            
            scalar_t qk_sum = 0;
            #pragma unroll
            for (int j = 0; j < min(head_dim_q, head_dim_k); ++j) {
                qk_sum += q[batch_idx * num_heads * num_slices * seq_len_q * head_dim_q + 
                            head_idx * num_slices * seq_len_q * head_dim_q +
                            slice_idx * seq_len_q * head_dim_q +
                            seq_idx * head_dim_q + j] * s_k[i][j];
            }
            qk_sum *= softmax_scale * (1 + s_pos[0]);  // Include position in attention
            
            scalar_t attention_weight = __expf(qk_sum - m[batch_idx * num_heads * num_slices * seq_len_q + 
                                                         head_idx * num_slices * seq_len_q +
                                                         slice_idx * seq_len_q + seq_idx]) / 
                                        l[batch_idx * num_heads * num_slices * seq_len_q + 
                                          head_idx * num_slices * seq_len_q +
                                          slice_idx * seq_len_q + seq_idx];
            
            // Gradient w.r.t. q
            #pragma unroll
            for (int j = 0; j < head_dim_q; ++j) {
                scalar_t dq_sum = 0;
                for (int k = 0; k < head_dim_v; ++k) {
                    dq_sum += s_do[0][k] * (s_v[i][k] - o[batch_idx * num_heads * num_slices * seq_len_q * head_dim_v + 
                                                          head_idx * num_slices * seq_len_q * head_dim_v +
                                                          slice_idx * seq_len_q * head_dim_v +
                                                          seq_idx * head_dim_v + k]);
                }
                atomicAdd(&dq[batch_idx * num_heads * num_slices * seq_len_q * head_dim_q + 
                              head_idx * num_slices * seq_len_q * head_dim_q +
                              slice_idx * seq_len_q * head_dim_q +
                              seq_idx * head_dim_q + j],
                          softmax_scale * (1 + s_pos[0]) * attention_weight * s_k[i][j] * dq_sum);
            }
            
            // Gradient w.r.t. k
            #pragma unroll
            for (int j = 0; j < head_dim_k; ++j) {
                scalar_t dk_sum = 0;
                for (int k = 0; k < head_dim_v; ++k) {
                    dk_sum += s_do[0][k] * (s_v[i][k] - o[batch_idx * num_heads * num_slices * seq_len_q * head_dim_v + 
                                                          head_idx * num_slices * seq_len_q * head_dim_v +
                                                          slice_idx * seq_len_q * head_dim_v +
                                                          seq_idx * head_dim_v + k]);
                }
                atomicAdd(&dk[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_k + 
                              head_idx * num_slices * seq_len_kv * head_dim_k +
                              slice_idx * seq_len_kv * head_dim_k +
                              (tile_start + i) * head_dim_k + j],
                          softmax_scale * (1 + s_pos[0]) * attention_weight * 
                          q[batch_idx * num_heads * num_slices * seq_len_q * head_dim_q + 
                            head_idx * num_slices * seq_len_q * head_dim_q +
                            slice_idx * seq_len_q * head_dim_q +
                            seq_idx * head_dim_q + j] * dk_sum);
            }
            
            // Gradient w.r.t. v
            #pragma unroll
            for (int j = 0; j < head_dim_v; ++j) {
                atomicAdd(&dv[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_v + 
                              head_idx * num_slices * seq_len_kv * head_dim_v +
                              slice_idx * seq_len_kv * head_dim_v +
                              (tile_start + i) * head_dim_v + j],
                          attention_weight * s_do[0][j]);
            }

            // Gradient w.r.t. position
            scalar_t dpos_sum = 0;
            for (int k = 0; k < head_dim_v; ++k) {
                dpos_sum += s_do[0][k] * (s_v[i][k] - o[batch_idx * num_heads * num_slices * seq_len_q * head_dim_v + 
                                                        head_idx * num_slices * seq_len_q * head_dim_v +
                                                        slice_idx * seq_len_q * head_dim_v +
                                                        seq_idx * head_dim_v + k]);
            }
            atomicAdd(&dpositions[batch_idx * num_slices + slice_idx],
                      softmax_scale * attention_weight * qk_sum * dpos_sum);
        }
        __syncthreads();
    }

    // Gradient w.r.t. class logits and bounding boxes
    if (tidx == 0) {
        for (int c = 0; c < num_classes; ++c) {
            scalar_t dclass_sum = dclass_logits[batch_idx * num_classes + c];
            for (int j = 0; j < head_dim_v; ++j) {
                atomicAdd(&dv[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_v + 
                              head_idx * num_slices * seq_len_kv * head_dim_v +
                              slice_idx * seq_len_kv * head_dim_v +
                              seq_idx * head_dim_v + j],
                          dclass_sum * (c + 1));
            }
        }

        for (int b = 0; b < 6; ++b) {
            scalar_t dbox_sum = dbounding_boxes[batch_idx * 6 + b];
            for (int j = 0; j < head_dim_v; ++j) {
                atomicAdd(&dv[batch_idx * num_heads * num_slices * seq_len_kv * head_dim_v + 
                              head_idx * num_slices * seq_len_kv * head_dim_v +
                              slice_idx * seq_len_kv * head_dim_v +
                              seq_idx * head_dim_v + j],
                          dbox_sum * (b + 1));
            }
        }
    }
}


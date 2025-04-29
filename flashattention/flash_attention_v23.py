import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple, List, Dict, Any, Union
import math

from . import flash_attention_partial_v23 as flash_attn_cuda

class KVCache:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

def reorder_kv_cache(kv_cache: KVCache, beam_idx: torch.Tensor) -> None:
    """
    Reorders the KV cache based on beam indices for beam search.
    
    Args:
        kv_cache: The KV cache to reorder.
        beam_idx: LongTensor of shape [batch_size] containing indices of beams to select.
            For each position i, the value beam_idx[i] indicates which beam's cache
            to copy to position i.
    """
    assert beam_idx.dim() == 1, "beam_idx must be a 1D tensor"
    assert beam_idx.size(0) == kv_cache.keys.size(0), "beam_idx size must match batch dimension of KV cache"
    
    # Call the CUDA implementation
    flash_attn_cuda.reorder_kv_cache(kv_cache, beam_idx) 
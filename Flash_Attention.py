import torch
import torch.nn.functional as F
import time

# FLASH ATTENTION: It is similar to normal attention mechanism but in KV cache,during
# attention computation it has to work in a huge matrix where flash attention makes the gpu optimization by computing 
#the attention over smaller blocks which fits into the gpu tensor cores, updates the
#running softmax,discards tile and accumulate the final result

#So, it takes help of PyTorch scaled attention unit for effective gpu usage

device = "cuda"

torch.manual_seed(0)

seq_len = 512
d_model = 768
head_dim = 64

X = torch.randn(seq_len, d_model, device=device)

Wq = torch.randn(d_model, head_dim, device=device)
Wk = torch.randn(d_model, head_dim, device=device)
Wv = torch.randn(d_model, head_dim, device=device)

# ------------------------------------
# KV Cache + Flash Attention
# ------------------------------------

start=time.perf_counter()

k_cache=torch.empty(d_model,head_dim,device=device)
v_cache=torch.empty(d_model,head_dim,device=device)

for t in range(seq_len):
    q=X[t:t+1] @ Wq

    k_cache=X[t:t+1] @ Wk
    v_cache=X[t:t+1] @ Wv

    k=k_cache[:t+1]
    v=v_cache[:t+1]

    # reshape for SDPA
    q = q.unsqueeze(0).unsqueeze(0)
    k = k.unsqueeze(0).unsqueeze(0)
    v = v.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False
    )
    
flash_time = time.perf_counter() - start

print(f"KV Cache + Flash Attention: {flash_time:.4f}s")



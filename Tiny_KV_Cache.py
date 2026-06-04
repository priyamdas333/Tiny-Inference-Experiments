import torch
import math
import time

# --------------------------
#Q, K, V parameters where each one consists of qi,ki,vi of each token
# --------------------------

torch.manual_seed(0)
seq_len = 512
d_model = 768
d_k = 64

x=torch.randn(seq_len,d_model)

Wq = torch.randn(d_model,d_k)
Wk = torch.randn(d_model,d_k)
Wv = torch.randn(d_model,d_k)
# -------------------------------------------
# AUTO-REGRESSIVE PREDICTION WITHOUT KV CACHE
# -------------------------------------------
# ISSUE: Recomputation of Keys and Values of previous tokens w.r.t Query of current token

start=time.perf_counter()

for t in range(seq_len):
    q=x[t:t+1] @ Wq
    
    #Recompute all previous keys and values each time repeatedly
    k=x[:t+1] @ Wk
    v=x[:t+1] @ Wv

    score = q @ k.T
    attn = torch.softmax(score, dim=-1)
    output = attn @ v

end=time.perf_counter()
print("Naive time:", end-start)

# -----------------------------------------
#KV cache mechanism
# -----------------------------------------
start=time.perf_counter()

# Compute K,V exactly once only for the current token since all the previous 
# prefix's tokens are stored in KV cache

k_cache=torch.empty(seq_len,d_model)
v_cache=torch.empty(seq_len,d_model)

for t in range(seq_len):

    q= x[t:t+1] @ Wq

    k_cache[t] = x[t] @ Wk
    v_cache[t] = x[t] @ Wv

    k = k_cache[:t+1]
    v = v_cache[:t+1]

    scores= q @ k.T
    attn= torch.softmax(scores, dim=-1)
    output= attn @ v

kv_time = time.perf_counter() - start

print(f"Naive decoding : {naive_time:.4f} s")
print(f"KV cache       : {kv_time:.4f} s")
print(f"Speedup        : {naive_time / kv_time:.2f}x")




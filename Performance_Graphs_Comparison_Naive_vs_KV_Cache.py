import torch
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)

d_model = 512
d_k = 64

seq_lengths = [64, 128, 256, 512, 1024]

naive_times = []
kv_times = []

for seq_len in seq_lengths:

    X = torch.randn(seq_len, d_model)

    Wq = torch.randn(d_model, d_k)
    Wk = torch.randn(d_model, d_k)
    Wv = torch.randn(d_model, d_k)

    # --------------------
    # Naive
    # --------------------
    start = time.perf_counter()

    for t in range(seq_len):
        q = X[t:t+1] @ Wq
        k = X[:t+1] @ Wk
        v = X[:t+1] @ Wv

        scores = q @ k.T
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

    naive_times.append(time.perf_counter() - start)

    # --------------------
    # KV Cache
    # --------------------
    start = time.perf_counter()

    k_cache = torch.empty(seq_len, d_k)
    v_cache = torch.empty(seq_len, d_k)

    for t in range(seq_len):
        q = X[t:t+1] @ Wq

        k_cache[t] = X[t] @ Wk
        v_cache[t] = X[t] @ Wv

        scores = q @ k_cache[:t+1].T
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_cache[:t+1]

    kv_times.append(time.perf_counter() - start)

# --------------------
# Plot
# --------------------
plt.figure(figsize=(8, 5))
plt.plot(seq_lengths, naive_times, marker="o", label="Naive")
plt.plot(seq_lengths, kv_times, marker="o", label="KV Cache")
plt.xlabel("Sequence Length")
plt.ylabel("Time (seconds)")
plt.title("Naive vs KV Cache Decoding")
plt.legend()
plt.grid(True)
plt.show()
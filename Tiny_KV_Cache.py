import torch
import math
import time

# --------------------------
#Q, K, V parameters where each one consists of qi,ki,vi of each token
# --------------------------

torch.manual_seed(0)
seq_len=256
d_head=64

Q=torch.randn(seq_len,d_head)
K=torch.randn(seq_len,d_head)
V=torch.randn(seq_len,d_head)

# -------------------------------------------
# AUTO-REGRESSIVE PREDICTION WITHOUT KV CACHE
# -------------------------------------------

start=time.time()

for t in range(1,seq_len+1):
    q=Q[t-1:t]
    k=K[:t]
    v=V[:t]

    score = q @ k.T
    attn = torch.softmax(score, dim=-1)
    output = attn @ v

end=time.time()
print("Naive time:", end-start)




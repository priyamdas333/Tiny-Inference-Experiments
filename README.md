# Tiny - Inference -Experiments :

Inferencing is the process by which an LLM model provides response to the user's  prompt. Now, various inference optimization techniques are developed in order to reduce the inferencing time, reduce latency and enahnce user experience.

There are 2 different ways to improve inferencing: 1. RUNTIME LEVEL   2. INFRASTRUCTURE LEVEL

# RUNTIME
It is the process of optimizing the performance of a single model on a single GPU-backed instance. It is done at the software level. There are various software engineering techniques like KV caching, Batching, Quantization, Parallelism, Speculative Decoding etc are utilized.

There are different software stack utilizing this techniques to provide optimized inference support for various LLM models like VLLM,SGLang TensorRT-LLM etc

# INFRASTRUCTURE
It is the process of scaling across clusters,regions and clouds without creating silos while maintaining excellent uptime.

# Experiments Overview

This project explores two of the most important optimizations used in modern Large Language Models (LLMs):

KV Cache – avoids recomputing keys and values for previously generated tokens during autoregressive decoding.
Flash Attention – computes attention using memory-efficient GPU kernels that avoid materializing large attention matrices.

The goal is to understand how these techniques impact runtime and memory usage as sequence lengths grow.
Experiments

# 1. Naive Autoregressive Decoding

For every generated token, keys and values are recomputed for the entire sequence seen so far:
```
Step t:
K = X[:t+1] @ Wk
V = X[:t+1] @ Wv

```
This results in repeated computation and poor scalability.

# 2. KV Cache

Keys and values are computed only once and stored in a cache:

Step t:
Compute Kt, Vt
Store in cache
Attend over cached K,V

Benefits:

Eliminates redundant K/V projections
Significantly improves decoding speed
Standard optimization used in production LLMs

# 3. KV Cache + Flash Attention

Flash Attention uses PyTorch's:
```
torch.nn.functional.scaled_dot_product_attention()
```
which can dispatch to optimized GPU kernels.

Benefits:

Lower memory consumption
Better GPU utilization
Faster attention computation for long contexts
Used during prompt processing (prefill) and increasingly during decoding
Metrics Compared

The benchmark evaluates:

Runtime

Measures total decoding time across varying sequence lengths.

Expected trend:

Naive  >  KV Cache  >  KV Cache + Flash Attention
Peak GPU Memory

Measures maximum allocated GPU memory during execution.

Expected trend:

KV Cache  >  KV Cache + Flash Attention

Flash Attention reduces memory overhead by computing attention in blocks rather than storing large intermediate matrices.

Performance Graphs
Runtime vs Sequence Length

Shows how decoding latency scales with increasing context size.

Expected observations:

Naive decoding grows rapidly due to repeated K/V computation.
KV Cache substantially reduces runtime.
Flash Attention provides additional acceleration, especially for long contexts.

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/6b160342-29cf-4235-a02d-572010dfd9a2" />






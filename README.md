# Tiny - Inference -Experiments :

Inferencing is the process by which an LLM model provides response to the user's  prompt. Now, various inference optimization techniques are developed in order to reduce the inferencing time, reduce latency and enahnce user experience.

There are 2 different ways to improve inferencing: 1. RUNTIME LEVEL   2. INFRASTRUCTURE LEVEL

# RUNTIME
It is the process of optimizing the performance of a single model on a single GPU-backed instance. It is done at the software level. There are various software engineering techniques like KV caching, Batching, Quantization, Parallelism, Speculative Decoding etc are utilized.

There are different software stack utilizing this techniques to provide optimized inference support for various LLM models like VLLM,SGLang TensorRT-LLM etc

# INFRASTRUCTURE
It is the process of scaling across clusters,regions and clouds without creating silos while maintaining excellent uptime.



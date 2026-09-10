# Week 01 — Introduction to High-Performance Machine Learning


- **Why HPML matters**: Modern AI models need enormous amounts of computation. One GPU could take years, so we need many processors working in parallel.
- **Memory is not equally fast**: Data in cache is very fast to access, while data in DRAM is much slower. Sometimes moving data is more expensive than doing the calculation.
- **The roofline**: Performance is limited by either compute speed or memory bandwidth.
  * Low arithmetic intensity → memory-bound
  * High arithmetic intensity → compute-bound
- **Arithmetic intensity**: `Intensity=#arithmetic operations/ loaded data`, tells us how much useful computation we get from each byte loaded from memory.
- **CPU vs GPU vs TPU**: 
  * CPU: good for general and low-latency tasks
  * GPU: good for massive parallel computation
  * TPU: specialized for matrix operations used in machine learning
- **NumPy → JAX**: immutability (what makes `jit`/`grad`/`vmap`/
  sharding composable), float32 by default, async dispatch and explicit PRNG keys.
- **`jit`**: combine several small operations into one larger operation. This is called fusion.
Fusion reduces memory traffic. Instead of reading and writing the same data many times, the program can load it once, perform several calculations, and then store the result.
- **Main idea**: High-performance machine learning is about keeping the hardware busy by balancing *computation*, *memory access*, and *parallelism*.

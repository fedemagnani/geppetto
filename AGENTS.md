# Geppetto

Geppetto is a training and inference framework for large language models. It starts as an educational project, designed for commodity hardware and tailored on author's macbook with the following specs (details pulled via `system_profiler SPHardwareDataType SPDisplaysDataType SPMemoryDataType`): 
```text
Hardware Overview:

    Model Name: MacBook Pro
    Model Identifier: Mac16,7
    Chip: Apple M4 Pro
    Total Number of Cores: 14 (10 performance and 4 efficiency)
    Memory: 48 GB

Graphics/Displays:

    Apple M4 Pro:

        Chipset Model: Apple M4 Pro
        Type: GPU
        Bus: Built-In
        Total Number of Cores: 20
        Vendor: Apple (0x106b)
        Metal Support: Metal 4

Memory:

    Memory: 48 GB
    Type: LPDDR5
    Manufacturer: Hynix
```


## Goal
The current goal of this project is to define the machinery to train and run inference on [GPT-2 model](https://huggingface.co/openai-community/gpt2). This would require crafting from scratch the following components:
- **Tokenizer** (bpe)
- **Metal GPU kernels**
- **Activation functions**
- **Tensor module**
- **The attention mechanism**
- **Optimizer** (AdamW)
- **Backpropagation**
- **Autograd**
- **Layers**: Linear, Normalization, Dropout
- **Transformer block** (stack of layers)


## Benchmarking
In order to measure performance and consistency, it is strongly advised designing primitives which can be benchmarked against:
- [candle](https://github.com/huggingface/candle)
- [burn](https://github.com/tracel-ai/burn-lm) 
  - For benchmarking methodology, it is advisable to refer to [burn-bench](https://github.com/tracel-ai/burn-bench)
- [tiktoken_rs](https://github.com/zurawiki/tiktoken-rs) 

To assess correctness, you could try using a distance metric (e.g., L2 norm) to compare the output of your implementation with the output of a reference implementation 
we should be very strong in benchmarking and profiling using `criterion`, `divan` and `hotpath`


### Tensor module
Desiderata for the tensor primitve are all around mechanical sympathy and developed around the following principles:
- flat array with strides (no nested vectors): huge benefit in prefetching and cache locality
- lock-free design: no mutex, no rwlock, possibly no atomic operations (which migh cause cache-line boucing)
  - an alternative design is baseed on arena allocation, where you pre-allocate a big chunk of memory and then you manage it with a custom allocator, which is lock-free because each thread will operate only on its own chunk of memory. However, recycling memory from the arena might be tricky, because you need to make sure that no thread is still using the chunk of memory that you want to recycle. 
  - apparently metal itself provides an arena memory model (managed by metal itself) called `MTLHeap`: in rust you simply claim handles to this pre-allocated memory. The more savage approach instead is to use `bytesNoCopy` but this might be more error prone as you need to handle a bunch of stuff such as page-alignment etc.
- inplace operataions: to minimize heap allocations
- prefer lazy over eager evalations: fuse operations together which might eliminate intermediate allocations. Operations might be chained in an iterator and then evaluated only when required. I think that this is what `rug` does with the `complete()` call when operating on references


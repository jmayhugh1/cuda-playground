# CUDA Playground

This project is my CUDA experimentation space.

I am replicating a bunch of classic algorithms in CUDA, then benchmarking CPU vs GPU implementations to better understand:

- kernel design and launch configuration choices
- memory transfer overhead and synchronization costs
- when GPU acceleration actually provides speedup
- tradeoffs between correctness, simplicity, and performance


## Build

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

## Run

```bash
./build/vector_add_bench
```

## Notes

This repo is intentionally iterative and experimental. I expect to add more algorithms over time and track how different CUDA optimization techniques affect benchmark results.

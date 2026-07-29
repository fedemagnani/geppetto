check:
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test

# Model used by `run`, overridable per-invocation: `just model=path/to.gguf run ...`.
model := "models/gpt2-f16.gguf"

# Generate text with a GPT-2 GGUF model. Release build: a debug matmul is
# unusably slow. Every argument passes straight through to the binary, e.g.
# `just run -p "Once upon a time" -n 40 --temp 0.7`. Override the model with
# `just model=models/other.gguf run -p "..."`.
# positional-arguments + "$@" keeps a quoted multi-word prompt as one argument.
[positional-arguments]
run *args:
    cargo run --release --bin generate --features hotpath,hotpath-alloc -- {{model}} "$@"

# Per-function timing and allocation attribution. Perturbs the timings, so
# never record these as KPIs.
bench:
    cargo run --release --features hotpath,hotpath-alloc --bin bench -- {{model}}

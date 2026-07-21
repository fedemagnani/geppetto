check:
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test

# Per-function timing and allocation attribution. Perturbs the timings, so
# never record these as KPIs.
bench model="models/gpt2-f16.gguf":
    cargo run --release --features hotpath,hotpath-alloc --bin bench -- {{model}}

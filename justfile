check:
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test

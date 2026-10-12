// Accelerate backs the dense kernels on macOS when the `accelerate` feature is on.
fn main() {
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if macos && std::env::var_os("CARGO_FEATURE_ACCELERATE").is_some() {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

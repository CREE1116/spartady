use std::path::PathBuf;
fn main() {
    println!("cargo::rustc-check-cfg=cfg(loom_lmx_engine)");
    println!("cargo:rerun-if-changed=native/lmx/engine.cpp");
    let cmake = PathBuf::from(std::env::var_os("DEP_LLAMA_GGML_CMAKE_DIR").expect("ggml build path required"));
    let include = cmake.join("../../include");
    assert!(include.join("ggml-backend.h").is_file(), "ggml headers required");
    cc::Build::new().cpp(true).std("c++17").include(include)
        .file("native/lmx/engine.cpp").opt_level(2).warnings(false).compile("spartady_lmx_engine");
    println!("cargo:rustc-cfg=loom_lmx_engine");
}

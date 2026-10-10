pub mod lmx_native;
pub mod lmx_native_file;
pub mod lmx_engine;
pub fn initialize_backend() -> anyhow::Result<()> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    static BACKEND: std::sync::OnceLock<Result<LlamaBackend, String>> = std::sync::OnceLock::new();
    match BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init().map_err(|e| e.to_string())?;
        backend.void_logs();
        Ok(backend)
    }) {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow::anyhow!(e.clone())),
    }
}

pub mod lmx_native_import;
pub mod model_package;
pub mod local;

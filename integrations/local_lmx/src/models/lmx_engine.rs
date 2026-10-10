//! Native LMX v2 execution. The engine (`native/lmx/engine.cpp`) runs a validated native
//! file with ggml's kernels on whatever backend this build has (CPU, Metal, CUDA, Vulkan).
//! Resident weights are uploaded once; routed experts are paged from the file into a
//! bounded slot pool as the router picks them; the token embedding stays in the file.
//!
//! Supported recipes: recipe 2 `qwen35` / `qwen35_moe` (hybrid Gated DeltaNet + gated
//! attention, dense or MoE) and recipe 3 `lfm2` / `lfm2_moe` (short convolution + GQA,
//! dense or sigmoid-routed MoE with leading dense layers).
//! Storage must be dense or ggml blocks; affine storage (MLX imports) is not executable here.

use super::lmx_native::{
    Architecture, ConvPart, Dtype, FeedForward, Gating, LinearPart, Manifest, Projection, Role,
    Storage,
};
use super::lmx_native_file::NativeFile;
use anyhow::{Context, Result, bail, ensure};
use llama_cpp_sys_2 as sys;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawTensor {
    kind: i32,
    layer: i32,
    expert: i32,
    ty: i32,
    map: i32,
    pad: i32,
    ne: [i64; 4],
    offset: u64,
    bytes: u64,
}

#[repr(C)]
struct RawConfig {
    layers: u32,
    hidden: u32,
    vocab: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    rope_sections: [i32; 4],
    rope_theta: f32,
    rms_eps: f32,
    experts: u32,
    active: u32,
    expert_ff: u32,
    shared_ff: u32,
    conv_kernel: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    context: u32,
    chunk: u32,
    threads: u32,
    gpu: u32,
    train_context: u32,
    expert_budget: u64,
    prefetch: u32,
    kv_type: i32,
    kv_rotate: u32,
    pad: u32,
    /// Per layer: 0 gated attention, 1 Gated DeltaNet, 2 attention, 3 short convolution.
    mixer: *const u8,
    /// Per layer: 1 = dense FFN, 0 = routed experts.
    dense: *const u8,
    /// 0 = softmax router, 1 = sigmoid router (with optional selection bias).
    gating: u32,
    normalize_topk: u32,
    /// 0 = interleaved multi-section RoPE (Qwen3.5), 1 = split-half (NEOX) RoPE.
    rope_neox: u32,
    pad2: u32,
}

#[repr(C)]
struct RawStats {
    hits: u64,
    misses: u64,
    bytes_read: u64,
    resident_bytes: u64,
    pool_bytes: u64,
    state_bytes: u64,
    slots: u32,
    pools: u32,
    position: u32,
    chunk: u32,
    backend: [c_char; 64],
    build_us: f64,
    alloc_us: f64,
    compute_us: f64,
    route_us: f64,
    prefetched: u64,
    prefetch_hits: u64,
    budget: u64,
    all_experts: u64,
    warm_total: u64,
    warm_done: u64,
    released: u64,
    context: u32,
    pad2: u32,
    spec_tokens: u64,
    spec_restarts: u64,
}

impl Default for RawStats {
    fn default() -> Self {
        // Plain integers and a C string buffer: all-zero is a valid value.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(loom_lmx_engine)]
unsafe extern "C" {
    fn loom_lmx_open(
        path: *const c_char,
        cfg: *const RawConfig,
        tensors: *const RawTensor,
        count: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_void;
    fn loom_lmx_eval(
        engine: *mut c_void,
        tokens: *const i32,
        count: usize,
        logits: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn loom_lmx_reset(engine: *mut c_void);
    fn loom_lmx_checkpoint(engine: *mut c_void, err: *mut c_char, err_len: usize) -> i32;
    fn loom_lmx_rollback(engine: *mut c_void, id: i32, err: *mut c_char, err_len: usize) -> i32;
    fn loom_lmx_forget(engine: *mut c_void, id: i32);
    fn loom_lmx_save_state(engine: *mut c_void, path: *const c_char, err: *mut c_char, err_len: usize) -> i32;
    fn loom_lmx_load_state(engine: *mut c_void, path: *const c_char, err: *mut c_char, err_len: usize) -> i32;
    fn loom_lmx_stats(engine: *mut c_void, out: *mut RawStats);
    fn loom_lmx_free(engine: *mut c_void);
}

// Mirrors `LoomLmxKind` in engine.cpp.
const EMBEDDING: i32 = 0;
const OUTPUT: i32 = 1;
const OUTPUT_NORM: i32 = 2;
const ATTN_NORM: i32 = 3;
const POST_NORM: i32 = 4;
const Q: i32 = 5;
const K: i32 = 6;
const V: i32 = 7;
const O: i32 = 8;
const Q_NORM: i32 = 9;
const K_NORM: i32 = 10;
const QKV: i32 = 11;
const Z: i32 = 12;
const BETA: i32 = 13;
const ALPHA: i32 = 14;
const CONV: i32 = 15;
const DT_BIAS: i32 = 16;
const A_LOG: i32 = 17;
const SSM_NORM: i32 = 18;
const SSM_OUT: i32 = 19;
const ROUTER: i32 = 20;
const SHARED_GATE: i32 = 21;
// 22, 23: shared expert up, down (SHARED_GATE + 1, + 2)
const SHARED_ROUTER: i32 = 24;
const EXPERT_GATE: i32 = 25;
const SC_IN: i32 = 28;
const SC_CONV: i32 = 29;
const SC_OUT: i32 = 30;
const EXPERT_BIAS: i32 = 31;
const FFN_GATE: i32 = 32;

const MAP_NONE: i32 = 0;
const MAP_PLUS_ONE: i32 = 1;
const MAP_NEG_EXP: i32 = 2;

/// Where and how much the engine may hold.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// KV positions for full-attention layers; 0 = decide from RAM.
    pub context: u32,
    /// Bytes for the routed-expert slot pools; 0 = decide from free system/GPU memory.
    pub expert_budget: u64,
    /// Max tokens per forward pass.
    pub chunk: u32,
    /// CPU threads, 0 = half the cores.
    pub threads: u32,
    /// Use a GPU device when the build has one.
    pub gpu: bool,
    /// Experts per layer to predict and page in ahead while decoding, 0 = off.
    pub prefetch: u32,
    /// KV cache block type (`f16`, `q8_0`, `q4_0`, ...).
    pub kv: String,
    /// Rotate K/V/Q per head before caching (TurboQuant-style); cheap, helps 4-bit KV.
    pub kv_rotate: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(default)
        };
        Self {
            context: mb("SPARTADY_LMX_CONTEXT", 0) as u32,
            expert_budget: mb("SPARTADY_LMX_EXPERT_MB", 0) << 20,
            chunk: mb("SPARTADY_LMX_CHUNK", 1024) as u32,
            threads: mb("SPARTADY_LMX_THREADS", 0) as u32,
            gpu: std::env::var("SPARTADY_LMX_GPU").as_deref() != Ok("0"),
            prefetch: mb("SPARTADY_LMX_PREFETCH", 8) as u32,
            kv: std::env::var("SPARTADY_LMX_KV").unwrap_or_else(|_| "q8_0".into()),
            kv_rotate: std::env::var("SPARTADY_LMX_KV_ROTATE").as_deref() != Ok("0"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub backend: String,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
    pub pool_bytes: u64,
    pub state_bytes: u64,
    pub slots: u32,
    pub pools: u32,
    pub position: u32,
    pub chunk: u32,
    /// Cumulative milliseconds: graph build, graph allocation, compute, expert routing.
    pub build_ms: f64,
    pub alloc_ms: f64,
    pub compute_ms: f64,
    pub route_ms: f64,
    /// Experts paged in ahead on a prediction, and how many of them were then routed to.
    pub prefetched: u64,
    pub prefetch_hits: u64,
    /// Expert slot bytes the engine chose (or was given), and the size of every expert.
    pub budget: u64,
    pub all_experts: u64,
    /// Experts the load-time warm-up fills, and how many it has.
    pub warm_total: u64,
    pub warm_done: u64,
    /// Slot bytes currently handed back to the OS because memory ran low.
    pub released: u64,
    /// KV positions the engine allocated.
    pub context: u32,
    /// Tokens decoded in one whole-model pass, and resumptions after a missing expert.
    pub spec_tokens: u64,
    pub spec_restarts: u64,
}

/// The ggml type id of a ggml type name.
fn ggml_type(name: &str) -> Result<i32> {
    (0..sys::GGML_TYPE_COUNT)
        .find(|t| {
            let p = unsafe { sys::ggml_type_name(*t) };
            !p.is_null() && unsafe { CStr::from_ptr(p) }.to_str() == Ok(name)
        })
        .map(|t| t as i32)
        .with_context(|| format!("this build's ggml has no type {name}"))
}

fn kind_of(role: &Role) -> Result<(i32, i32, i32)> {
    let mlp = |p: Projection, base: i32| -> Result<i32> {
        Ok(base
            + match p {
                Projection::Gate => 0,
                Projection::Up => 1,
                Projection::Down => 2,
                other => bail!("{other:?} is not an FFN projection"),
            })
    };
    Ok(match *role {
        Role::Embedding => (EMBEDDING, -1, -1),
        Role::Head => (OUTPUT, -1, -1),
        Role::FinalNorm => (OUTPUT_NORM, -1, -1),
        Role::AttentionNorm { layer } => (ATTN_NORM, layer as i32, -1),
        Role::FeedForwardNorm { layer } => (POST_NORM, layer as i32, -1),
        Role::Attention { layer, projection } => (
            match projection {
                Projection::Query => Q,
                Projection::Key => K,
                Projection::Value => V,
                Projection::Output => O,
                other => bail!("{other:?} is not an attention projection"),
            },
            layer as i32,
            -1,
        ),
        Role::QueryNorm { layer } => (Q_NORM, layer as i32, -1),
        Role::KeyNorm { layer } => (K_NORM, layer as i32, -1),
        Role::LinearAttention { layer, part } => (
            match part {
                LinearPart::Qkv => QKV,
                LinearPart::Z => Z,
                LinearPart::B => BETA,
                LinearPart::A => ALPHA,
                LinearPart::Conv => CONV,
                LinearPart::DtBias => DT_BIAS,
                LinearPart::ALog => A_LOG,
                LinearPart::Norm => SSM_NORM,
                LinearPart::Output => SSM_OUT,
            },
            layer as i32,
            -1,
        ),
        Role::Router { layer } => (ROUTER, layer as i32, -1),
        Role::SharedExpert { layer, projection } => (mlp(projection, SHARED_GATE)?, layer as i32, -1),
        Role::SharedExpertGate { layer } => (SHARED_ROUTER, layer as i32, -1),
        Role::Expert { layer, expert, projection } => {
            (mlp(projection, EXPERT_GATE)?, layer as i32, expert as i32)
        }
        Role::FeedForward { layer, projection } => (mlp(projection, FFN_GATE)?, layer as i32, -1),
        Role::ShortConv { layer, part } => (
            match part {
                ConvPart::InProj => SC_IN,
                ConvPart::Conv => SC_CONV,
                ConvPart::OutProj => SC_OUT,
            },
            layer as i32,
            -1,
        ),
        Role::ExpertBias { layer } => (EXPERT_BIAS, layer as i32, -1),
        Role::AttentionBias { .. } => {
            bail!("{} is not part of a recipe this engine runs yet", role.hf_name())
        }
    })
}

/// Executed form of a stored value (recipe 2 keeps HF values; LFM2 norms are plain).
fn map_of(architecture: Architecture, role: &Role) -> i32 {
    if !architecture.is_qwen35() {
        return MAP_NONE;
    }
    match role {
        Role::AttentionNorm { .. }
        | Role::FeedForwardNorm { .. }
        | Role::FinalNorm
        | Role::QueryNorm { .. }
        | Role::KeyNorm { .. } => MAP_PLUS_ONE,
        Role::LinearAttention { part: LinearPart::ALog, .. } => MAP_NEG_EXP,
        _ => MAP_NONE,
    }
}

/// Architectures the engine executes.
pub fn runnable(architecture: Architecture) -> bool {
    matches!(
        architecture,
        Architecture::Qwen35 | Architecture::Qwen35Moe | Architecture::Lfm2 | Architecture::Lfm2Moe
    )
}

fn describe(m: &Manifest) -> Result<Vec<RawTensor>> {
    let mut out = Vec::with_capacity(m.tensors.len());
    for (name, t) in &m.tensors {
        let (kind, layer, expert) = kind_of(&t.role)?;
        let (ty, buffer) = match &t.storage {
            Storage::Dense { values } => (
                match values.dtype {
                    Dtype::F32 => ggml_type("f32")?,
                    Dtype::F16 => ggml_type("f16")?,
                    Dtype::Bf16 => ggml_type("bf16")?,
                },
                &values.data,
            ),
            Storage::Ggml { format, data, .. } => (ggml_type(format.name())?, data),
            Storage::Affine { .. } => {
                bail!("{name}: affine storage (MLX import) is not executable by the ggml engine yet")
            }
        };
        ensure!(t.shape.len() <= 4 && !t.shape.is_empty(), "{name}: rank {} unsupported", t.shape.len());
        let mut ne = [1i64; 4];
        for (i, d) in t.shape.iter().rev().enumerate() {
            ne[i] = i64::try_from(*d)?;
        }
        let block = &m.blocks[&buffer.block];
        let map = map_of(m.model.architecture, &t.role);
        ensure!(
            map == MAP_NONE || matches!(&t.storage, Storage::Dense { values } if values.dtype == Dtype::F32),
            "{name}: expected F32 values"
        );
        out.push(RawTensor {
            kind,
            layer,
            expert,
            ty,
            map,
            pad: 0,
            ne,
            offset: block.offset + buffer.offset,
            bytes: buffer.bytes,
        });
    }
    Ok(out)
}

pub struct Engine {
    #[cfg(loom_lmx_engine)]
    raw: *mut c_void,
    pub vocab: usize,
    pub eog: Vec<u32>,
    pub identity: String,
}

// The engine is used from one thread at a time; it owns its buffers and file handle.
unsafe impl Send for Engine {}

fn error_text(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

impl Engine {
    pub fn available() -> bool {
        cfg!(loom_lmx_engine)
    }

    pub fn open(path: &Path, options: &EngineOptions) -> Result<Self> {
        let file = NativeFile::open(path)?;
        let m = &file.manifest;
        ensure!(
            runnable(m.model.architecture),
            "the native engine does not run {:?} yet",
            m.model.architecture
        );
        let layers = m.model.layers as usize;
        let a = &m.model.attention;
        let (experts, active, expert_ff, normalize_topk) = match m.model.feed_forward {
            FeedForward::Moe { intermediate, experts, active, normalize_topk } => {
                (experts, active, intermediate, normalize_topk)
            }
            FeedForward::Dense { .. } => (0, 0, 0, true),
        };
        let routing = m.model.routing.as_ref();
        let mut dense = vec![u8::from(experts == 0); layers];
        if let Some(r) = routing {
            for d in dense.iter_mut().take(r.dense_layers as usize) {
                *d = 1;
            }
        }
        let mut mixer = vec![if m.model.architecture.is_lfm2() { 2u8 } else { 0 }; layers];
        let mut cfg_hybrid = None;
        if let Some(hybrid) = &m.model.hybrid {
            for &l in &hybrid.linear_layers {
                *mixer.get_mut(l as usize).context("linear layer out of range")? = 1;
            }
            cfg_hybrid = Some(hybrid);
        }
        let mut conv_kernel = cfg_hybrid.map_or(0, |h| h.conv_kernel);
        if let Some(sc) = &m.model.shortconv {
            for &l in &sc.conv_layers {
                *mixer.get_mut(l as usize).context("conv layer out of range")? = 3;
            }
            conv_kernel = sc.kernel;
        }
        let tensors = describe(m)?;
        let cfg = RawConfig {
            layers: m.model.layers,
            hidden: m.model.hidden,
            vocab: m.model.vocab,
            heads: a.heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            rope_dims: cfg_hybrid.map_or(a.head_dim, |h| h.rope_dims),
            rope_sections: cfg_hybrid.map_or([0; 4], |h| h.rope_sections.map(|s| s as i32)),
            rope_theta: a.rope_theta as f32,
            rms_eps: m.model.rms_epsilon as f32,
            experts,
            active,
            expert_ff,
            shared_ff: cfg_hybrid.map_or(0, |h| h.shared_intermediate),
            conv_kernel,
            key_heads: cfg_hybrid.map_or(0, |h| h.key_heads),
            value_heads: cfg_hybrid.map_or(0, |h| h.value_heads),
            key_dim: cfg_hybrid.map_or(0, |h| h.key_dim),
            value_dim: cfg_hybrid.map_or(0, |h| h.value_dim),
            context: if options.context == 0 { 0 } else { options.context.min(m.model.context) },
            chunk: options.chunk.max(1),
            threads: options.threads,
            gpu: options.gpu as u32,
            train_context: m.model.context,
            expert_budget: options.expert_budget,
            prefetch: options.prefetch,
            kv_type: ggml_type(&options.kv)?,
            kv_rotate: options.kv_rotate as u32,
            pad: 0,
            mixer: mixer.as_ptr(),
            dense: dense.as_ptr(),
            gating: u32::from(routing.is_some_and(|r| r.gating == Gating::Sigmoid)),
            normalize_topk: u32::from(normalize_topk),
            rope_neox: u32::from(m.model.architecture.is_lfm2()),
            pad2: 0,
        };
        let vocab = m.model.vocab as usize;
        let eog = m.tokenizer.eog.clone();
        let identity = file.identity.clone();
        drop(file);
        Self::start(path, &cfg, &tensors, vocab, eog, identity)
    }

    #[cfg(loom_lmx_engine)]
    fn start(
        path: &Path,
        cfg: &RawConfig,
        tensors: &[RawTensor],
        vocab: usize,
        eog: Vec<u32>,
        identity: String,
    ) -> Result<Self> {
        // Registers the backends of dynamic-backend builds.
        super::initialize_backend()?;
        let path = CString::new(path.to_str().context("non-UTF8 model path")?)?;
        let mut err = [0 as c_char; 512];
        let raw = unsafe {
            loom_lmx_open(path.as_ptr(), cfg, tensors.as_ptr(), tensors.len(), err.as_mut_ptr(), err.len())
        };
        ensure!(!raw.is_null(), "native LMX engine: {}", error_text(&err));
        Ok(Self { raw, vocab, eog, identity })
    }

    #[cfg(not(loom_lmx_engine))]
    fn start(_: &Path, _: &RawConfig, _: &[RawTensor], _: usize, _: Vec<u32>, _: String) -> Result<Self> {
        bail!("this build has no native LMX engine")
    }

    /// Feeds `tokens` after the current position; returns the last token's logits.
    pub fn eval(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "no tokens to evaluate");
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut logits = vec![0f32; self.vocab];
        self.eval_into(&ids, &mut logits)?;
        Ok(logits)
    }

    #[cfg(loom_lmx_engine)]
    fn eval_into(&mut self, ids: &[i32], logits: &mut [f32]) -> Result<()> {
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            loom_lmx_eval(self.raw, ids.as_ptr(), ids.len(), logits.as_mut_ptr(), err.as_mut_ptr(), err.len())
        };
        ensure!(rc == 0, "native LMX engine: {}", error_text(&err));
        Ok(())
    }

    #[cfg(not(loom_lmx_engine))]
    fn eval_into(&mut self, _: &[i32], _: &mut [f32]) -> Result<()> {
        bail!("this build has no native LMX engine")
    }

    /// Forgets the sequence: KV cache, recurrent state and position. Expert slots stay warm.
    pub fn reset(&mut self) {
        #[cfg(loom_lmx_engine)]
        unsafe {
            loom_lmx_reset(self.raw)
        }
    }

    /// Saves the sequence state at the current position (recurrent state only; KV below
    /// the position stays valid until the caller rolls back past it).
    pub fn checkpoint(&mut self) -> Result<i32> {
        #[cfg(loom_lmx_engine)]
        {
            let mut err = [0 as c_char; 512];
            let id = unsafe { loom_lmx_checkpoint(self.raw, err.as_mut_ptr(), err.len()) };
            ensure!(id >= 0, "native LMX checkpoint: {}", error_text(&err));
            Ok(id)
        }
        #[cfg(not(loom_lmx_engine))]
        bail!("this build has no native LMX engine")
    }

    /// Returns to a checkpoint. Checkpoints made later than it are stale afterwards.
    pub fn rollback(&mut self, id: i32) -> Result<()> {
        #[cfg(loom_lmx_engine)]
        {
            let mut err = [0 as c_char; 512];
            let rc = unsafe { loom_lmx_rollback(self.raw, id, err.as_mut_ptr(), err.len()) };
            ensure!(rc == 0, "native LMX rollback: {}", error_text(&err));
            Ok(())
        }
        #[cfg(not(loom_lmx_engine))]
        {
            let _ = id;
            bail!("this build has no native LMX engine")
        }
    }

    pub fn forget(&mut self, id: i32) {
        #[cfg(loom_lmx_engine)]
        unsafe {
            loom_lmx_forget(self.raw, id)
        }
        #[cfg(not(loom_lmx_engine))]
        let _ = id;
    }

    /// Writes the sequence (KV below the position and recurrent state) to `path`.
    pub fn save_state(&mut self, path: &Path) -> Result<()> {
        #[cfg(loom_lmx_engine)]
        {
            let c = CString::new(path.to_str().context("non-UTF8 state path")?)?;
            let mut err = [0 as c_char; 512];
            let rc = unsafe { loom_lmx_save_state(self.raw, c.as_ptr(), err.as_mut_ptr(), err.len()) };
            ensure!(rc == 0, "native LMX state: {}", error_text(&err));
            Ok(())
        }
        #[cfg(not(loom_lmx_engine))]
        {
            let _ = path;
            bail!("this build has no native LMX engine")
        }
    }

    /// Replaces the sequence with a saved one. `Ok(false)` when the file is missing or was
    /// saved with other settings; the sequence is then empty. Checkpoints become invalid.
    pub fn load_state(&mut self, path: &Path) -> Result<bool> {
        #[cfg(loom_lmx_engine)]
        {
            let c = CString::new(path.to_str().context("non-UTF8 state path")?)?;
            let mut err = [0 as c_char; 512];
            let rc = unsafe { loom_lmx_load_state(self.raw, c.as_ptr(), err.as_mut_ptr(), err.len()) };
            ensure!(rc >= 0, "native LMX state: {}", error_text(&err));
            Ok(rc == 1)
        }
        #[cfg(not(loom_lmx_engine))]
        {
            let _ = path;
            bail!("this build has no native LMX engine")
        }
    }

    pub fn stats(&self) -> EngineStats {
        #[cfg(loom_lmx_engine)]
        {
            let mut s = RawStats::default();
            unsafe { loom_lmx_stats(self.raw, &mut s) };
            EngineStats {
                backend: error_text(&s.backend),
                hits: s.hits,
                misses: s.misses,
                bytes_read: s.bytes_read,
                resident_bytes: s.resident_bytes,
                pool_bytes: s.pool_bytes,
                state_bytes: s.state_bytes,
                slots: s.slots,
                pools: s.pools,
                position: s.position,
                chunk: s.chunk,
                build_ms: s.build_us / 1e3,
                alloc_ms: s.alloc_us / 1e3,
                compute_ms: s.compute_us / 1e3,
                route_ms: s.route_us / 1e3,
                prefetched: s.prefetched,
                prefetch_hits: s.prefetch_hits,
                budget: s.budget,
                all_experts: s.all_experts,
                warm_total: s.warm_total,
                warm_done: s.warm_done,
                released: s.released,
                context: s.context,
                spec_tokens: s.spec_tokens,
                spec_restarts: s.spec_restarts,
            }
        }
        #[cfg(not(loom_lmx_engine))]
        EngineStats::default()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        #[cfg(loom_lmx_engine)]
        unsafe {
            loom_lmx_free(self.raw)
        }
    }
}


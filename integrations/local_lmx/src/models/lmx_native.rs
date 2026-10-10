//! Native LMX v2 semantics. No GGUF, safetensors, model loader or device dependency.
//! Binary layout and streaming IO live in `lmx_native_file`.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const VERSION: u32 = 2;
pub const ALIGNMENT: u64 = 16_384;
pub const HEADER_BYTES: u64 = 128;
pub const METADATA_LIMIT: u64 = 32 << 20;
const MAX_TENSORS: usize = 100_000;

// Duplicate JSON keys must not silently replace a tensor or an authenticated block.
fn unique_map<'de, D, T>(de: D) -> std::result::Result<BTreeMap<String, T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique names")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = a.next_entry()? {
                if out.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate name"));
                }
            }
            Ok(out)
        }
    }
    de.deserialize_map(Visitor(std::marker::PhantomData))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub model: Model,
    #[serde(deserialize_with = "unique_map")]
    pub tensors: BTreeMap<String, Tensor>,
    #[serde(deserialize_with = "unique_map")]
    pub blocks: BTreeMap<String, Block>,
    pub tokenizer: Tokenizer,
    pub prompt: Prompt,
    pub provenance: Provenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    Qwen2,
    Qwen3,
    Qwen3Moe,
    /// Recipe 2: Qwen3.5/3.6 MoE, Gated DeltaNet layers between gated full attention.
    Qwen35Moe,
    /// Recipe 2 with a dense feed-forward network in every layer.
    Qwen35,
    /// Recipe 3: LFM2, gated short-convolution layers between GQA attention layers.
    Lfm2,
    /// Recipe 3 with MoE feed-forward layers after `routing.dense_layers` dense ones.
    Lfm2Moe,
}

impl Architecture {
    /// Gated DeltaNet hybrids (recipe 2).
    pub fn is_qwen35(self) -> bool {
        matches!(self, Self::Qwen35 | Self::Qwen35Moe)
    }
    /// Short-convolution hybrids (recipe 3).
    pub fn is_lfm2(self) -> bool {
        matches!(self, Self::Lfm2 | Self::Lfm2Moe)
    }
    pub fn is_moe(self) -> bool {
        matches!(self, Self::Qwen3Moe | Self::Qwen35Moe | Self::Lfm2Moe)
    }
    pub fn recipe(self) -> u32 {
        if self.is_qwen35() { 2 } else if self.is_lfm2() { 3 } else { 1 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub recipe_version: u32,
    pub architecture: Architecture,
    pub hidden: u32,
    pub layers: u32,
    pub vocab: u32,
    pub context: u32,
    pub attention: Attention,
    pub rms_epsilon: f64,
    pub feed_forward: FeedForward,
    pub tied_embeddings: bool,
    /// Recipe 2 only; absent from recipe 1 files so their identity is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<Hybrid>,
    /// Recipe 3 only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortconv: Option<ShortConv>,
    /// MoE routing beyond recipe 1's softmax top-k; absent means softmax, no bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Routing>,
}

/// Recipe 3 (LFM2). Layers in `conv_layers` replace attention with a gated short
/// convolution: `in_proj` yields `[b, c, x]`; `y = c · conv(b · x)` with a causal
/// depthwise kernel of `kernel` taps, then `out_proj`. Other layers use GQA attention with
/// Q/K RMS norms and split-half (NEOX) RoPE over the whole head. Norms are plain
/// RMS norms (`x̂·w`); the final norm is HF's `embedding_norm`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortConv {
    pub conv_layers: Vec<u32>,
    pub kernel: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gating {
    Softmax,
    Sigmoid,
}

/// How experts are chosen and weighted: probabilities from `gating`; selection adds the
/// per-expert bias when `expert_bias`; the chosen probabilities are renormalized when the
/// feed-forward says `normalize_topk`. The first `dense_layers` layers use a dense
/// feed-forward network of `dense_intermediate`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    pub gating: Gating,
    pub expert_bias: bool,
    pub dense_layers: u32,
    pub dense_intermediate: u32,
}

/// Short-convolution tensors of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvPart {
    InProj,
    Conv,
    OutProj,
}

/// Recipe 2 (Qwen3.5 MoE). Layers listed in `linear_layers` use Gated DeltaNet linear
/// attention; the others use full attention whose query projection also yields a
/// per-head sigmoid output gate (`[query, gate]` per head). RoPE is split-half over the
/// first `rope_dims` dimensions of each head; `rope_sections` are the multimodal
/// sections, all at the token position for text. RMS norms of layers, the final norm
/// and Q/K norms are zero-centered (`x̂·(1+w)`); the linear-attention output norm is a
/// plain gated RMS norm (`x̂·w·silu(z)`); `A_log` keeps HF's form (`A = −exp(A_log)`).
/// Every MoE layer adds a shared expert scaled by `sigmoid(shared_expert_gate·x)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hybrid {
    pub linear_layers: Vec<u32>,
    pub conv_kernel: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub rope_dims: u32,
    pub rope_sections: [u32; 4],
    pub shared_intermediate: u32,
}

/// Linear-attention (Gated DeltaNet) tensors of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearPart {
    /// Query, key and value channels in convolution order.
    Qkv,
    /// Output gate `z`.
    Z,
    /// Per-head write strength β (sigmoid).
    B,
    /// Per-head decay input (softplus with `dt_bias`, scaled by `A`).
    A,
    Conv,
    DtBias,
    ALog,
    Norm,
    Output,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attention {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    /// Recipe 1: causal attention, split-half RoPE on the entire head, no scaling.
    pub rope_theta: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeedForward {
    /// Recipe 1 activation is SiLU(gate) * up, followed by down.
    Dense { intermediate: u32 },
    /// All layers use softmax(router) then top-k; optional renormalization after top-k.
    Moe {
        intermediate: u32,
        experts: u32,
        active: u32,
        normalize_topk: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Role {
    Embedding,
    Head,
    FinalNorm,
    AttentionNorm {
        layer: u32,
    },
    FeedForwardNorm {
        layer: u32,
    },
    Attention {
        layer: u32,
        projection: Projection,
    },
    AttentionBias {
        layer: u32,
        projection: Projection,
    },
    QueryNorm {
        layer: u32,
    },
    KeyNorm {
        layer: u32,
    },
    FeedForward {
        layer: u32,
        projection: Projection,
    },
    Router {
        layer: u32,
    },
    Expert {
        layer: u32,
        expert: u32,
        projection: Projection,
    },
    LinearAttention {
        layer: u32,
        part: LinearPart,
    },
    SharedExpert {
        layer: u32,
        projection: Projection,
    },
    SharedExpertGate {
        layer: u32,
    },
    ShortConv {
        layer: u32,
        part: ConvPart,
    },
    ExpertBias {
        layer: u32,
    },
}

impl Role {
    /// The HF name of this role in a model of `architecture`.
    pub fn name_in(&self, architecture: Architecture) -> String {
        if !architecture.is_lfm2() {
            return self.hf_name();
        }
        let ffn = |p: Projection| match p {
            Projection::Gate => "w1",
            Projection::Up => "w3",
            _ => "w2",
        };
        match self {
            Self::FinalNorm => "model.embedding_norm.weight".into(),
            Self::AttentionNorm { layer } => format!("model.layers.{layer}.operator_norm.weight"),
            Self::FeedForwardNorm { layer } => format!("model.layers.{layer}.ffn_norm.weight"),
            Self::Attention { layer, projection: Projection::Output } => {
                format!("model.layers.{layer}.self_attn.out_proj.weight")
            }
            Self::QueryNorm { layer } => format!("model.layers.{layer}.self_attn.q_layernorm.weight"),
            Self::KeyNorm { layer } => format!("model.layers.{layer}.self_attn.k_layernorm.weight"),
            Self::FeedForward { layer, projection } => {
                format!("model.layers.{layer}.feed_forward.{}.weight", ffn(*projection))
            }
            Self::Router { layer } => format!("model.layers.{layer}.feed_forward.gate.weight"),
            Self::Expert { layer, expert, projection } => {
                format!("model.layers.{layer}.feed_forward.experts.{expert}.{}.weight", ffn(*projection))
            }
            Self::ExpertBias { layer } => format!("model.layers.{layer}.feed_forward.expert_bias"),
            _ => self.hf_name(),
        }
    }

    pub fn hf_name(&self) -> String {
        fn attention(p: Projection) -> &'static str {
            match p {
                Projection::Query => "q_proj",
                Projection::Key => "k_proj",
                Projection::Value => "v_proj",
                Projection::Output => "o_proj",
                _ => "invalid",
            }
        }
        fn mlp(p: Projection) -> &'static str {
            match p {
                Projection::Gate => "gate_proj",
                Projection::Up => "up_proj",
                Projection::Down => "down_proj",
                _ => "invalid",
            }
        }
        match self {
            Self::Embedding => "model.embed_tokens.weight".into(),
            Self::Head => "lm_head.weight".into(),
            Self::FinalNorm => "model.norm.weight".into(),
            Self::AttentionNorm { layer } => format!("model.layers.{layer}.input_layernorm.weight"),
            Self::FeedForwardNorm { layer } => {
                format!("model.layers.{layer}.post_attention_layernorm.weight")
            }
            Self::Attention { layer, projection } => format!(
                "model.layers.{layer}.self_attn.{}.weight",
                attention(*projection)
            ),
            Self::AttentionBias { layer, projection } => format!(
                "model.layers.{layer}.self_attn.{}.bias",
                attention(*projection)
            ),
            Self::QueryNorm { layer } => format!("model.layers.{layer}.self_attn.q_norm.weight"),
            Self::KeyNorm { layer } => format!("model.layers.{layer}.self_attn.k_norm.weight"),
            Self::FeedForward { layer, projection } => {
                format!("model.layers.{layer}.mlp.{}.weight", mlp(*projection))
            }
            Self::Router { layer } => format!("model.layers.{layer}.mlp.gate.weight"),
            Self::Expert {
                layer,
                expert,
                projection,
            } => format!(
                "model.layers.{layer}.mlp.experts.{expert}.{}.weight",
                mlp(*projection)
            ),
            Self::LinearAttention { layer, part } => format!(
                "model.layers.{layer}.linear_attn.{}",
                match part {
                    LinearPart::Qkv => "in_proj_qkv.weight",
                    LinearPart::Z => "in_proj_z.weight",
                    LinearPart::B => "in_proj_b.weight",
                    LinearPart::A => "in_proj_a.weight",
                    LinearPart::Conv => "conv1d.weight",
                    LinearPart::DtBias => "dt_bias",
                    LinearPart::ALog => "A_log",
                    LinearPart::Norm => "norm.weight",
                    LinearPart::Output => "out_proj.weight",
                }
            ),
            Self::SharedExpert { layer, projection } => format!(
                "model.layers.{layer}.mlp.shared_expert.{}.weight",
                mlp(*projection)
            ),
            Self::SharedExpertGate { layer } => {
                format!("model.layers.{layer}.mlp.shared_expert_gate.weight")
            }
            Self::ShortConv { layer, part } => format!(
                "model.layers.{layer}.conv.{}.weight",
                match part {
                    ConvPart::InProj => "in_proj",
                    ConvPart::Conv => "conv",
                    ConvPart::OutProj => "out_proj",
                }
            ),
            Self::ExpertBias { layer } => format!("model.layers.{layer}.mlp.expert_bias"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}
impl Dtype {
    pub fn bytes(self) -> u64 {
        if self == Self::F32 { 4 } else { 2 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub block: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedBuffer {
    pub dtype: Dtype,
    pub data: Buffer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Storage {
    /// Row-major, little-endian, exact logical rank; no implicit F32 -> F16 conversion.
    Dense { values: TypedBuffer },
    /// Recipe 1: value = unsigned_code * scale + bias (bias defaults to zero).
    /// Codes form a continuous little-endian bitstream within EACH row, byte padded.
    /// Grouping is along the last logical axis; codes may cross U32 boundaries.
    Affine {
        version: u32,
        bits: u8,
        group_size: u32,
        codes: Buffer,
        scales: TypedBuffer,
        biases: Option<TypedBuffer>,
    },
    /// Recipe 1: llama.cpp quantization blocks kept byte for byte. Each row of the last
    /// logical axis is a whole number of `format` blocks, rows are consecutive.
    Ggml {
        version: u32,
        format: GgmlBlock,
        data: Buffer,
    },
}

/// llama.cpp quantization blocks. Values follow llama.cpp's `dequantize_row_*` for the
/// type of the same name. A GGUF import keeps them byte for byte: no requantization, and
/// the native engine runs them with ggml's kernels directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlBlock {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq4Nl,
    Iq4Xs,
    Tq1_0,
    Tq2_0,
    Mxfp4,
}

/// (format, ggml type name, values per block, bytes per block).
const GGML_BLOCKS: [(GgmlBlock, &str, u64, u64); 22] = [
    (GgmlBlock::Q4_0, "q4_0", 32, 18),
    (GgmlBlock::Q4_1, "q4_1", 32, 20),
    (GgmlBlock::Q5_0, "q5_0", 32, 22),
    (GgmlBlock::Q5_1, "q5_1", 32, 24),
    (GgmlBlock::Q8_0, "q8_0", 32, 34),
    (GgmlBlock::Q2K, "q2_K", 256, 84),
    (GgmlBlock::Q3K, "q3_K", 256, 110),
    (GgmlBlock::Q4K, "q4_K", 256, 144),
    (GgmlBlock::Q5K, "q5_K", 256, 176),
    (GgmlBlock::Q6K, "q6_K", 256, 210),
    (GgmlBlock::Iq2Xxs, "iq2_xxs", 256, 66),
    (GgmlBlock::Iq2Xs, "iq2_xs", 256, 74),
    (GgmlBlock::Iq2S, "iq2_s", 256, 82),
    (GgmlBlock::Iq3Xxs, "iq3_xxs", 256, 98),
    (GgmlBlock::Iq3S, "iq3_s", 256, 110),
    (GgmlBlock::Iq1S, "iq1_s", 256, 50),
    (GgmlBlock::Iq1M, "iq1_m", 256, 56),
    (GgmlBlock::Iq4Nl, "iq4_nl", 32, 18),
    (GgmlBlock::Iq4Xs, "iq4_xs", 256, 136),
    (GgmlBlock::Tq1_0, "tq1_0", 256, 54),
    (GgmlBlock::Tq2_0, "tq2_0", 256, 66),
    (GgmlBlock::Mxfp4, "mxfp4", 32, 17),
];

impl GgmlBlock {
    fn entry(self) -> &'static (GgmlBlock, &'static str, u64, u64) {
        GGML_BLOCKS.iter().find(|e| e.0 == self).expect("every format has a table entry")
    }
    /// (values, bytes) per block.
    pub fn layout(self) -> (u64, u64) {
        let e = self.entry();
        (e.2, e.3)
    }
    /// The ggml type name (`iq3_s`).
    pub fn name(self) -> &'static str {
        self.entry().1
    }
    /// The format of a ggml type name, if it is one of these.
    pub fn named(name: &str) -> Option<Self> {
        GGML_BLOCKS.iter().find(|e| e.1 == name).map(|e| e.0)
    }
}

impl Storage {
    pub fn buffers(&self) -> Vec<&Buffer> {
        match self {
            Self::Dense { values } => vec![&values.data],
            Self::Affine {
                codes,
                scales,
                biases,
                ..
            } => {
                let mut out = vec![codes, &scales.data];
                if let Some(b) = biases {
                    out.push(&b.data);
                }
                out
            }
            Self::Ggml { data, .. } => vec![data],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub role: Role,
    pub shape: Vec<u64>,
    pub storage: Storage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BlockKind {
    Resident,
    Expert { layer: u32, expert: u32 },
    Tokenizer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub kind: BlockKind,
    pub offset: u64,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokenizer {
    /// Recipe 1 is self-contained Hugging Face tokenizers JSON with BPE algorithm.
    /// Normalizer/pre-tokenizer/decoder/added tokens are included in the same asset.
    pub version: u32,
    pub algorithm: TokenizerAlgorithm,
    pub data: Buffer,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub eog: Vec<u32>,
    pub add_bos: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerAlgorithm {
    Bpe,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prompt {
    /// Recipe 1: Jinja with minijinja's JSON + Python compatibility semantics.
    pub version: u32,
    pub template: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source: String,
    pub revision: Option<String>,
    pub importer: String,
    pub precision_changed: bool,
}

fn hash_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn elements(shape: &[u64]) -> Result<u64> {
    ensure!(
        !shape.is_empty() && shape.len() <= 4 && shape.iter().all(|&n| n > 0),
        "invalid logical rank/shape"
    );
    shape.iter().try_fold(1u64, |n, &v| {
        n.checked_mul(v).context("tensor shape overflow")
    })
}

impl Model {
    pub(crate) fn roles(&self) -> Result<BTreeMap<Role, Vec<u64>>> {
        use Projection::*;
        let a = &self.attention;
        let arch = self.architecture;
        ensure!(
            self.recipe_version == arch.recipe()
                && self.hybrid.is_some() == arch.is_qwen35()
                && self.shortconv.is_some() == arch.is_lfm2()
                && (self.routing.is_none() || arch == Architecture::Lfm2Moe),
            "unsupported model recipe"
        );
        ensure!(
            [
                self.hidden,
                self.layers,
                self.vocab,
                self.context,
                a.heads,
                a.kv_heads,
                a.head_dim
            ]
            .iter()
            .all(|&v| v > 0),
            "zero model dimension"
        );
        ensure!(self.layers <= 4096, "too many layers");
        ensure!(
            a.heads % a.kv_heads == 0 && a.head_dim % 2 == 0,
            "invalid attention heads/RoPE dimension"
        );
        ensure!(
            a.rope_theta.is_finite()
                && a.rope_theta > 0.0
                && self.rms_epsilon.is_finite()
                && self.rms_epsilon > 0.0,
            "invalid RoPE or RMS norm parameter"
        );
        ensure!(
            a.qk_norm == (self.architecture != Architecture::Qwen2),
            "architecture and Q/K normalization disagree"
        );
        let routing = self.routing.clone();
        let h = u64::from(self.hidden);
        let q = u64::from(a.heads) * u64::from(a.head_dim);
        let kv = u64::from(a.kv_heads) * u64::from(a.head_dim);
        let mut roles = BTreeMap::from([
            (Role::Embedding, vec![self.vocab.into(), h]),
            (Role::Head, vec![self.vocab.into(), h]),
            (Role::FinalNorm, vec![h]),
        ]);
        let (intermediate, experts) = match self.feed_forward {
            FeedForward::Dense { intermediate } => {
                ensure!(!arch.is_moe(), "MoE recipe requires experts");
                (intermediate, None)
            }
            FeedForward::Moe {
                intermediate,
                experts,
                active,
                ..
            } => {
                ensure!(arch.is_moe() && active > 0 && active <= experts, "invalid MoE routing");
                ensure!(
                    u64::from(experts) * u64::from(self.layers) * 3 <= MAX_TENSORS as u64,
                    "too many expert tensors"
                );
                (intermediate, Some(experts))
            }
        };
        ensure!(intermediate > 0, "zero intermediate dimension");
        if let Some(hy) = &self.hybrid {
            ensure!(
                [hy.conv_kernel, hy.key_heads, hy.value_heads, hy.key_dim, hy.value_dim]
                    .iter()
                    .all(|&v| v > 0)
                    && hy.value_heads % hy.key_heads == 0
                    // Only MoE models have a shared expert.
                    && (hy.shared_intermediate > 0) == arch.is_moe(),
                "invalid linear attention dimensions"
            );
            ensure!(
                hy.rope_dims > 0 && hy.rope_dims % 2 == 0 && hy.rope_dims <= a.head_dim,
                "invalid partial RoPE dimension"
            );
            ensure!(
                hy.linear_layers.windows(2).all(|w| w[0] < w[1])
                    && hy.linear_layers.last().is_none_or(|&l| l < self.layers),
                "linear attention layers must be sorted, unique and in range"
            );
        }
        if let Some(sc) = &self.shortconv {
            ensure!(sc.kernel >= 2, "invalid short convolution kernel");
            ensure!(
                sc.conv_layers.windows(2).all(|w| w[0] < w[1])
                    && sc.conv_layers.last().is_none_or(|&l| l < self.layers),
                "convolution layers must be sorted, unique and in range"
            );
        }
        if let Some(r) = &routing {
            ensure!(
                r.dense_layers < self.layers && (r.dense_layers == 0 || r.dense_intermediate > 0),
                "invalid dense lead layers"
            );
        }
        for layer in 0..self.layers {
            roles.insert(Role::AttentionNorm { layer }, vec![h]);
            roles.insert(Role::FeedForwardNorm { layer }, vec![h]);
            if let Some(hy) = &self.hybrid {
                if hy.linear_layers.binary_search(&layer).is_ok() {
                    let (kh, vh) = (u64::from(hy.key_heads), u64::from(hy.value_heads));
                    let (key, value) = (kh * u64::from(hy.key_dim), vh * u64::from(hy.value_dim));
                    let conv = 2 * key + value;
                    for (part, shape) in [
                        (LinearPart::Qkv, vec![conv, h]),
                        (LinearPart::Z, vec![value, h]),
                        (LinearPart::B, vec![vh, h]),
                        (LinearPart::A, vec![vh, h]),
                        (LinearPart::Conv, vec![conv, hy.conv_kernel.into()]),
                        (LinearPart::DtBias, vec![vh]),
                        (LinearPart::ALog, vec![vh]),
                        (LinearPart::Norm, vec![hy.value_dim.into()]),
                        (LinearPart::Output, vec![h, value]),
                    ] {
                        roles.insert(Role::LinearAttention { layer, part }, shape);
                    }
                } else {
                    // The query projection carries a sigmoid output gate per head.
                    for (projection, shape) in [
                        (Query, vec![2 * q, h]),
                        (Key, vec![kv, h]),
                        (Value, vec![kv, h]),
                        (Output, vec![h, q]),
                    ] {
                        roles.insert(Role::Attention { layer, projection }, shape);
                    }
                    roles.insert(Role::QueryNorm { layer }, vec![a.head_dim.into()]);
                    roles.insert(Role::KeyNorm { layer }, vec![a.head_dim.into()]);
                }
                if experts.is_some() {
                    let si = u64::from(hy.shared_intermediate);
                    for (projection, shape) in [(Gate, vec![si, h]), (Up, vec![si, h]), (Down, vec![h, si])] {
                        roles.insert(Role::SharedExpert { layer, projection }, shape);
                    }
                    roles.insert(Role::SharedExpertGate { layer }, vec![1, h]);
                }
            }
            let conv_layer = self
                .shortconv
                .as_ref()
                .is_some_and(|sc| sc.conv_layers.binary_search(&layer).is_ok());
            if let Some(sc) = self.shortconv.as_ref().filter(|_| conv_layer) {
                roles.insert(Role::ShortConv { layer, part: ConvPart::InProj }, vec![3 * h, h]);
                roles.insert(Role::ShortConv { layer, part: ConvPart::Conv }, vec![h, sc.kernel.into()]);
                roles.insert(Role::ShortConv { layer, part: ConvPart::OutProj }, vec![h, h]);
            }
            for (projection, shape) in [
                (Query, vec![q, h]),
                (Key, vec![kv, h]),
                (Value, vec![kv, h]),
                (Output, vec![h, q]),
            ]
            .into_iter()
            .filter(|_| self.hybrid.is_none() && !conv_layer)
            {
                if a.qkv_bias && projection != Output {
                    roles.insert(Role::AttentionBias { layer, projection }, vec![shape[0]]);
                }
                roles.insert(Role::Attention { layer, projection }, shape);
            }
            if a.qk_norm && self.hybrid.is_none() && !conv_layer {
                roles.insert(Role::QueryNorm { layer }, vec![a.head_dim.into()]);
                roles.insert(Role::KeyNorm { layer }, vec![a.head_dim.into()]);
            }
            let dense_lead = routing.as_ref().filter(|r| layer < r.dense_layers);
            if let Some(r) = dense_lead {
                let di = u64::from(r.dense_intermediate);
                for (projection, shape) in [(Gate, vec![di, h]), (Up, vec![di, h]), (Down, vec![h, di])] {
                    roles.insert(Role::FeedForward { layer, projection }, shape);
                }
            } else if let Some(experts) = experts {
                roles.insert(Role::Router { layer }, vec![experts.into(), h]);
                if routing.as_ref().is_some_and(|r| r.expert_bias) {
                    roles.insert(Role::ExpertBias { layer }, vec![experts.into()]);
                }
                for expert in 0..experts {
                    for projection in [Gate, Up, Down] {
                        let shape = if projection == Down {
                            vec![h, intermediate.into()]
                        } else {
                            vec![intermediate.into(), h]
                        };
                        roles.insert(
                            Role::Expert {
                                layer,
                                expert,
                                projection,
                            },
                            shape,
                        );
                    }
                }
            } else {
                for projection in [Gate, Up, Down] {
                    let shape = if projection == Down {
                        vec![h, intermediate.into()]
                    } else {
                        vec![intermediate.into(), h]
                    };
                    roles.insert(Role::FeedForward { layer, projection }, shape);
                }
            }
        }
        ensure!(roles.len() <= MAX_TENSORS, "too many model tensors");
        Ok(roles)
    }
}

impl Manifest {
    fn check_buffer(&self, b: &Buffer) -> Result<&Block> {
        let block = self
            .blocks
            .get(&b.block)
            .with_context(|| format!("missing block {}", b.block))?;
        ensure!(
            b.bytes > 0
                && b.offset
                    .checked_add(b.bytes)
                    .is_some_and(|end| end <= block.bytes),
            "buffer escapes block {}",
            b.block
        );
        Ok(block)
    }

    pub fn validate(&self, index_offset: u64) -> Result<()> {
        ensure!(self.version == VERSION, "unsupported native LMX version");
        ensure!(
            !self.blocks.is_empty()
                && self.blocks.len() <= MAX_TENSORS + 1
                && self.tensors.len() <= MAX_TENSORS,
            "invalid table size"
        );
        let mut occupied = Vec::new();
        for (id, b) in &self.blocks {
            ensure!(
                !id.is_empty() && id.len() <= 512 && !id.chars().any(char::is_control),
                "invalid block name"
            );
            ensure!(
                b.offset >= HEADER_BYTES && b.offset % ALIGNMENT == 0 && b.bytes > 0,
                "invalid block alignment/size: {id}"
            );
            let end = b
                .offset
                .checked_add(b.bytes)
                .context("block size overflow")?;
            ensure!(
                end <= index_offset && hash_valid(&b.sha256),
                "block escapes weights or has invalid hash: {id}"
            );
            occupied.push((b.offset, end));
        }
        occupied.sort_unstable();
        ensure!(
            occupied.windows(2).all(|p| p[0].1 <= p[1].0),
            "overlapping blocks"
        );
        let mut required = self.model.roles()?;
        let mut references: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
        for (name, tensor) in &self.tensors {
            ensure!(
                *name == tensor.role.name_in(self.model.architecture),
                "noncanonical HF tensor name: {name}"
            );
            let shape = required
                .remove(&tensor.role)
                .with_context(|| format!("unexpected/duplicate role: {name}"))?;
            ensure!(tensor.shape == shape, "wrong logical shape: {name}");
            let n = elements(&tensor.shape)?;
            match &tensor.storage {
                Storage::Dense { values } => {
                    ensure!(
                        values.data.offset % values.dtype.bytes() == 0,
                        "unaligned dense values: {name}"
                    );
                    ensure!(
                        n.checked_mul(values.dtype.bytes()) == Some(values.data.bytes),
                        "dense byte count mismatch: {name}"
                    );
                }
                Storage::Affine {
                    version,
                    bits,
                    group_size,
                    codes,
                    scales,
                    biases,
                } => {
                    ensure!(
                        *version == 1 && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) && *group_size > 0,
                        "unsupported affine schema: {name}"
                    );
                    ensure!(
                        tensor.shape.len() >= 2,
                        "affine requires matrix rows: {name}"
                    );
                    let last = *tensor.shape.last().unwrap();
                    ensure!(
                        last % u64::from(*group_size) == 0,
                        "incomplete affine group: {name}"
                    );
                    let row = last
                        .checked_mul(u64::from(*bits))
                        .and_then(|v| v.checked_add(7))
                        .context("packed size overflow")?
                        / 8;
                    ensure!(
                        (n / last).checked_mul(row) == Some(codes.bytes),
                        "code byte count mismatch: {name}"
                    );
                    for side in std::iter::once(scales).chain(biases) {
                        ensure!(
                            side.data.offset % side.dtype.bytes() == 0,
                            "unaligned affine coefficients: {name}"
                        );
                        ensure!(
                            (n / u64::from(*group_size)).checked_mul(side.dtype.bytes())
                                == Some(side.data.bytes),
                            "affine coefficient size mismatch: {name}"
                        );
                    }
                }
                Storage::Ggml { version, format, data } => {
                    ensure!(*version == 1, "unsupported ggml block schema: {name}");
                    let (values, bytes) = format.layout();
                    let last = *tensor.shape.last().unwrap();
                    ensure!(last % values == 0, "rows do not hold whole ggml blocks: {name}");
                    ensure!(
                        (n / values).checked_mul(bytes) == Some(data.bytes),
                        "ggml block byte count mismatch: {name}"
                    );
                }
            }
            let mut expert_block = None;
            for b in tensor.storage.buffers() {
                let block = self.check_buffer(b)?;
                match tensor.role {
                    Role::Expert { layer, expert, .. } => {
                        ensure!(
                            block.kind == (BlockKind::Expert { layer, expert }),
                            "expert must have its own block: {name}"
                        );
                        ensure!(
                            expert_block.is_none_or(|id| id == &b.block),
                            "expert coefficients must share the code block: {name}"
                        );
                        expert_block = Some(&b.block);
                    }
                    _ => ensure!(
                        block.kind == BlockKind::Resident,
                        "resident tensor in pageable block: {name}"
                    ),
                }
                if !(self.model.tied_embeddings && tensor.role == Role::Head) {
                    references
                        .entry(&b.block)
                        .or_default()
                        .push((b.offset, b.offset + b.bytes));
                }
            }
        }
        ensure!(
            required.is_empty(),
            "missing model tensors: {:?}",
            required.keys().take(4).collect::<Vec<_>>()
        );
        if self.model.tied_embeddings {
            let embedding = &self.tensors[&Role::Embedding.name_in(self.model.architecture)];
            let head = &self.tensors[&Role::Head.name_in(self.model.architecture)];
            ensure!(
                embedding.storage == head.storage,
                "tied head must share embedding storage"
            );
        }
        // All gate/up/down tensors for one expert use ONE independently readable block.
        let mut experts = BTreeMap::new();
        for b in self.blocks.values() {
            if let BlockKind::Expert { layer, expert } = b.kind {
                ensure!(
                    experts.insert((layer, expert), ()).is_none(),
                    "multiple blocks for one expert"
                );
            }
        }
        ensure!(
            self.tokenizer.version == 1 && self.prompt.version == 1,
            "unsupported tokenizer/template recipe"
        );
        ensure!(
            !self.prompt.template.trim().is_empty() && self.prompt.template.len() <= 1 << 20,
            "invalid chat template"
        );
        ensure!(
            !self.tokenizer.eos.is_empty()
                && (!self.tokenizer.add_bos || self.tokenizer.bos.is_some()),
            "invalid BOS/EOS policy"
        );
        for id in self
            .tokenizer
            .bos
            .iter()
            .chain(&self.tokenizer.eos)
            .chain(&self.tokenizer.eog)
        {
            ensure!(
                *id < self.model.vocab,
                "special token ID outside vocabulary"
            );
        }
        let t = &self.tokenizer.data;
        ensure!(
            self.check_buffer(t)?.kind == BlockKind::Tokenizer && t.bytes <= METADATA_LIMIT,
            "invalid tokenizer asset"
        );
        references
            .entry(&t.block)
            .or_default()
            .push((t.offset, t.offset + t.bytes));
        ensure!(
            references.len() == self.blocks.len(),
            "unreferenced native block"
        );
        for ranges in references.values_mut() {
            ranges.sort_unstable();
            // The explicitly tied head was checked against embedding storage above.
            // All other exact or partial aliases, including code/scale aliases, fail.
            ensure!(
                ranges.windows(2).all(|p| p[0].1 <= p[1].0),
                "overlapping tensor buffers"
            );
        }
        Ok(())
    }

    /// Stable across physical block ordering, outer alignment and provenance changes.
    /// Native block names and internal packing are part of schema 1's base identity.
    pub fn base_identity(&self) -> Result<String> {
        let blocks: BTreeMap<_, _> = self
            .blocks
            .iter()
            .map(|(id, b)| (id, (&b.kind, b.bytes, &b.sha256)))
            .collect();
        let bytes = serde_json::to_vec(&(
            "loom-native-base-v1",
            &self.model,
            &self.tensors,
            &blocks,
            &self.tokenizer,
            &self.prompt,
        ))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Independent expert ranges for the runtime; this never reads expert weights.
    pub fn expert_blocks(&self) -> BTreeMap<(u32, u32), &Block> {
        self.blocks
            .values()
            .filter_map(|b| match b.kind {
                BlockKind::Expert { layer, expert } => Some(((layer, expert), b)),
                _ => None,
            })
            .collect()
    }
}

/// Paging admission counts resident weights, KV, workspace AND slot buffers.
/// Slots cannot be recycled until the backend completes the consuming dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPlan {
    base_identity: String,
    budget: u64,
    kv_bytes: u64,
    workspace_bytes: u64,
    pub slot_bytes: u64,
    pub slots: u32,
    pub resident_bytes: u64,
    pub cache_bytes: u64,
    pub total_bytes: u64,
    pub max_prefill_tokens: u32,
}
impl ExpertPlan {
    pub fn new(m: &Manifest, budget: u64, kv_bytes: u64, workspace_bytes: u64) -> Result<Self> {
        m.validate(u64::MAX)?;
        let FeedForward::Moe { active, .. } = m.model.feed_forward else {
            anyhow::bail!("expert paging requires MoE");
        };
        let expert = m.expert_blocks();
        let max = expert
            .values()
            .map(|b| b.bytes)
            .max()
            .context("no expert blocks")?;
        let slot_bytes = max
            .checked_add(ALIGNMENT - 1)
            .context("slot size overflow")?
            / ALIGNMENT
            * ALIGNMENT;
        let resident_bytes = m
            .blocks
            .values()
            .filter(|b| !matches!(b.kind, BlockKind::Expert { .. }))
            .try_fold(0u64, |n, b| {
                n.checked_add(b.bytes).context("resident size overflow")
            })?;
        let fixed = resident_bytes
            .checked_add(kv_bytes)
            .and_then(|n| n.checked_add(workspace_bytes))
            .context("memory plan overflow")?;
        let free = budget
            .checked_sub(fixed)
            .context("resident weights/KV/workspace exceed memory budget")?;
        let slots = (free / slot_bytes)
            .min(expert.len() as u64)
            .min(u64::from(u32::MAX)) as u32;
        ensure!(
            slots >= active,
            "expert budget cannot hold one token's top-k experts"
        );
        let cache_bytes = u64::from(slots) * slot_bytes;
        Ok(Self {
            base_identity: m.base_identity()?,
            budget,
            kv_bytes,
            workspace_bytes,
            slot_bytes,
            slots,
            resident_bytes,
            cache_bytes,
            total_bytes: fixed + cache_bytes,
            max_prefill_tokens: slots / active,
        })
    }

    pub(crate) fn validate_for(&self, m: &Manifest) -> Result<()> {
        ensure!(
            *self == Self::new(m, self.budget, self.kv_bytes, self.workspace_bytes)?,
            "expert plan does not match this model or its derived memory totals"
        );
        Ok(())
    }
}


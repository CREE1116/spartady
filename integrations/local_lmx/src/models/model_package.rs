//! Device-independent description and structural validation of preserved MLX payloads.
//! Reads headers only; tensor data is streamed by the importer, never expanded to FP16.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub const MANIFEST: &str = "loom-model.json";
pub const PLAN_VERSION: u32 = 1;
pub const METADATA_LIMIT: u64 = 32 << 20;
const HEADER_LIMIT: u64 = 8 << 20;
const TENSOR_LIMIT: usize = 50_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PayloadFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AffineLayout {
    pub bits: u64,
    pub group_size: u64,
    pub axis: i32,
    pub packing: String,
    pub scales: String,
    pub biases: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorSpec {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offset in the safetensors payload, unlike the header's relative offset.
    pub offset: u64,
    pub bytes: u64,
    pub affine: Option<AffineLayout>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelManifest {
    pub version: u32,
    pub plan_version: u32,
    pub profile: String,
    pub backend: String,
    pub source: String,
    pub revision: Option<String>,
    pub identity: String,
    pub files: Vec<PayloadFile>,
    pub config: Value,
    pub tensors: BTreeMap<String, TensorSpec>,
    pub validation: String,
}

pub fn read_json(path: &Path) -> Result<Value> {
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= METADATA_LIMIT,
        "metadata too large: {}",
        path.display()
    );
    let mut data = Vec::new();
    file.by_ref()
        .take(METADATA_LIMIT + 1)
        .read_to_end(&mut data)?;
    ensure!(data.len() as u64 <= METADATA_LIMIT, "metadata too large");
    Ok(serde_json::from_slice(&data).with_context(|| path.display().to_string())?)
}

pub fn positive(config: &Value, name: &str) -> Result<u64> {
    let value = config[name]
        .as_u64()
        .filter(|value| *value > 0 && *value <= i32::MAX as u64);
    value.with_context(|| format!("invalid or missing {name}"))
}

/// Shared by admission and MLX discovery. Unsupported operations fail before weight download.
pub fn validate_mlx_config(config: &Value) -> Result<()> {
    let kind = config["model_type"]
        .as_str()
        .context("missing model_type")?;
    ensure!(
        matches!(kind, "qwen2" | "qwen3" | "qwen3_moe" | "llama" | "mistral"),
        "unsupported MLX architecture: {kind}"
    );
    ensure!(
        config["rope_scaling"].is_null(),
        "scaled RoPE is not implemented"
    );
    ensure!(
        config["hidden_act"]
            .as_str()
            .is_none_or(|act| act == "silu"),
        "unsupported activation"
    );
    ensure!(
        config["use_sliding_window"].as_bool() != Some(true)
            && config["sliding_window"]
                .as_i64()
                .is_none_or(|size| size <= 0),
        "sliding-window attention is not implemented"
    );
    ensure!(
        config["shared_expert_intermediate_size"]
            .as_i64()
            .is_none_or(|size| size <= 0),
        "shared experts are not implemented"
    );
    ensure!(
        config["partial_rotary_factor"].is_null()
            || config["partial_rotary_factor"].as_f64() == Some(1.0),
        "partial rotary dimensions are not implemented"
    );
    let hidden = positive(config, "hidden_size")?;
    let heads = positive(config, "num_attention_heads")?;
    positive(config, "num_hidden_layers")?;
    let kv = if config["num_key_value_heads"].is_null() {
        heads
    } else {
        positive(config, "num_key_value_heads")?
    };
    ensure!(heads % kv == 0, "attention head counts are incompatible");
    let head = if config["head_dim"].is_null() {
        ensure!(
            hidden % heads == 0,
            "hidden size does not divide attention heads"
        );
        hidden / heads
    } else {
        positive(config, "head_dim")?
    };
    ensure!(head > 0 && head % 2 == 0, "invalid RoPE head dimension");
    for field in ["rms_norm_eps", "rope_theta"] {
        ensure!(
            config[field].is_null()
                || config[field]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v > 0.0 && v <= f32::MAX as f64),
            "invalid {field}"
        );
    }
    if kind == "qwen3_moe" {
        let experts = positive(config, "num_experts")?;
        ensure!(
            positive(config, "num_experts_per_tok")? <= experts,
            "invalid active expert count"
        );
    }
    let quant = quantization(config);
    ensure!(
        quant.is_null() || quant.is_object(),
        "invalid quantization configuration"
    );
    ensure!(
        quant["quant_method"].is_null(),
        "GPTQ/AWQ and other quant_method layouts need a converter"
    );
    let group = quant["group_size"].as_u64().unwrap_or(64);
    ensure!(matches!(group, 32 | 64 | 128), "unsupported MLX group size");
    validate_quant_entry(quant, group)?;
    if let Some(entries) = quant.as_object() {
        for entry in entries.values().filter(|entry| entry.is_object()) {
            validate_quant_entry(entry, group)?;
        }
    }
    Ok(())
}

fn validate_quant_entry(entry: &Value, group: u64) -> Result<()> {
    ensure!(
        entry["mode"].as_str().is_none_or(|mode| mode == "affine"),
        "unsupported MLX quantization mode"
    );
    ensure!(
        entry["group_size"].is_null() || entry["group_size"].as_u64() == Some(group),
        "mixed group sizes are not implemented"
    );
    ensure!(
        entry["bits"].is_null()
            || entry["bits"]
                .as_u64()
                .is_some_and(|bits| matches!(bits, 0 | 2 | 3 | 4 | 5 | 6 | 8)),
        "unsupported MLX bit width"
    );
    Ok(())
}

pub fn quantization(config: &Value) -> &Value {
    if config["quantization"].is_object() {
        &config["quantization"]
    } else {
        &config["quantization_config"]
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// JSON maps normally overwrite duplicate names. Safetensors explicitly forbids them.
struct UniqueHeader(BTreeMap<String, Value>);
impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a unique tensor map")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut out = BTreeMap::new();
                while let Some((name, value)) = map.next_entry::<String, Value>()? {
                    if out.insert(name, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate tensor name"));
                    }
                }
                Ok(UniqueHeader(out))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

pub fn read_tensors(
    path: &Path,
    file_name: &str,
    out: &mut BTreeMap<String, TensorSpec>,
) -> Result<()> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    read_tensor_header(&mut file, length, file_name, out)
}

pub(crate) fn read_tensors_slice(
    path: &Path,
    file_name: &str,
    offset: u64,
    length: u64,
    out: &mut BTreeMap<String, TensorSpec>,
) -> Result<()> {
    let mut file = File::open(path)?;
    ensure!(
        offset
            .checked_add(length)
            .is_some_and(|end| end <= file.metadata().map(|m| m.len()).unwrap_or(0)),
        "truncated safetensors slice"
    );
    file.seek(SeekFrom::Start(offset))?;
    read_tensor_header(&mut file, length, file_name, out)
}

fn read_tensor_header(
    file: &mut File,
    length: u64,
    file_name: &str,
    out: &mut BTreeMap<String, TensorSpec>,
) -> Result<()> {
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)
        .context("truncated safetensors prefix")?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(
        header_len > 0 && header_len <= HEADER_LIMIT && header_len + 8 <= length,
        "invalid or truncated safetensors header"
    );
    let mut header = vec![0; header_len as usize];
    file.read_exact(&mut header)?;
    ensure!(header.first() == Some(&b'{'), "invalid safetensors header");
    let header: UniqueHeader = serde_json::from_slice(&header)?;
    let start = header_len + 8;
    let mut ranges = Vec::new();
    for (name, value) in header.0 {
        if name == "__metadata__" {
            let _: BTreeMap<String, String> = serde_json::from_value(value)?;
            continue;
        }
        ensure!(
            out.len() < TENSOR_LIMIT && name.len() <= 512,
            "tensor metadata exceeds import limit"
        );
        let tensor: HeaderTensor = serde_json::from_value(value).with_context(|| name.clone())?;
        ensure!(tensor.shape.len() <= 8, "tensor rank exceeds import limit");
        let element_bytes = match tensor.dtype.as_str() {
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            "BOOL" | "I8" | "U8" => 1,
            other => bail!("unsupported tensor dtype: {other}"),
        };
        let bytes = tensor
            .shape
            .iter()
            .try_fold(element_bytes, |n: u64, d| n.checked_mul(*d))
            .context("tensor size overflow")?;
        let [begin, end] = tensor.data_offsets;
        ensure!(
            begin <= end && end <= length - start && end - begin == bytes,
            "truncated tensor or invalid shape/range: {name}"
        );
        ranges.push((begin, end));
        ensure!(
            out.insert(
                name.clone(),
                TensorSpec {
                    file: file_name.into(),
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    offset: start + begin,
                    bytes,
                    affine: None
                }
            )
            .is_none(),
            "duplicate tensor across shards: {name}"
        );
    }
    ranges.sort_unstable();
    let mut cursor = 0;
    for (begin, end) in ranges {
        ensure!(
            begin == cursor,
            "safetensors data contains holes or overlapping tensors"
        );
        cursor = end;
    }
    ensure!(
        cursor == length - start,
        "safetensors contains unindexed or truncated data"
    );
    Ok(())
}

fn float_tensor(tensors: &BTreeMap<String, TensorSpec>, name: &str, shape: &[u64]) -> Result<()> {
    let tensor = tensors
        .get(name)
        .with_context(|| format!("missing tensor: {name}"))?;
    ensure!(
        tensor.shape == shape && matches!(tensor.dtype.as_str(), "F16" | "BF16" | "F32"),
        "unsupported tensor shape/dtype: {name}"
    );
    Ok(())
}

fn linear(
    tensors: &mut BTreeMap<String, TensorSpec>,
    prefix: &str,
    logical: &[u64],
    group: u64,
) -> Result<()> {
    let name = format!("{prefix}.weight");
    let scale_name = format!("{prefix}.scales");
    let bias_name = format!("{prefix}.biases");
    let weight = tensors
        .get(&name)
        .with_context(|| format!("missing tensor: {name}"))?
        .clone();
    if tensors.contains_key(&scale_name) {
        let input = *logical.last().unwrap();
        ensure!(
            input % group == 0,
            "quantized input does not divide group size: {prefix}"
        );
        let mut scales = logical.to_vec();
        *scales.last_mut().unwrap() = input / group;
        float_tensor(tensors, &scale_name, &scales)?;
        ensure!(
            weight.dtype == "U32"
                && weight.shape.len() == logical.len()
                && weight.shape[..weight.shape.len() - 1] == logical[..logical.len() - 1],
            "invalid packed weight: {prefix}"
        );
        let packed_bits = weight
            .shape
            .last()
            .unwrap()
            .checked_mul(32)
            .context("packing overflow")?;
        ensure!(packed_bits % input == 0, "invalid affine packing: {prefix}");
        let bits = packed_bits / input;
        ensure!(
            matches!(bits, 2 | 3 | 4 | 5 | 6 | 8),
            "unsupported packed bit width: {prefix}"
        );
        let biases = if tensors.contains_key(&bias_name) {
            float_tensor(tensors, &bias_name, &scales)?;
            Some(bias_name)
        } else {
            None
        };
        tensors.get_mut(&name).unwrap().affine = Some(AffineLayout {
            bits,
            group_size: group,
            axis: -1,
            packing: "mlx-affine-u32-v1".into(),
            scales: scale_name,
            biases,
        });
    } else {
        ensure!(
            !tensors.contains_key(&bias_name),
            "affine biases without scales: {prefix}"
        );
        float_tensor(tensors, &name, logical)?;
    }
    let bias = format!("{prefix}.bias");
    if tensors.contains_key(&bias) {
        float_tensor(tensors, &bias, &logical[..logical.len() - 1])?;
    }
    Ok(())
}

pub fn validate_tensors(config: &Value, tensors: &mut BTreeMap<String, TensorSpec>) -> Result<()> {
    validate_mlx_config(config)?;
    ensure!(
        tensors
            .values()
            .all(|tensor| matches!(tensor.dtype.as_str(), "F16" | "BF16" | "F32" | "U32")),
        "payload contains a dtype outside the current MLX import contract"
    );
    let hidden = positive(config, "hidden_size")?;
    let vocab = positive(config, "vocab_size")?;
    let layers = positive(config, "num_hidden_layers")?;
    ensure!(layers <= 1024, "layer count exceeds import limit");
    let heads = positive(config, "num_attention_heads")?;
    let kv = config["num_key_value_heads"].as_u64().unwrap_or(heads);
    let head = config["head_dim"].as_u64().unwrap_or(hidden / heads);
    let moe = config["model_type"] == "qwen3_moe";
    let intermediate = positive(
        config,
        if moe {
            "moe_intermediate_size"
        } else {
            "intermediate_size"
        },
    )?;
    let group = quantization(config)["group_size"].as_u64().unwrap_or(64);
    linear(tensors, "model.embed_tokens", &[vocab, hidden], group)?;
    if config["tie_word_embeddings"].as_bool() != Some(true) {
        linear(tensors, "lm_head", &[vocab, hidden], group)?;
    }
    float_tensor(tensors, "model.norm.weight", &[hidden])?;
    for layer in 0..layers {
        let p = format!("model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            float_tensor(tensors, &format!("{p}.{name}.weight"), &[hidden])?;
        }
        for (name, rows, cols) in [
            ("q_proj", heads * head, hidden),
            ("k_proj", kv * head, hidden),
            ("v_proj", kv * head, hidden),
            ("o_proj", hidden, heads * head),
        ] {
            linear(
                tensors,
                &format!("{p}.self_attn.{name}"),
                &[rows, cols],
                group,
            )?;
        }
        if matches!(config["model_type"].as_str(), Some("qwen3" | "qwen3_moe")) {
            for name in ["q_norm", "k_norm"] {
                float_tensor(tensors, &format!("{p}.self_attn.{name}.weight"), &[head])?;
            }
        }
        let bank = if moe {
            let experts = positive(config, "num_experts")?;
            linear(tensors, &format!("{p}.mlp.gate"), &[experts, hidden], group)?;
            Some(experts)
        } else {
            None
        };
        for (name, rows, cols) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            let (prefix, shape) = match bank {
                Some(experts) => (
                    format!("{p}.mlp.switch_mlp.{name}"),
                    vec![experts, rows, cols],
                ),
                None => (format!("{p}.mlp.{name}"), vec![rows, cols]),
            };
            linear(tensors, &prefix, &shape, group)?;
        }
    }
    Ok(())
}


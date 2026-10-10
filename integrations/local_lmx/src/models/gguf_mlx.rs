pub fn hf_name(gguf: &str) -> Option<String> {
    // Qwen3.5 linear-attention parameters whose GGUF names have no `.weight` form.
    if let Some((layer, part)) = gguf.strip_prefix("blk.").and_then(|rest| rest.split_once('.')) {
        layer.parse::<u32>().ok()?;
        match part {
            "ssm_a" => return Some(format!("model.layers.{layer}.linear_attn.A_log")),
            "ssm_dt.bias" => return Some(format!("model.layers.{layer}.linear_attn.dt_bias")),
            // LFM2-MoE: the expert selection bias.
            "exp_probs_b.bias" => return Some(format!("model.layers.{layer}.mlp.expert_bias")),
            _ => {}
        }
    }
    let (base, suffix) = gguf.rsplit_once('.')?;
    if !matches!(suffix, "weight" | "bias") {
        return None;
    }
    let mapped = match base {
        "token_embd" => "model.embed_tokens".to_owned(),
        // LFM2 names its final norm `token_embd_norm`.
        "output_norm" | "token_embd_norm" => "model.norm".to_owned(),
        "output" => "lm_head".to_owned(),
        _ => {
            let (layer, part) = base.strip_prefix("blk.")?.split_once('.')?;
            layer.parse::<u32>().ok()?;
            let part = match part {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "attn_q_norm" => "self_attn.q_norm",
                "attn_k_norm" => "self_attn.k_norm",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                "ffn_gate_inp" => "mlp.gate",
                "ffn_gate_exps" => "mlp.switch_mlp.gate_proj",
                "ffn_up_exps" => "mlp.switch_mlp.up_proj",
                "ffn_down_exps" => "mlp.switch_mlp.down_proj",
                // Qwen3.5: post-attention norm, gated DeltaNet and the shared expert.
                "post_attention_norm" => "post_attention_layernorm",
                "attn_qkv" => "linear_attn.in_proj_qkv",
                "attn_gate" => "linear_attn.in_proj_z",
                "ssm_beta" => "linear_attn.in_proj_b",
                "ssm_alpha" => "linear_attn.in_proj_a",
                "ssm_conv1d" => "linear_attn.conv1d",
                "ssm_norm" => "linear_attn.norm",
                "ssm_out" => "linear_attn.out_proj",
                "ffn_gate_shexp" => "mlp.shared_expert.gate_proj",
                "ffn_up_shexp" => "mlp.shared_expert.up_proj",
                "ffn_down_shexp" => "mlp.shared_expert.down_proj",
                "ffn_gate_inp_shexp" => "mlp.shared_expert_gate",
                // LFM2: gated short convolution.
                "shortconv.conv" => "conv.conv",
                "shortconv.in_proj" => "conv.in_proj",
                "shortconv.out_proj" => "conv.out_proj",
                _ => return None,
            };
            format!("model.layers.{layer}.{part}")
        }
    };
    Some(format!("{mapped}.{suffix}"))
}


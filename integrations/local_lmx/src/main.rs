mod models;
use anyhow::{Context, Result, ensure};
use models::{lmx_engine::{Engine, EngineOptions}, lmx_native_file::NativeFile};
use serde_json::{Value, json};
use std::{io::{self, BufRead, Write}, path::Path, time::Instant};

fn render(file: &NativeFile, tokenizer: &tokenizers::Tokenizer, r: &Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |message: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
    });
    let mut template = file.manifest.prompt.template.clone();
    for tag in ["generation", "endgeneration"] {
        for (open, close) in [("{%", "%}"), ("{%-", "%}"), ("{%", "-%}"), ("{%-", "-%}")] {
            for spaced in [format!("{open} {tag} {close}"), format!("{open}{tag}{close}")] {
                template = template.replace(&spaced, "");
            }
        }
    }
    env.add_template("chat", &template)?;
    let t = &file.manifest.tokenizer;
    let piece = |id: Option<u32>| id.and_then(|i| tokenizer.id_to_token(i)).unwrap_or_default();
    Ok(env.get_template("chat")?.render(minijinja::Value::from_serialize(json!({
        "messages":[{"role":"system","content":r["system"].as_str().unwrap_or("")},
                    {"role":"user","content":r["prompt"].as_str().context("prompt required")?}],
        "add_generation_prompt":true, "enable_thinking":false,
        "bos_token":if t.add_bos {piece(t.bos)} else {String::new()},
        "eos_token":piece(t.eos.first().copied())
    })))?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("convert") {
        ensure!(args.len()==4, "usage: spartady-local-lmx convert CHECKPOINT_DIRECTORY OUTPUT.lmx");
        let source=Path::new(&args[2]);
        let file=if source.is_file(){models::lmx_native_gguf::import_gguf(source,Path::new(&args[3]),&mut |done,total| {eprintln!("{done}/{total}");true})?}else{models::lmx_native_import::import_mlx(source,Path::new(&args[3]),&mut |done,total| {eprintln!("{done}/{total}");true})?};
        if !models::lmx_engine::runnable(file.manifest.model.architecture){
            std::fs::remove_file(&args[3])?;anyhow::bail!("Converted architecture is not executable by this runtime; choose Qwen3.5 or LFM2 GGUF");
        }
        file.verify(&mut |_,_|true)?;
        println!("{}",json!({"path":args[3],"identity":file.identity,"verified":true}));
        return Ok(());
    }
    let path = std::env::args().nth(1).context("usage: spartady-local-lmx MODEL.lmx")?;
    let file = NativeFile::open(Path::new(&path))?;
    let mut data = vec![0; usize::try_from(file.manifest.tokenizer.data.bytes)?];
    file.read_range_into(&file.manifest.tokenizer.data, &mut data)?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&data).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let options = EngineOptions {context:8192, chunk:256, ..EngineOptions::default()};
    let mut engine = Engine::open(Path::new(&path), &options)?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = (|| -> Result<Value> {
            let start = Instant::now();
            let r: Value = serde_json::from_str(&line)?;
            let limit = r["max_output_tokens"].as_u64().unwrap_or(2048) as usize;
            ensure!((1..=8192).contains(&limit), "invalid output limit");
            let prompt = render(&file, &tokenizer, &r)?;
            let enc = tokenizer.encode(prompt.as_str(), false).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let mut ids = enc.get_ids().to_vec();
            let t = &file.manifest.tokenizer;
            if t.add_bos && t.bos.is_some() && ids.first()!=t.bos.as_ref() {ids.insert(0,t.bos.unwrap());}
            ensure!(!ids.is_empty() && ids.len()+limit <= engine.stats().context as usize, "context budget exceeded");
            engine.reset();
            let mut logits = vec![];
            for chunk in ids.chunks(options.chunk as usize) {logits=engine.eval(chunk)?;}
            let mut generated = Vec::new();
            let mut stopped = false;
            for _ in 0..limit {
                let token = logits.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).context("empty logits")?.0 as u32;
                if t.eos.contains(&token) || t.eog.contains(&token) {stopped=true;break;}
                generated.push(token);
                logits=engine.eval(&[token])?;
            }
            let text = tokenizer.decode(&generated, true).map_err(|e|anyhow::anyhow!(e.to_string()))?;
            Ok(json!({"text":text,"model":file.identity,"provider":"local_lmx",
                "usage":{"input_tokens":ids.len(),"output_tokens":generated.len()},
                "stop":if stopped {"stop"} else {"length"},"latency_ms":start.elapsed().as_millis()}))
        })();
        println!("{}",match result {Ok(v)=>v,Err(e)=>json!({"error":e.to_string()})});
        io::stdout().flush()?;
    }
    Ok(())
}

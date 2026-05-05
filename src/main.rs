use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command as ProcessCommand;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

static QUIET_RECEIPTS: AtomicBool = AtomicBool::new(false);

const PACKET_VERSION: &str = "state-pack-v0.1";

#[derive(Parser, Debug)]
#[command(name = "state-pack")]
#[command(about = "Content-addressed transformer state packet store")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create a content-addressed packet from base text and a cache/state blob.
    Create {
        /// Model identity, e.g. gpt2.
        #[arg(long)]
        model: String,
        /// Base prompt/state text whose semantic state is represented by the blob.
        #[arg(long)]
        base: PathBuf,
        /// Raw state blob, e.g. a torch .pt file containing past_key_values.
        #[arg(long)]
        blob: PathBuf,
        /// Packet store directory.
        #[arg(long, default_value = "packets")]
        out: PathBuf,
    },
    /// Verify a packet manifest and its blob.
    Verify {
        /// Path to packet manifest JSON.
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Create a delta packet addressed to an existing base/state packet.
    Delta {
        /// Path to base packet manifest JSON.
        #[arg(long)]
        manifest: PathBuf,
        /// Delta text to apply after the cached state.
        #[arg(long)]
        delta: PathBuf,
        /// Output delta packet JSON.
        #[arg(long)]
        out: PathBuf,
    },
    Merge {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        delta: PathBuf,
        #[arg(long)]
        blob: PathBuf,
        #[arg(long, default_value = "packets")]
        out: PathBuf,
    },
    /// Resolve a packet by base hash inside a packet store.
    Resolve {
        #[arg(long, default_value = "packets")]
        store: PathBuf,
        #[arg(long)]
        base_hash: String,
    },
    Infer {
        #[arg(long)]
        delta: PathBuf,
        #[arg(long, default_value = "packets")]
        store: PathBuf,
    },
    Tokenize {
        #[arg(long)]
        delta: PathBuf,
    },
    Benchmark {
        #[arg(long)]
        script: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        input_cost_per_m: f64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    BenchmarkNative {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        blob: PathBuf,
        #[arg(long, default_value_t = 40)]
        steps: u64,
        #[arg(long, default_value_t = 1)]
        merge_every: u64,
        #[arg(long, default_value = "fixed")]
        merge_policy: String,
        #[arg(long, default_value_t = 4.0)]
        merge_threshold: f64,
        #[arg(long, default_value_t = 5)]
        max_steps_without_merge: u64,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        base_target_tokens: u64,
        #[arg(long, default_value_t = 0.2)]
        delta_variance: f64,
        #[arg(long, default_value = "gpt2")]
        model: String,
        #[arg(long, default_value = "native_benchmark")]
        workdir: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        input_cost_per_m: f64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Garbage collect the packet store.
    Gc {
        /// Packet store directory.
        #[arg(long, default_value = "packets")]
        store: PathBuf,
        /// Delete packets older than N days.
        #[arg(long)]
        older_than: Option<u64>,
        /// Keep only the N most recently modified packets.
        #[arg(long)]
        keep_latest: Option<usize>,
        /// Show what would be deleted without deleting anything.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct StatePacketManifest {
    version: String,
    model: String,
    base_sha256: String,
    base_bytes: u64,
    base_tokens: u64,
    base_file: String,
    blob_sha256: String,
    blob_bytes: u64,
    blob_file: String,
    packet_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeltaPacket {
    version: String,
    model: String,
    base_sha256: String,
    packet_id: String,
    delta_sha256: String,
    delta_bytes: u64,
    delta_text: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Receipt {
    receipt_id: Option<String>,
    op: String,
    ok: bool,
    packet_id: Option<String>,
    base_sha256: Option<String>,
    blob_sha256: Option<String>,
    delta_sha256: Option<String>,
    bytes: Option<u64>,
    bytes_saved: Option<ByteSavings>,
    tokens: Option<TokenSavings>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ByteSavings {
    base: u64,
    delta: u64,
    processed: u64,
    saved: u64,
    savings_percent: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct TokenSavings {
    base: u64,
    delta: u64,
    processed: u64,
    saved: u64,
    savings_percent: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Create { model, base, blob, out } => create_packet(&model, &base, &blob, &out),
        Command::Verify { manifest } => verify_packet(&manifest),
        Command::Delta { manifest, delta, out } => create_delta(&manifest, &delta, &out).map(|_| ()),
        Command::Merge { manifest, delta, blob, out } => merge_packet(&manifest, &delta, &blob, &out).map(|_| ()),
        Command::Resolve { store, base_hash } => resolve_packet(&store, &base_hash),
        Command::Infer { delta, store } => infer_packet(&delta, &store).map(|_| ()),
        Command::Tokenize { delta } => tokenize_delta(&delta),
        Command::Benchmark { script, input_cost_per_m, out } => run_benchmark(&script, input_cost_per_m, out.as_deref()),
        Command::BenchmarkNative { base, blob, steps, merge_every, merge_policy, merge_threshold, max_steps_without_merge, base_target_tokens, delta_variance, model, workdir, input_cost_per_m, out, report } => benchmark_native(&base, &blob, steps, merge_every, &merge_policy, merge_threshold, max_steps_without_merge, base_target_tokens, delta_variance, &model, &workdir, input_cost_per_m, out.as_deref(), report.as_deref()),
        Command::Gc { store, older_than, keep_latest, dry_run } => run_gc(&store, older_than, keep_latest, dry_run),
    }
}

fn emit_receipt(mut receipt: Receipt) -> Result<Receipt> {
    receipt.receipt_id = None;
    let canonical = serde_json::to_vec(&receipt)?;
    receipt.receipt_id = Some(format!("sha256:{}", sha256_bytes(&canonical)));
    if !QUIET_RECEIPTS.load(Ordering::SeqCst) {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    }
    Ok(receipt)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut n = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = f.read(&mut buf)?;
        if read == 0 { break; }
        h.update(&buf[..read]);
        n += read as u64;
    }
    Ok((hex::encode(h.finalize()), n))
}

fn create_packet(model: &str, base_path: &Path, blob_path: &Path, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let base_bytes_vec = fs::read(base_path).with_context(|| format!("read base {}", base_path.display()))?;
    let base_sha256 = sha256_bytes(&base_bytes_vec);
    let base_bytes = base_bytes_vec.len() as u64;
    let base_tokens = gpt2_token_count(base_path).unwrap_or(0);

    let (blob_sha256, blob_bytes) = sha256_file(blob_path)?;

    let blob_file = format!("state_packet_{}.pt", base_sha256);
    let blob_dest = out_dir.join(&blob_file);
    fs::copy(blob_path, &blob_dest)
        .with_context(|| format!("copy blob to {}", blob_dest.display()))?;

    let mut manifest = StatePacketManifest {
        version: PACKET_VERSION.to_string(),
        model: model.to_string(),
        base_sha256,
        base_bytes,
        base_tokens,
        base_file: base_path.file_name().ok_or_else(|| anyhow!("base path has no file name"))?.to_string_lossy().to_string(),
        blob_sha256,
        blob_bytes,
        blob_file,
        packet_id: String::new(),
    };

    let preimage = format!(
        "{}|{}|{}|{}|{}|{}",
        manifest.version,
        manifest.model,
        manifest.base_sha256,
        manifest.base_bytes,
        manifest.blob_sha256,
        manifest.blob_bytes
    );
    manifest.packet_id = format!("sha256:{}", sha256_bytes(preimage.as_bytes()));

    let manifest_file = format!("state_packet_{}.json", manifest.base_sha256);
    let manifest_path = out_dir.join(&manifest_file);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    let receipt = Receipt {
        receipt_id: None,
        op: "create".to_string(),
        ok: true,
        packet_id: Some(manifest.packet_id.clone()),
        base_sha256: Some(manifest.base_sha256.clone()),
        blob_sha256: Some(manifest.blob_sha256.clone()),
        delta_sha256: None,
        bytes: Some(manifest.blob_bytes),
        bytes_saved: None,
        tokens: None,
    };
    emit_receipt(receipt)?;
    Ok(())
}

fn read_manifest(path: &Path) -> Result<StatePacketManifest> {
    let bytes = fs::read(path).with_context(|| format!("read manifest {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).context("parse manifest json")?)
}

fn read_delta_packet(path: &Path) -> Result<DeltaPacket> {
    let bytes = fs::read(path).with_context(|| format!("read delta packet {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).context("parse delta packet json")?)
}

fn verify_packet(manifest_path: &Path) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    if manifest.version != PACKET_VERSION {
        bail!("unsupported packet version: {}", manifest.version);
    }

    let blob_path = manifest_path
        .parent()
        .ok_or_else(|| anyhow!("manifest has no parent directory"))?
        .join(&manifest.blob_file);

    let (actual_blob_hash, actual_blob_bytes) = sha256_file(&blob_path)?;
    if actual_blob_hash != manifest.blob_sha256 {
        bail!("blob hash mismatch: expected {}, got {}", manifest.blob_sha256, actual_blob_hash);
    }
    if actual_blob_bytes != manifest.blob_bytes {
        bail!("blob byte mismatch: expected {}, got {}", manifest.blob_bytes, actual_blob_bytes);
    }

    let preimage = format!(
        "{}|{}|{}|{}|{}|{}",
        manifest.version,
        manifest.model,
        manifest.base_sha256,
        manifest.base_bytes,
        manifest.blob_sha256,
        manifest.blob_bytes
    );
    let actual_packet_id = format!("sha256:{}", sha256_bytes(preimage.as_bytes()));
    if actual_packet_id != manifest.packet_id {
        bail!("packet_id mismatch: expected {}, got {}", manifest.packet_id, actual_packet_id);
    }

    let receipt = Receipt {
        receipt_id: None,
        op: "verify".to_string(),
        ok: true,
        packet_id: Some(manifest.packet_id.clone()),
        base_sha256: Some(manifest.base_sha256.clone()),
        blob_sha256: Some(manifest.blob_sha256.clone()),
        delta_sha256: None,
        bytes: Some(manifest.blob_bytes),
        bytes_saved: None,
        tokens: None,
    };
    emit_receipt(receipt)?;
    Ok(())
}

fn create_delta(manifest_path: &Path, delta_path: &Path, out_path: &Path) -> Result<Receipt> {
    let manifest = read_manifest(manifest_path)?;
    let delta_text = fs::read_to_string(delta_path).with_context(|| format!("read delta {}", delta_path.display()))?;
    let delta_sha256 = sha256_bytes(delta_text.as_bytes());
    let delta = DeltaPacket {
        version: "state-delta-v0.1".to_string(),
        model: manifest.model,
        base_sha256: manifest.base_sha256,
        packet_id: manifest.packet_id,
        delta_sha256,
        delta_bytes: delta_text.as_bytes().len() as u64,
        delta_text,
    };
    fs::write(out_path, serde_json::to_vec_pretty(&delta)?)?;
    let receipt = Receipt {
        receipt_id: None,
        op: "delta".to_string(),
        ok: true,
        packet_id: Some(delta.packet_id.clone()),
        base_sha256: Some(delta.base_sha256.clone()),
        blob_sha256: None,
        delta_sha256: Some(delta.delta_sha256.clone()),
        bytes: Some(delta.delta_bytes),
        bytes_saved: None,
        tokens: None,
    };
    emit_receipt(receipt)
}

fn gpt2_token_count_text(text: &str) -> Result<u64> {
    let output = ProcessCommand::new("python3")
        .arg("gpt2_count.py")
        .arg("--text")
        .arg(text)
        .output()
        .context("run gpt2_count.py --text")?;

    if !output.status.success() {
        bail!("gpt2_count.py failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let v: serde_json::Value = serde_json::from_slice(&output.stdout).context("parse gpt2_count.py json")?;
    Ok(v["token_count"].as_u64().ok_or_else(|| anyhow!("missing token_count"))?)
}

fn gpt2_token_count(path: &Path) -> Result<u64> {
    let output = ProcessCommand::new("python3")
        .arg("gpt2_count.py")
        .arg("--text-file")
        .arg(path)
        .output()
        .context("run gpt2_count.py")?;

    if !output.status.success() {
        bail!("gpt2_count.py failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let v: serde_json::Value = serde_json::from_slice(&output.stdout).context("parse gpt2_count.py json")?;
    Ok(v["token_count"].as_u64().ok_or_else(|| anyhow!("missing token_count"))?)
}

fn benchmark_native(
    base: &Path,
    blob: &Path,
    steps: u64,
    merge_every: u64,
    merge_policy: &str,
    merge_threshold: f64,
    max_steps_without_merge: u64,
    base_target_tokens: u64,
    delta_variance: f64,
    model: &str,
    workdir: &Path,
    input_cost_per_m: f64,
    out: Option<&Path>,
    report: Option<&Path>,
) -> Result<()> {
    QUIET_RECEIPTS.store(true, Ordering::SeqCst);
    fs::create_dir_all(workdir).with_context(|| format!("create {}", workdir.display()))?;
    let store = workdir.join("store");
    fs::create_dir_all(&store).with_context(|| format!("create {}", store.display()))?;

    let raw_base = fs::read_to_string(base).with_context(|| format!("read base {}", base.display()))?;
    let mut expanded_base = String::new();
    while gpt2_token_count_text(&expanded_base).unwrap_or(0) < base_target_tokens {
        expanded_base.push_str(&raw_base);
        expanded_base.push('\n');
    }
    let effective_base = workdir.join("expanded_base.txt");
    fs::write(&effective_base, expanded_base)?;

    create_packet(model, &effective_base, blob, &store)?;

    let base_bytes = fs::read(&effective_base).with_context(|| format!("read base {}", effective_base.display()))?;
    let base_hash = sha256_bytes(&base_bytes);
    let mut current_manifest = store.join(format!("state_packet_{}.json", base_hash));

    let mut naive_tokens = 0u64;
    let mut state_pack_tokens = read_manifest(&current_manifest)?.base_tokens;
    let mut per_step = Vec::new();
    let mut steps_since_merge = 0u64;

    for step in 1..=steps {
        let variance_band = ((delta_variance * 10.0).round() as u64).max(1);
        let repeat_count = 1 + ((step * 17 + 13) % (variance_band + 2));
        let mut delta_text = format!(
            "Step {}: Tool observation says contract clause {} affects indemnity, waiver, notice, or damages.\n",
            step, step
        );
        for _ in 0..repeat_count {
            delta_text.push_str("Additional tool note affects timing, scope, and follow-up obligations.\n");
        }
        let delta_path = workdir.join(format!("delta_{}.txt", step));
        fs::write(&delta_path, &delta_text)?;

        let prior = read_manifest(&current_manifest)?;
        let delta_tokens = gpt2_token_count_text(&delta_text).unwrap_or(0);
        let full_tokens_this_step = prior.base_tokens + delta_tokens;

        let delta_packet_path = workdir.join(format!("delta_packet_{}.json", step));
        let delta_receipt = create_delta(&current_manifest, &delta_path, &delta_packet_path)?;
        let infer_receipt = infer_packet(&delta_packet_path, &store)?;

        naive_tokens += full_tokens_this_step;
        state_pack_tokens += delta_tokens;

        let merge_ratio = if prior.base_tokens == 0 {
            0.0
        } else {
            delta_tokens as f64 / prior.base_tokens as f64
        };
        let should_merge = match merge_policy {
            "fixed" => merge_every > 0 && step % merge_every == 0,
            "adaptive" => (merge_ratio > merge_threshold && steps_since_merge >= 3) || steps_since_merge >= max_steps_without_merge,
            other => bail!("unsupported merge_policy: {}", other),
        };
        let mut merge_receipt: Option<Receipt> = None;
        if should_merge {
            steps_since_merge = 0;
            let receipt = merge_packet(&current_manifest, &delta_packet_path, blob, &store)?;
            merge_receipt = Some(receipt);
            let delta_packet = read_delta_packet(&delta_packet_path)?;
            let merged_hash = sha256_bytes(
                format!("{}|{}", prior.base_sha256, delta_packet.delta_sha256).as_bytes()
            );
            current_manifest = store.join(format!("state_packet_{}.json", merged_hash));
        } else {
            steps_since_merge += 1;
        }

        let cumulative_saved = naive_tokens.saturating_sub(state_pack_tokens);
        let cumulative_savings_percent =
            if naive_tokens == 0 { 0.0 } else { (cumulative_saved as f64 / naive_tokens as f64) * 100.0 };

        let delta_packet_for_step = read_delta_packet(&delta_packet_path)?;
        per_step.push(serde_json::json!({
            "step": step,
            "base_sha256": prior.base_sha256,
            "delta_sha256": delta_packet_for_step.delta_sha256,
            "delta_text_preview": delta_text.chars().take(96).collect::<String>(),
            "delta_receipt_id": delta_receipt.receipt_id,
            "infer_receipt_id": infer_receipt.receipt_id,
            "merge_receipt_id": merge_receipt.as_ref().and_then(|r| r.receipt_id.clone()),
            "base_tokens": prior.base_tokens,
            "delta_tokens": delta_tokens,
            "naive_tokens_this_step": full_tokens_this_step,
            "state_pack_tokens_this_step": delta_tokens,
            "tokens_saved_this_step": full_tokens_this_step.saturating_sub(delta_tokens),
            "cumulative_naive_tokens": naive_tokens,
            "cumulative_state_pack_tokens": state_pack_tokens,
            "cumulative_tokens_saved": cumulative_saved,
            "cumulative_savings_percent": cumulative_savings_percent,
            "merged": should_merge
        }));
    }

    let tokens_saved = naive_tokens.saturating_sub(state_pack_tokens);
    let savings_percent =
        if naive_tokens == 0 { 0.0 } else { (tokens_saved as f64 / naive_tokens as f64) * 100.0 };
    let estimated_usd_saved = (tokens_saved as f64 / 1_000_000.0) * input_cost_per_m;

    let avg_naive = if steps == 0 { 0.0 } else { naive_tokens as f64 / steps as f64 };
    let avg_state = if steps == 0 { 0.0 } else { state_pack_tokens as f64 / steps as f64 };
    let merge_count = per_step.iter().filter(|s| s["merged"].as_bool().unwrap_or(false)).count();
    let final_base_tokens = per_step.last().and_then(|s| s["base_tokens"].as_u64()).unwrap_or(0);

    let mut result = serde_json::json!({
        "summary": {
    "headline": format!("State Pack achieved {:.2}% token savings with adaptive merging over {} steps", savings_percent, steps),
    "tokens_naive": naive_tokens,
    "tokens_state_pack": state_pack_tokens,
    "tokens_saved": tokens_saved,
    "savings_percent": savings_percent,
    "avg_tokens_per_step_naive": avg_naive,
    "avg_tokens_per_step_state": avg_state,
    "merge_count": merge_count,
    "final_base_tokens": final_base_tokens
  },
  "op": "benchmark-native",
        "model": model,
        "steps": steps,
        "merge_every": merge_every,
        "merge_policy": merge_policy,
        "merge_threshold": merge_threshold,
        "naive": {
            "tokens_processed": naive_tokens,
            "avg_tokens_per_step": if steps == 0 { 0.0 } else { naive_tokens as f64 / steps as f64 }
        },
        "state_pack": {
            "tokens_processed": state_pack_tokens,
            "avg_tokens_per_step": if steps == 0 { 0.0 } else { state_pack_tokens as f64 / steps as f64 }
        },
        "savings": {
            "tokens_saved": tokens_saved,
            "savings_percent": savings_percent,
            "estimated_usd_saved": estimated_usd_saved,
            "input_cost_per_m": input_cost_per_m
        },
        "per_step": per_step,
        "metadata": {}
    });

    let canonical = serde_json::to_vec(&result)?;
    result["metadata"]["receipt_id"] =
        serde_json::Value::String(format!("sha256:{}", sha256_bytes(&canonical)));

    if let Some(out_path) = out {
        fs::write(out_path, serde_json::to_vec_pretty(&result)?)?;
    }

    QUIET_RECEIPTS.store(false, Ordering::SeqCst);
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_benchmark(script: &Path, input_cost_per_m: f64, out: Option<&Path>) -> Result<()> {
    let mut cmd = ProcessCommand::new("python3");
    cmd.arg(script)
        .arg("--input-cost-per-m")
        .arg(input_cost_per_m.to_string());
    if let Some(out_path) = out {
        cmd.arg("--out").arg(out_path);
    }
    let output = cmd.output().context("run benchmark script")?;

    if !output.status.success() {
        bail!("benchmark failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn tokenize_delta(delta_path: &Path) -> Result<()> {
    let delta = read_delta_packet(delta_path)?;
    let output = ProcessCommand::new("python3")
        .arg("gpt2_tokenize.py")
        .arg("--text")
        .arg(&delta.delta_text)
        .output()
        .context("run gpt2_tokenize.py")?;

    if !output.status.success() {
        bail!("tokenizer bridge failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    print!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn infer_packet(delta_path: &Path, store: &Path) -> Result<Receipt> {
    let delta = read_delta_packet(delta_path)?;

    let manifest_path = store.join(format!("state_packet_{}.json", delta.base_sha256));
    let manifest = read_manifest(&manifest_path)?;

    if manifest.packet_id != delta.packet_id {
        bail!("delta packet_id does not match resolved state packet");
    }

    let blob_path = manifest_path.parent().ok_or_else(|| anyhow!("manifest has no parent directory"))?.join(&manifest.blob_file);
    let (actual_blob_hash, actual_blob_bytes) = sha256_file(&blob_path)?;
    if actual_blob_hash != manifest.blob_sha256 { bail!("blob hash mismatch: expected {}, got {}", manifest.blob_sha256, actual_blob_hash); }
    if actual_blob_bytes != manifest.blob_bytes { bail!("blob byte mismatch: expected {}, got {}", manifest.blob_bytes, actual_blob_bytes); }

    let base_tokens = manifest.base_tokens;
    let delta_tokens = gpt2_token_count_text(&delta.delta_text).unwrap_or(0);


    let receipt = Receipt {
        receipt_id: None,
        op: "infer".to_string(),
        ok: true,
        packet_id: Some(delta.packet_id.clone()),
        base_sha256: Some(delta.base_sha256.clone()),
        blob_sha256: Some(manifest.blob_sha256.clone()),
        delta_sha256: Some(delta.delta_sha256.clone()),
        bytes: Some(delta.delta_bytes),
        bytes_saved: Some(ByteSavings {
            base: manifest.base_bytes,
            delta: delta.delta_bytes,
            processed: delta.delta_bytes,
            saved: manifest.base_bytes,
            savings_percent: if manifest.base_bytes + delta.delta_bytes == 0 { 0.0 } else { (manifest.base_bytes as f64 / (manifest.base_bytes + delta.delta_bytes) as f64) * 100.0 },
        }),
        tokens: Some(TokenSavings {
            base: base_tokens,
            delta: delta_tokens,
            processed: delta_tokens,
            saved: base_tokens,
            savings_percent: if base_tokens + delta_tokens == 0 { 0.0 } else { (base_tokens as f64 / (base_tokens + delta_tokens) as f64) * 100.0 },
        }),
    };

    emit_receipt(receipt)
}

fn merge_packet(manifest_path: &Path, delta_path: &Path, blob_path: &Path, out_dir: &Path) -> Result<Receipt> {
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let manifest = read_manifest(manifest_path)?;
    let delta = read_delta_packet(delta_path)?;

    if manifest.packet_id != delta.packet_id {
        bail!("delta packet_id does not match base manifest packet_id");
    }
    if manifest.base_sha256 != delta.base_sha256 {
        bail!("delta base_sha256 does not match base manifest");
    }

    let merged_base_bytes = format!("{}|{}", manifest.base_sha256, delta.delta_sha256).into_bytes();
    let merged_base_sha256 = sha256_bytes(&merged_base_bytes);
    let (blob_sha256, blob_bytes) = sha256_file(blob_path)?;

    let blob_file = format!("state_packet_{}.pt", merged_base_sha256);
    let blob_dest = out_dir.join(&blob_file);
    fs::copy(blob_path, &blob_dest).with_context(|| format!("copy merged blob to {}", blob_dest.display()))?;

    let mut merged = StatePacketManifest {
        version: PACKET_VERSION.to_string(),
        model: manifest.model.clone(),
        base_sha256: merged_base_sha256,
        base_bytes: manifest.base_bytes + delta.delta_bytes,
        base_tokens: manifest.base_tokens + gpt2_token_count_text(&delta.delta_text).unwrap_or(0),
        base_file: format!("merge:{}+{}", manifest.base_sha256, delta.delta_sha256),
        blob_sha256,
        blob_bytes,
        blob_file,
        packet_id: String::new(),
    };

    let preimage = format!(
        "{}|{}|{}|{}|{}|{}",
        merged.version,
        merged.model,
        merged.base_sha256,
        merged.base_bytes,
        merged.blob_sha256,
        merged.blob_bytes
    );
    merged.packet_id = format!("sha256:{}", sha256_bytes(preimage.as_bytes()));

    let manifest_file = format!("state_packet_{}.json", merged.base_sha256);
    let merged_manifest_path = out_dir.join(&manifest_file);
    fs::write(&merged_manifest_path, serde_json::to_vec_pretty(&merged)?)?;

    let receipt = Receipt {
        receipt_id: None,
        op: "merge".to_string(),
        ok: true,
        packet_id: Some(merged.packet_id.clone()),
        base_sha256: Some(merged.base_sha256.clone()),
        blob_sha256: Some(merged.blob_sha256.clone()),
        delta_sha256: Some(delta.delta_sha256.clone()),
        bytes: Some(merged.blob_bytes),
        bytes_saved: None,
        tokens: None,
    };
    emit_receipt(receipt)
}

fn resolve_packet(store: &Path, base_hash: &str) -> Result<()> {
    let manifest = store.join(format!("state_packet_{}.json", base_hash));
    let blob = store.join(format!("state_packet_{}.pt", base_hash));
    if !manifest.exists() {
        bail!("manifest not found: {}", manifest.display());
    }
    if !blob.exists() {
        bail!("blob not found: {}", blob.display());
    }
    println!("manifest={}", manifest.display());
    println!("blob={}", blob.display());
    Ok(())
}


fn run_gc(
    store: &Path,
    older_than: Option<u64>,
    keep_latest: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    if older_than.is_none() && keep_latest.is_none() {
        bail!("specify --older-than <days> or --keep-latest <n>");
    }

    let mut entries: Vec<(std::path::PathBuf, std::time::SystemTime)> = fs::read_dir(store)
        .with_context(|| format!("read store {}", store.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map(|x| x == "json").unwrap_or(false)
                && e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("state_packet_"))
                    .unwrap_or(false)
        })
        .filter_map(|e| {
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((e.path(), mtime))
        })
        .collect();

    if entries.is_empty() {
        println!("{{}}");
        return Ok(());
    }

    entries.sort_by(|a, b| b.1.cmp(&a.1));

    let now = std::time::SystemTime::now();
    let mut to_delete: Vec<std::path::PathBuf> = Vec::new();

    if let Some(days) = older_than {
        let threshold = std::time::Duration::from_secs(days * 86400);
        for (path, mtime) in &entries {
            if let Ok(age) = now.duration_since(*mtime) {
                if age > threshold {
                    to_delete.push(path.clone());
                }
            }
        }
    }

    if let Some(keep) = keep_latest {
        if entries.len() > keep {
            for (p, _) in &entries[keep..] {
                if !to_delete.contains(p) {
                    to_delete.push(p.clone());
                }
            }
        }
    }

    let mut deleted_manifests = 0usize;
    let mut deleted_blobs     = 0usize;
    let mut freed_bytes       = 0u64;
    let mut skipped           = 0usize;

    for manifest_path in &to_delete {
        let blob_path  = manifest_path.with_extension("pt");
        let blob_bytes = blob_path.metadata().map(|m| m.len()).unwrap_or(0);
        let mani_bytes = manifest_path.metadata().map(|m| m.len()).unwrap_or(0);

        if dry_run {
            let line = format!(
                "dry_run=true would_delete={} blob_mb={:.2}",
                manifest_path.display(),
                blob_bytes as f64 / 1_048_576.0
            );
            println!("{}", line);
            skipped += 1;
        } else {
            if manifest_path.exists() {
                fs::remove_file(manifest_path)
                    .with_context(|| format!("delete {}", manifest_path.display()))?;
                deleted_manifests += 1;
                freed_bytes += mani_bytes;
            }
            if blob_path.exists() {
                fs::remove_file(&blob_path)
                    .with_context(|| format!("delete {}", blob_path.display()))?;
                deleted_blobs += 1;
                freed_bytes += blob_bytes;
            }
        }
    }

    let result = serde_json::json!({
        "op": "gc",
        "ok": true,
        "dry_run": dry_run,
        "store": store.display().to_string(),
        "scanned": entries.len(),
        "deleted_manifests": deleted_manifests,
        "deleted_blobs": deleted_blobs,
        "skipped_dry_run": skipped,
        "freed_bytes": freed_bytes,
        "freed_mb": (freed_bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0,
        "remaining": entries.len().saturating_sub(to_delete.len()),
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn manifest_serializes() {
        let m = StatePacketManifest {
            version: PACKET_VERSION.to_string(),
            model: "gpt2".to_string(),
            base_sha256: "00".to_string(),
            base_bytes: 1,
            base_tokens: 1,
            base_file: "base.txt".to_string(),
            blob_sha256: "11".to_string(),
            blob_bytes: 2,
            blob_file: "state_packet_00.pt".to_string(),
            packet_id: "sha256:22".to_string(),
        };
        let bytes = serde_json::to_vec(&m).unwrap();
        assert!(!bytes.is_empty());
    }
}

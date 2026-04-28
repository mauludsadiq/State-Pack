use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command as ProcessCommand;
use std::io::Read;
use std::path::{Path, PathBuf};

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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct StatePacketManifest {
    version: String,
    model: String,
    base_sha256: String,
    base_bytes: u64,
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
    tokens: Option<TokenSavings>,
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
        Command::Delta { manifest, delta, out } => create_delta(&manifest, &delta, &out),
        Command::Resolve { store, base_hash } => resolve_packet(&store, &base_hash),
        Command::Infer { delta, store } => infer_packet(&delta, &store),
        Command::Tokenize { delta } => tokenize_delta(&delta),
    }
}

fn emit_receipt(mut receipt: Receipt) -> Result<()> {
    receipt.receipt_id = None;
    let canonical = serde_json::to_vec(&receipt)?;
    receipt.receipt_id = Some(format!("sha256:{}", sha256_bytes(&canonical)));
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
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
        tokens: None,
    };
    emit_receipt(receipt)?;
    Ok(())
}

fn create_delta(manifest_path: &Path, delta_path: &Path, out_path: &Path) -> Result<()> {
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
        tokens: None,
    };
    emit_receipt(receipt)?;
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

fn infer_packet(delta_path: &Path, store: &Path) -> Result<()> {
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

    let receipt = Receipt {
        receipt_id: None,
        op: "infer".to_string(),
        ok: true,
        packet_id: Some(delta.packet_id.clone()),
        base_sha256: Some(delta.base_sha256.clone()),
        blob_sha256: Some(manifest.blob_sha256.clone()),
        delta_sha256: Some(delta.delta_sha256.clone()),
        bytes: Some(delta.delta_bytes),
        tokens: Some(TokenSavings {
            base: manifest.base_bytes,
            delta: delta.delta_bytes,
            processed: delta.delta_bytes,
            saved: manifest.base_bytes,
            savings_percent: if manifest.base_bytes + delta.delta_bytes == 0 { 0.0 } else { (manifest.base_bytes as f64 / (manifest.base_bytes + delta.delta_bytes) as f64) * 100.0 },
        }),
    };

    emit_receipt(receipt)?;
    Ok(())
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
            blob_sha256: "11".to_string(),
            blob_bytes: 2,
            blob_file: "state_packet_00.pt".to_string(),
            packet_id: "sha256:22".to_string(),
        };
        let bytes = serde_json::to_vec(&m).unwrap();
        assert!(!bytes.is_empty());
    }
}

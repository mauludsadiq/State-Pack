//! state-pack-server
//! ==================
//! Rust HTTP server implementing the State Pack stateless inference protocol.
//!
//! Routes:
//!   POST /states            - register a state packet (manifest + blob already on disk)
//!   POST /infer             - verify state + delta, emit receipt
//!   POST /merge             - merge delta into base, emit receipt
//!   POST /compact           - fold delta chain into fresh base
//!   GET  /states/:hash      - inspect a cached state
//!   GET  /health            - server status
//!
//! The server owns NO inference logic. It owns:
//!   - content-addressed blob store (SHA-256)
//!   - receipt generation
//!   - manifest read/write
//!   - HTTP routing
//!
//! Inference is handled by the Python sidecar (stateless_server.py).
//! This server handles everything except running the model.

use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PACKET_VERSION: &str = "state-pack-v0.1";

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    store: PathBuf,
    model: String,
    request_count: Arc<RwLock<u64>>,
}

impl AppState {
    fn new(store: PathBuf, model: String) -> Self {
        fs::create_dir_all(&store).expect("create store dir");
        Self {
            store,
            model,
            request_count: Arc::new(RwLock::new(0)),
        }
    }

    fn manifest_path(&self, hash: &str) -> PathBuf {
        self.store.join(format!("state_packet_{}.json", hash))
    }

    fn blob_path(&self, hash: &str) -> PathBuf {
        self.store.join(format!("state_packet_{}.pt", hash))
    }

    fn inc(&self) {
        *self.request_count.write().unwrap() += 1;
    }

    fn count(&self) -> u64 {
        *self.request_count.read().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct StateManifest {
    version:     String,
    model:       String,
    base_sha256: String,
    base_bytes:  u64,
    base_tokens: u64,
    base_file:   String,
    blob_sha256: String,
    blob_bytes:  u64,
    blob_file:   String,
    packet_id:   String,
}

#[derive(Serialize, Debug)]
struct Receipt {
    receipt_id:  String,
    op:          String,
    ok:          bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_hash:  Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    new_state_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    packet_id:   Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blob_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes:       Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens:      Option<TokenInfo>,
    timestamp:   u64,
}

#[derive(Serialize, Debug)]
struct TokenInfo {
    base:         u64,
    delta:        u64,
    saved:        u64,
    savings_pct:  f64,
}

// ---------------------------------------------------------------------------
// Request bodies
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RegisterRequest {
    state_hash:  String,
    base_tokens: u64,
    base_text:   Option<String>,
}

#[derive(Deserialize)]
struct InferRequest {
    state_hash:  String,
    delta_text:  String,
    delta_tokens: u64,
}

#[derive(Deserialize)]
struct MergeRequest {
    state_hash:      String,
    delta_text:      String,
    new_state_hash:  String,
    new_blob_path:   String,
    new_base_tokens: u64,
}

#[derive(Deserialize)]
struct CompactRequest {
    state_hash:      String,
    new_state_hash:  String,
    new_blob_path:   String,
    new_base_tokens: u64,
    steps_folded:    u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sha256_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn sha256_file(path: &std::path::Path) -> Result<(String, u64)> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
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

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn make_receipt(op: &str) -> Receipt {
    Receipt {
        receipt_id:     String::new(),
        op:             op.to_string(),
        ok:             true,
        state_hash:     None,
        new_state_hash: None,
        packet_id:      None,
        blob_sha256:    None,
        bytes:          None,
        tokens:         None,
        timestamp:      now_unix(),
    }
}

fn seal_receipt(r: &mut Receipt) {
    let canonical = serde_json::to_vec(&*r).unwrap_or_default();
    r.receipt_id = format!("sha256:{}", sha256_bytes(&canonical));
}

fn load_manifest(state: &AppState, hash: &str) -> Result<StateManifest> {
    let path = state.manifest_path(hash);
    let bytes = fs::read(&path)
        .with_context(|| format!("manifest not found: {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse manifest")
}

fn packet_id(m: &StateManifest) -> String {
    let preimage = format!(
        "{}|{}|{}|{}|{}|{}",
        m.version, m.model, m.base_sha256,
        m.base_bytes, m.blob_sha256, m.blob_bytes
    );
    format!("sha256:{}", sha256_bytes(preimage.as_bytes()))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

struct AppError(anyhow::Error, StatusCode);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.0.to_string(), "ok": false });
        (self.1, Json(body)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into(), StatusCode::INTERNAL_SERVER_ERROR)
    }
}

type AR<T> = std::result::Result<T, AppError>;

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// GET /health
async fn health(State(app): State<AppState>) -> impl IntoResponse {
    let manifest_count = fs::read_dir(&app.store)
        .map(|d| d.filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
            .count())
        .unwrap_or(0);

    Json(serde_json::json!({
        "ok": true,
        "version": "0.2.0",
        "model": app.model,
        "store": app.store.display().to_string(),
        "states_cached": manifest_count,
        "requests_served": app.count(),
    }))
}

/// POST /states — register a state packet that Python has already written to disk
async fn register_state(
    State(app): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> AR<impl IntoResponse> {
    app.inc();

    let blob_path = app.blob_path(&req.state_hash);
    let manifest_path = app.manifest_path(&req.state_hash);

    if !blob_path.exists() {
        return Err(AppError(
            anyhow!("blob not found: {}", blob_path.display()),
            StatusCode::NOT_FOUND,
        ));
    }

    let (blob_sha256, blob_bytes) = sha256_file(&blob_path)?;
    let base_text = req.base_text.as_deref().unwrap_or(&req.state_hash);
    let base_bytes = base_text.len() as u64;

    let mut manifest = StateManifest {
        version:     PACKET_VERSION.to_string(),
        model:       app.model.clone(),
        base_sha256: req.state_hash.clone(),
        base_bytes,
        base_tokens: req.base_tokens,
        base_file:   base_text.chars().take(64).collect(),
        blob_sha256: blob_sha256.clone(),
        blob_bytes,
        blob_file:   blob_path.file_name().unwrap().to_string_lossy().to_string(),
        packet_id:   String::new(),
    };
    manifest.packet_id = packet_id(&manifest);

    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    let mut r = make_receipt("register");
    r.state_hash  = Some(req.state_hash.clone());
    r.packet_id   = Some(manifest.packet_id.clone());
    r.blob_sha256 = Some(blob_sha256);
    r.bytes       = Some(blob_bytes);
    seal_receipt(&mut r);

    info!("register state={} tokens={}", &req.state_hash[..16], req.base_tokens);
    Ok(Json(r))
}

/// GET /states/:hash
async fn get_state(
    State(app): State<AppState>,
    Path(hash): Path<String>,
) -> AR<impl IntoResponse> {
    app.inc();
    let manifest = load_manifest(&app, &hash).map_err(|e| {
        AppError(e, StatusCode::NOT_FOUND)
    })?;
    let blob_path = app.blob_path(&hash);
    let hot = blob_path.exists();

    Ok(Json(serde_json::json!({
        "ok": true,
        "state_hash": hash,
        "tokens": manifest.base_tokens,
        "bytes": manifest.blob_bytes,
        "model": manifest.model,
        "packet_id": manifest.packet_id,
        "hot": hot,
    })))
}

/// POST /infer — verify state integrity + emit receipt (no model inference)
async fn infer(
    State(app): State<AppState>,
    Json(req): Json<InferRequest>,
) -> AR<impl IntoResponse> {
    app.inc();

    let manifest = load_manifest(&app, &req.state_hash).map_err(|e| {
        AppError(e, StatusCode::NOT_FOUND)
    })?;

    // Verify blob integrity
    let blob_path = app.blob_path(&req.state_hash);
    if !blob_path.exists() {
        return Err(AppError(
            anyhow!("blob missing for state {}", &req.state_hash[..16]),
            StatusCode::NOT_FOUND,
        ));
    }
    let (actual_hash, _) = sha256_file(&blob_path)?;
    if actual_hash != manifest.blob_sha256 {
        return Err(AppError(
            anyhow!("blob integrity failure: hash mismatch"),
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }

    let delta_sha256 = sha256_bytes(req.delta_text.as_bytes());
    let new_state_hash = sha256_bytes(
        format!("{}|{}", req.state_hash, delta_sha256).as_bytes()
    );

    let base_tokens  = manifest.base_tokens;
    let delta_tokens = req.delta_tokens;
    let saved        = base_tokens;
    let savings_pct  = if base_tokens + delta_tokens > 0 {
        base_tokens as f64 / (base_tokens + delta_tokens) as f64 * 100.0
    } else { 0.0 };

    let mut r = make_receipt("infer");
    r.state_hash     = Some(req.state_hash.clone());
    r.new_state_hash = Some(new_state_hash.clone());
    r.packet_id      = Some(manifest.packet_id.clone());
    r.blob_sha256    = Some(manifest.blob_sha256.clone());
    r.tokens = Some(TokenInfo { base: base_tokens, delta: delta_tokens, saved, savings_pct });
    seal_receipt(&mut r);

    info!("infer state={} delta_tokens={} savings={:.1}%",
          &req.state_hash[..16], delta_tokens, savings_pct);
    Ok(Json(r))
}

/// POST /merge — register merged state
async fn merge(
    State(app): State<AppState>,
    Json(req): Json<MergeRequest>,
) -> AR<impl IntoResponse> {
    app.inc();

    let base_manifest = load_manifest(&app, &req.state_hash).map_err(|e| {
        AppError(e, StatusCode::NOT_FOUND)
    })?;

    let new_blob_src = std::path::Path::new(&req.new_blob_path);
    let new_blob_dst = app.blob_path(&req.new_state_hash);

    if new_blob_src != new_blob_dst && new_blob_src.exists() {
        fs::copy(new_blob_src, &new_blob_dst)?;
    }

    let (blob_sha256, blob_bytes) = sha256_file(&new_blob_dst)?;
    let delta_sha256 = sha256_bytes(req.delta_text.as_bytes());

    let mut merged = StateManifest {
        version:     PACKET_VERSION.to_string(),
        model:       app.model.clone(),
        base_sha256: req.new_state_hash.clone(),
        base_bytes:  base_manifest.base_bytes + req.delta_text.len() as u64,
        base_tokens: req.new_base_tokens,
        base_file:   format!("merge:{}+{}", &req.state_hash[..16], &delta_sha256[..16]),
        blob_sha256: blob_sha256.clone(),
        blob_bytes,
        blob_file:   new_blob_dst.file_name().unwrap().to_string_lossy().to_string(),
        packet_id:   String::new(),
    };
    merged.packet_id = packet_id(&merged);

    let manifest_path = app.manifest_path(&req.new_state_hash);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&merged)?)?;

    let mut r = make_receipt("merge");
    r.state_hash     = Some(req.state_hash.clone());
    r.new_state_hash = Some(req.new_state_hash.clone());
    r.packet_id      = Some(merged.packet_id);
    r.blob_sha256    = Some(blob_sha256);
    r.bytes          = Some(blob_bytes);
    seal_receipt(&mut r);

    info!("merge {} -> {}", &req.state_hash[..16], &req.new_state_hash[..16]);
    Ok(Json(r))
}

/// POST /compact — register compacted state
async fn compact(
    State(app): State<AppState>,
    Json(req): Json<CompactRequest>,
) -> AR<impl IntoResponse> {
    app.inc();

    let new_blob_src = std::path::Path::new(&req.new_blob_path);
    let new_blob_dst = app.blob_path(&req.new_state_hash);

    if new_blob_src != new_blob_dst && new_blob_src.exists() {
        fs::copy(new_blob_src, &new_blob_dst)?;
    }

    let (blob_sha256, blob_bytes) = sha256_file(&new_blob_dst)?;

    let mut compacted = StateManifest {
        version:     PACKET_VERSION.to_string(),
        model:       app.model.clone(),
        base_sha256: req.new_state_hash.clone(),
        base_bytes:  blob_bytes,
        base_tokens: req.new_base_tokens,
        base_file:   format!("compact:{}+{}steps", &req.state_hash[..16], req.steps_folded),
        blob_sha256: blob_sha256.clone(),
        blob_bytes,
        blob_file:   new_blob_dst.file_name().unwrap().to_string_lossy().to_string(),
        packet_id:   String::new(),
    };
    compacted.packet_id = packet_id(&compacted);

    let manifest_path = app.manifest_path(&req.new_state_hash);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&compacted)?)?;

    let mut r = make_receipt("compact");
    r.state_hash     = Some(req.state_hash.clone());
    r.new_state_hash = Some(req.new_state_hash.clone());
    r.packet_id      = Some(compacted.packet_id);
    r.blob_sha256    = Some(blob_sha256);
    r.bytes          = Some(blob_bytes);
    seal_receipt(&mut r);

    info!("compact {} steps_folded={} -> {}",
          &req.state_hash[..16], req.steps_folded, &req.new_state_hash[..16]);
    Ok(Json(r))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("state_pack_server=info".parse()?)
                .add_directive("tower_http=info".parse()?)
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let store = args.iter().position(|a| a == "--store")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("packets"));
    let model = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "gpt2".to_string());
    let host = args.iter().position(|a| a == "--host")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8003);

    let state = AppState::new(store.clone(), model.clone());

    let app = Router::new()
        .route("/health",         get(health))
        .route("/states",         post(register_state))
        .route("/states/:hash",   get(get_state))
        .route("/infer",          post(infer))
        .route("/merge",          post(merge))
        .route("/compact",        post(compact))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    info!("State Pack Rust server v0.2.0");
    info!("Store: {}", store.display());
    info!("Model: {}", model);
    info!("Listening on http://{}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

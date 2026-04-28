from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import hashlib
import os

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PACKET_PATH = "llama_state_packet.pt"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

seed = (
    "The legal memorandum analyzes indemnification, liability allocation, notice requirements, "
    "survival clauses, governing law, dispute resolution, limitation of damages, and procedural duties. "
)

base = seed * 20
delta = " The next issue is waiver, estoppel, and material breach."

def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0

def full_run():
    x = tok(base + delta, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        return model(**x, use_cache=True), x["input_ids"].shape[1]

def save_packet():
    x = tok(base, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(**x, use_cache=True)
    packet = {
        "model": MODEL_ID,
        "base_sha256": sha256_text(base),
        "base_tokens": x["input_ids"].shape[1],
        "past_key_values": out.past_key_values,
    }
    torch.save(packet, PACKET_PATH)
    return packet

def load_packet():
    return torch.load(PACKET_PATH, map_location="cpu", weights_only=False)

def delta_run(packet):
    assert packet["base_sha256"] == sha256_text(base)
    dx = tok(delta, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(
            input_ids=dx["input_ids"],
            past_key_values=packet["past_key_values"],
            use_cache=True,
        )
    return out, dx["input_ids"].shape[1]

_, save_s = timed(save_packet)
packet_size = os.path.getsize(PACKET_PATH)

packet, load_s = timed(load_packet)
full_pair, full_s = timed(full_run)
delta_pair, delta_s = timed(lambda: delta_run(packet))

full_out, full_tokens = full_pair
delta_out, delta_tokens = delta_pair

diff = torch.max(torch.abs(full_out.logits[:, -1, :] - delta_out.logits[:, -1, :])).item()

print("model:", MODEL_ID)
print("packet_path:", PACKET_PATH)
print("packet_bytes:", packet_size)
print("base_tokens:", packet["base_tokens"])
print("delta_tokens:", delta_tokens)
print("full_tokens:", full_tokens)
print("save_seconds:", round(save_s, 6))
print("load_seconds:", round(load_s, 6))
print("full_seconds:", round(full_s, 6))
print("delta_seconds:", round(delta_s, 6))
print("compute_speedup_excluding_load:", round(full_s / delta_s, 3))
print("end_to_end_speedup_including_load:", round(full_s / (load_s + delta_s), 3))
print("max_abs_logit_diff:", diff)

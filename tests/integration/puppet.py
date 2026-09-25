"""Deterministic "puppet" models for integration tests against a real Ollama server.

A puppet is a tiny GGUF model whose weights are hand-set so that its output is
scripted instead of learned:

* token embeddings are one-hot and every attention/feed-forward weight is zero,
  so the model's next token depends only on the current token (a lookup table);
* the chat template ends a normal turn with a newline token and a turn that
  follows a tool result with a dedicated marker token;
* the lookup table maps newline -> FIRST reply token and marker -> AFTER_TOOL
  reply token, each a single user-defined token carrying the whole reply text,
  then end-of-turn.

So a puppet answers the first model call with a scripted reply — typically a
tool call in the ``<tool_call>{json}</tool_call>`` format its template declares —
and answers the call after the tool result with a scripted text. That exercises
the real server end to end: template rendering with tools, llama.cpp inference,
Ollama's own tool-call parsing, streaming and model lifecycle. It proves the
plumbing, not intelligence; capability tests use real models (see
``test_live_capabilities.py``).

Requires the ``gguf`` and ``numpy`` packages (``pip install gguf numpy``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

N = 272              # vocabulary size == embedding width (one-hot embeddings)
HEADS = 17           # 272 / 17 = 16-dimensional heads
FF = 32

IM_START, IM_END, ENDOFTEXT, AFTER_TOOL, FIRST, SECOND = "<|im_start|>", "<|im_end|>", "<|endoftext|>", \
    "<|after_tool|>", "<|puppet_first|>", "<|puppet_second|>"

TEMPLATE = """{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}
Tools:
{{- range .Tools }}
{{ json . }}
{{- end }}
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ if $last }}<|im_start|>assistant
{{ end }}
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}{{ end }}
{{- if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>tool
{{ .Content }}<|im_end|>
{{ if $last }}<|im_start|>assistant<|after_tool|>{{ end }}
{{- end }}
{{- end }}"""


def tool_call(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


@dataclass
class Script:
    first: str                 # reply to a user turn (e.g. a tool call)
    after_tool: str = "Done."  # reply after a tool result


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE mapping of bytes to printable characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build_gguf(path: Path, script: Script) -> Path:
    import gguf
    import numpy as np

    mapping = _bytes_to_unicode()
    tokens = [mapping[b] for b in range(256)] + ["Ġt"]
    types = [gguf.TokenType.NORMAL] * 257
    specials = [ENDOFTEXT, IM_START, IM_END]
    tokens += specials
    types += [gguf.TokenType.CONTROL] * 3
    replies = [script.first] if script.after_tool == script.first else [script.first, script.after_tool]
    tokens += [AFTER_TOOL] + replies            # vocabulary entries must be unique
    types += [gguf.TokenType.USER_DEFINED] * (1 + len(replies))
    while len(tokens) < N:
        tokens.append(f"<|pad_{len(tokens)}|>")
        types.append(gguf.TokenType.CONTROL)
    ids = {t: i for i, t in enumerate(tokens)}
    newline = ids[mapping[ord("\n")]]
    eos = ids[IM_END]

    out = np.zeros((N, N), dtype=np.float32)          # out[next, current]
    out[eos, :] = 10.0                                  # default: end the turn
    out[:, newline] = 0.0
    out[ids[script.first], newline] = 10.0
    out[:, ids[AFTER_TOOL]] = 0.0
    out[ids[script.after_tool], ids[AFTER_TOOL]] = 10.0

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("jarvis-puppet")
    writer.add_context_length(8192)
    writer.add_embedding_length(N)
    writer.add_block_count(1)
    writer.add_feed_forward_length(FF)
    writer.add_rope_dimension_count(N // HEADS)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(N)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(["Ġ t"])
    writer.add_bos_token_id(ids[ENDOFTEXT])
    writer.add_eos_token_id(eos)
    writer.add_add_bos_token(False)
    ones, zeros = np.ones(N, dtype=np.float32), np.zeros
    writer.add_tensor("token_embd.weight", np.eye(N, dtype=np.float32))
    writer.add_tensor("output_norm.weight", ones)
    writer.add_tensor("output.weight", out)
    writer.add_tensor("blk.0.attn_norm.weight", ones)
    for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
        writer.add_tensor(f"blk.0.{name}.weight", zeros((N, N), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_norm.weight", ones)
    writer.add_tensor("blk.0.ffn_gate.weight", zeros((FF, N), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up.weight", zeros((FF, N), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", zeros((N, FF), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def create_puppet(base_url: str, name: str, script: Script, workdir: Path) -> str:
    """Build the GGUF and register it with Ollama through its HTTP API (no CLI needed)."""
    path = build_gguf(workdir / f"{name.replace(':', '_')}.gguf", script)
    data = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    with httpx.Client(base_url=base_url, timeout=120, trust_env=False) as client:
        resp = client.post(f"/api/blobs/{digest}", content=data)
        resp.raise_for_status()
        resp = client.post("/api/create", json={
            "model": name, "files": {path.name: digest}, "template": TEMPLATE, "stream": False,
            "parameters": {"stop": [IM_END], "temperature": 0, "num_predict": 64}})
        resp.raise_for_status()
    return name


# -- vision puppets (Phase 4) -----------------------------------------------------------------------------------------
#
# A vision puppet is a puppet language model plus a tiny CLIP projector (a real "mmproj" GGUF with one vision layer
# and an MLP projector into the puppet's embedding space). Ollama then reports the "vision" capability, accepts images,
# decodes them and runs them through llama.cpp's real image encoder; the reply stays scripted because the puppet's next
# token depends only on the current one. It proves images reach the model through the real server (and that a broken
# image is refused there), not that anything is understood.

V_EMBD, V_HEADS, V_FF, V_PATCH, V_IMAGE = 16, 2, 32, 14, 28


def build_projector(path: Path) -> Path:
    import gguf
    import numpy as np

    rng = np.random.default_rng(0)

    def small(*shape: int) -> "np.ndarray":
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    positions = (V_IMAGE // V_PATCH) ** 2 + 1
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_type("mmproj")
    writer.add_name("jarvis-vision-puppet-projector")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.use_gelu", True)
    writer.add_string("clip.projector_type", "mlp")
    writer.add_uint32("clip.vision.embedding_length", V_EMBD)
    writer.add_uint32("clip.vision.feed_forward_length", V_FF)
    writer.add_uint32("clip.vision.block_count", 1)
    writer.add_uint32("clip.vision.attention.head_count", V_HEADS)
    writer.add_uint32("clip.vision.projection_dim", N)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    writer.add_uint32("clip.vision.image_size", V_IMAGE)
    writer.add_uint32("clip.vision.patch_size", V_PATCH)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    ones = np.ones(V_EMBD, dtype=np.float32)
    zeros = np.zeros(V_EMBD, dtype=np.float32)
    writer.add_tensor("v.patch_embd.weight", small(V_EMBD, 3, V_PATCH, V_PATCH))
    writer.add_tensor("v.class_embd", small(V_EMBD))
    writer.add_tensor("v.position_embd.weight", small(positions, V_EMBD))
    writer.add_tensor("v.pre_ln.weight", ones)
    writer.add_tensor("v.pre_ln.bias", zeros)
    for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
        writer.add_tensor(f"v.blk.0.{name}.weight", small(V_EMBD, V_EMBD))
        writer.add_tensor(f"v.blk.0.{name}.bias", zeros)
    for name in ("ln1", "ln2"):
        writer.add_tensor(f"v.blk.0.{name}.weight", ones)
        writer.add_tensor(f"v.blk.0.{name}.bias", zeros)
    writer.add_tensor("v.blk.0.ffn_up.weight", small(V_FF, V_EMBD))
    writer.add_tensor("v.blk.0.ffn_up.bias", np.zeros(V_FF, dtype=np.float32))
    writer.add_tensor("v.blk.0.ffn_down.weight", small(V_EMBD, V_FF))
    writer.add_tensor("v.blk.0.ffn_down.bias", zeros)
    writer.add_tensor("v.post_ln.weight", ones)
    writer.add_tensor("v.post_ln.bias", zeros)
    writer.add_tensor("mm.0.weight", small(N, V_EMBD))
    writer.add_tensor("mm.0.bias", np.zeros(N, dtype=np.float32))
    writer.add_tensor("mm.2.weight", small(N, N))
    writer.add_tensor("mm.2.bias", np.zeros(N, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def create_vision_puppet(base_url: str, name: str, script: Script, workdir: Path) -> str:
    """A puppet with a projector: Ollama lists it with the vision capability and runs images through it."""
    model = build_gguf(workdir / f"{name.replace(':', '_')}.gguf", script)
    projector = build_projector(workdir / f"{name.replace(':', '_')}-mmproj.gguf")
    files = {}
    with httpx.Client(base_url=base_url, timeout=120, trust_env=False) as client:
        for path in (model, projector):
            data = path.read_bytes()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            client.post(f"/api/blobs/{digest}", content=data).raise_for_status()
            files[path.name] = digest
        resp = client.post("/api/create", json={
            "model": name, "files": files, "template": TEMPLATE, "stream": False,
            "parameters": {"stop": [IM_END], "temperature": 0, "num_predict": 64}})
        resp.raise_for_status()
    return name

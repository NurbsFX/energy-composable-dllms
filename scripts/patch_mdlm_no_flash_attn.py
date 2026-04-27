"""Patch the MDLM-OWT custom modeling code to interoperate with HF Trainer.

The kuleshov-group/mdlm-owt repo ships `modeling_mdlm.py` written for a 2024
research codebase, not for the modern HF Trainer / dllm.core.trainers stack.
Five issues to fix; we resolve them all by rewriting the file in-place after
`huggingface_hub.snapshot_download` (idempotent):

1. flash_attn dependency. The two call sites (rotary embedding + variable-
   length attention) are replaced by pure-PyTorch (`rotate_half`,
   `F.scaled_dot_product_attention`). transformers'
   `dynamic_module_utils.check_imports` scans top-level imports statically,
   so we comment those out too.
2. `MDLM.forward()` does not accept HF-Trainer kwargs (e.g. `attention_mask`).
   We add `**kwargs` to absorb them.
3. The backbone forward calls `torch.zeros_like(sigma)` even when
   `time_conditioning=False` — but the trainer doesn't pass `timesteps`,
   so `sigma=None` and the call crashes. We accept `sigma=None`.
4. `MDLM.forward()` returns a bare Tensor when `return_dict=False`, but the
   trainer expects an `outputs.logits` attribute. We force `return_dict=True`.
5. `flash_attn-free` builds need the dynamic-modules cache invalidated; we
   sync our patched snapshot copy into it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def patch(path: Path) -> bool:
    src = path.read_text()
    if "# flash_attn imports patched out" in src:
        return False

    # 1. Comment out top-level flash_attn imports.
    src = src.replace(
        "import flash_attn\nimport flash_attn.layers.rotary\n",
        "# flash_attn imports patched out — using PyTorch SDPA fallback\n"
        "# import flash_attn\n# import flash_attn.layers.rotary\n",
    )
    if "# flash_attn imports patched out" not in src:
        raise RuntimeError(f"flash_attn import anchor not found in {path}")

    # 2. Replace rotary embedding with pure-torch implementation.
    old_rot = (
        "def apply_rotary_pos_emb(qkv, cos, sin):\n"
        "  cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]\n"
        "  sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]\n"
        "  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv,\n"
        "                                                        cos,\n"
        "                                                        sin)"
    )
    new_rot = (
        "def apply_rotary_pos_emb(qkv, cos, sin):\n"
        "  # qkv: [b, s, 3, h, d]; cos,sin: [1, s, 3, 1, d]\n"
        "  cos_h = cos[0, :, 0, 0, :cos.shape[-1] // 2]\n"
        "  sin_h = sin[0, :, 0, 0, :sin.shape[-1] // 2]\n"
        "  cos_full = torch.cat([cos_h, cos_h], dim=-1)[None, :, None, :]\n"
        "  sin_full = torch.cat([sin_h, sin_h], dim=-1)[None, :, None, :]\n"
        "  q = qkv[:, :, 0]\n"
        "  k = qkv[:, :, 1]\n"
        "  v = qkv[:, :, 2]\n"
        "  q_rot = q * cos_full + rotate_half(q) * sin_full\n"
        "  k_rot = k * cos_full + rotate_half(k) * sin_full\n"
        "  return torch.stack([q_rot, k_rot, v], dim=2)"
    )
    if old_rot not in src:
        raise RuntimeError(f"rotary patch anchor not found in {path}")
    src = src.replace(old_rot, new_rot)

    # 3. Replace varlen flash attention with SDPA.
    old_attn = (
        "    qkv = rearrange(qkv, 'b s ... -> (b s) ...')\n"
        "    if seqlens is None:\n"
        "      cu_seqlens = torch.arange(\n"
        "        0, (batch_size + 1) * seq_len, step=seq_len,\n"
        "        dtype=torch.int32, device=qkv.device)\n"
        "    else:\n"
        "      cu_seqlens = seqlens.cumsum(-1)\n"
        "    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(\n"
        "      qkv, cu_seqlens, seq_len, 0., causal=False)\n\n"
        "    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)"
    )
    new_attn = (
        "    # SDPA fallback — flash_attn unavailable; non-varlen path only (seqlens ignored)\n"
        "    q = qkv[:, :, 0].transpose(1, 2).contiguous()\n"
        "    k = qkv[:, :, 1].transpose(1, 2).contiguous()\n"
        "    v = qkv[:, :, 2].transpose(1, 2).contiguous()\n"
        "    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)\n"
        "    x = rearrange(x, 'b h s d -> b s (h d)')"
    )
    if old_attn not in src:
        raise RuntimeError(f"attention patch anchor not found in {path}")
    src = src.replace(old_attn, new_attn)

    # 4. Top-level forward must accept HF Trainer kwargs (attention_mask, etc.).
    old_top_fwd = (
        "  def forward(\n"
        "      self,\n"
        "      input_ids: torch.LongTensor = None,\n"
        "      timesteps: torch.FloatTensor = None,\n"
        "      output_hidden_states: typing.Optional[bool] = None,\n"
        "      return_dict: typing.Optional[bool] = None,\n"
        "  ) -> typing.Union["
    )
    new_top_fwd = (
        "  def forward(\n"
        "      self,\n"
        "      input_ids: torch.LongTensor = None,\n"
        "      timesteps: torch.FloatTensor = None,\n"
        "      output_hidden_states: typing.Optional[bool] = None,\n"
        "      return_dict: typing.Optional[bool] = None,\n"
        "      **kwargs,  # ignore HF Trainer extras (attention_mask, labels, ...)\n"
        "  ) -> typing.Union["
    )
    if old_top_fwd not in src:
        raise RuntimeError(f"top-level forward anchor not found in {path}")
    src = src.replace(old_top_fwd, new_top_fwd)

    # 5. Force return_dict=True so the trainer sees an `.logits` attribute.
    old_rd = (
        "    return_dict = return_dict \\\n"
        "      if return_dict is not None \\\n"
        "      else self.config.use_return_dict"
    )
    new_rd = (
        "    # Force HF-style dict output (trainer expects outputs.logits)\n    return_dict = True"
    )
    if old_rd not in src:
        raise RuntimeError(f"return_dict anchor not found in {path}")
    src = src.replace(old_rd, new_rd)

    # 6. Backbone forward must accept sigma=None (trainer doesn't pass timesteps).
    old_sigma = (
        "  def forward(self, indices, sigma,\n"
        "              output_hidden_states=False):\n"
        "    if not self.config.time_conditioning:\n"
        "      sigma = torch.zeros_like(sigma)"
    )
    new_sigma = (
        "  def forward(self, indices, sigma=None,\n"
        "              output_hidden_states=False):\n"
        "    if sigma is None or not self.config.time_conditioning:\n"
        "      sigma = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.float32)"
    )
    if old_sigma not in src:
        raise RuntimeError(f"sigma anchor not found in {path}")
    src = src.replace(old_sigma, new_sigma)

    path.write_text(src)
    return True


def sync_to_dynamic_modules_cache(snapshot_path: Path) -> None:
    """The dynamic-modules cache shadows the snapshot once a load succeeds.
    Mirror our patched snapshot copy into it so subsequent loads see the patches."""
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not cache_root.exists():
        return
    rev = snapshot_path.name
    for mirror in cache_root.rglob(f"{rev}/modeling_mdlm.py"):
        mirror.write_text(snapshot_path.read_text())
        print(f"synced cache: {mirror}")


def main() -> int:
    snapshot = Path(snapshot_download(repo_id="kuleshov-group/mdlm-owt"))
    target = snapshot / "modeling_mdlm.py"
    if not target.exists():
        print(f"ERROR: {target} missing", file=sys.stderr)
        return 1
    changed = patch(target)
    print(f"{'patched' if changed else 'already patched'}: {target}")
    sync_to_dynamic_modules_cache(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

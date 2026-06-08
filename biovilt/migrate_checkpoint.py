"""
migrate_checkpoint.py
=====================

Checkpoint migration utility for multi-prior K_max changes.

Use cases
---------
1. **Upstream → K_max-aware**: take a BioViL-T official checkpoint
   (`biovil_t_image_model_proj_size_128.pt`) which has
   `encoder.vit_pooler.type_embed` of shape `(2, 1, D)` and produce a
   K_max-aware checkpoint that loads strictly into our model when it
   constructs `MultiPriorTransformerPooler(K_max=...)`.

2. **K_max change between training runs**: take one of our own training
   checkpoints (`epoch_N.pt`) that was saved at K_max_old and expand its
   `image_encoder.multi_pooler.type_embed_multi` from `(K_old+1, 1, D)`
   to `(K_new+1, 1, D)` via copy-row-0 + replicate-row-1.

3. **K_max shrink** (less common, e.g. eval-only at smaller K): also
   handled — the leading rows are truncated.

Behavior
--------
The migration is *behavior-preserving at the lower K_max*: rows 0
through min(K_old, K_new) are copied verbatim; if K_new > K_old, the
new rows are filled with a *clone* of the prior row (so a fresh model
behaves like the original at small K and learns to differentiate the
extra prior slots later).

Both raw `state_dict` checkpoints and training checkpoints (dict with
`{model, optimizer, scheduler, epoch, ...}` keys) are supported.

CLI
---
    python biovilt/migrate_checkpoint.py \
        --in  /path/to/official_or_old.pt \
        --out /path/to/new.pt \
        --k-max 4

You can also drop the `--out` flag to print the migration plan without
writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


# ----------------------------------------------------------------------
# Migration core
# ----------------------------------------------------------------------
# Keys we may have to migrate. The first form is the "raw upstream"
# state dict (state_dict of MultiImageModel directly). The second form is
# what TempCXR.state_dict() produces when wrapped by our pipeline.
TYPE_EMBED_KEYS = [
    # raw upstream MultiImageModel state dict
    "encoder.vit_pooler.type_embed",
    # TempCXR full state dict (image_encoder.model.encoder.vit_pooler.type_embed)
    "image_encoder.model.encoder.vit_pooler.type_embed",
]

TYPE_EMBED_MULTI_KEYS = [
    # When loaded through TempCXR / BioViLTImageEncoder
    "image_encoder.multi_pooler.type_embed_multi",
    # If someone calls state_dict() on just the pooler
    "multi_pooler.type_embed_multi",
]


def _migrate_tensor(
    old: torch.Tensor,
    K_max_new: int,
) -> torch.Tensor:
    """Migrate a `(R, 1, D)` temporal embedding tensor to `(K_max_new+1, 1, D)`.

    Rows 0..min(R-1, K_max_new) are copied verbatim. If K_max_new+1 > R,
    the extra rows are filled with a clone of the LAST prior row
    (`old[1]` if R == 2; the last existing row otherwise) so the new
    slots behave identically to a "fresh prior" at init.
    """
    if old.dim() != 3 or old.shape[1] != 1:
        raise ValueError(
            f"Unexpected tensor shape {tuple(old.shape)}; "
            f"expected (R, 1, D)."
        )
    R, _, D = old.shape
    new_rows = K_max_new + 1
    new = torch.zeros(new_rows, 1, D, dtype=old.dtype, device=old.device)

    n_copy = min(R, new_rows)
    new[:n_copy] = old[:n_copy].clone()

    if new_rows > R:
        # Replicate the last "prior" row into the extra slots. For the
        # canonical R=2 upstream case this is row 1 (the prior row).
        fill_row = old[R - 1].clone() if R >= 2 else old[0].clone()
        new[R:] = fill_row.unsqueeze(0).expand(new_rows - R, -1, -1).clone()

    return new


def _unwrap_state_dict(
    obj,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Return the underlying tensor dict and a tag describing how it was found."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"], "train_ckpt"
    if isinstance(obj, dict):
        # Heuristic: if values are tensors, treat as a raw state dict.
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, "raw_state_dict"
    raise ValueError(
        "Could not interpret input as a state dict or training checkpoint."
    )


def migrate_state_dict(
    state: Dict[str, torch.Tensor],
    K_max_new: int,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Migrate a state dict in place; return (new_dict, log_lines)."""
    log: List[str] = []
    out = dict(state)

    # 1. Locate the type_embed_multi row count we should target.
    expected_rows = K_max_new + 1

    # 2. Migrate / inject `type_embed_multi`.
    multi_key = None
    for k in TYPE_EMBED_MULTI_KEYS:
        if k in out:
            multi_key = k
            break

    if multi_key is not None:
        old = out[multi_key]
        if old.shape[0] == expected_rows:
            log.append(
                f"  [skip] `{multi_key}` already has {expected_rows} rows."
            )
        else:
            new = _migrate_tensor(old, K_max_new)
            out[multi_key] = new
            log.append(
                f"  [migrate] `{multi_key}`: "
                f"{tuple(old.shape)} → {tuple(new.shape)}"
            )
    else:
        # We must FABRICATE type_embed_multi from upstream type_embed.
        src_key = None
        for k in TYPE_EMBED_KEYS:
            if k in out:
                src_key = k
                break
        if src_key is None:
            log.append(
                "  [no-op] No type_embed or type_embed_multi found; "
                "nothing to migrate."
            )
            return out, log

        old = out[src_key]
        new = _migrate_tensor(old, K_max_new)
        # Determine the multi key prefix from the src key
        if src_key.startswith("image_encoder."):
            tgt_key = "image_encoder.multi_pooler.type_embed_multi"
        else:
            tgt_key = "multi_pooler.type_embed_multi"
        out[tgt_key] = new
        log.append(
            f"  [fabricate] `{tgt_key}` ← migrated from `{src_key}`: "
            f"{tuple(old.shape)} → {tuple(new.shape)}"
        )

    return out, log


def migrate_checkpoint(
    in_path: Path,
    out_path: Path | None,
    K_max_new: int,
    verbose: bool = True,
) -> None:
    obj = torch.load(in_path, map_location="cpu", weights_only=False)
    state, tag = _unwrap_state_dict(obj)
    if verbose:
        print(f"[migrate] Detected input type: {tag}")
        print(f"[migrate] Target K_max = {K_max_new} → "
              f"type_embed_multi rows = {K_max_new + 1}")

    new_state, log = migrate_state_dict(state, K_max_new, verbose=verbose)
    if verbose:
        for line in log:
            print(line)

    if out_path is None:
        if verbose:
            print("[migrate] No --out given; this was a dry run.")
        return

    # Re-wrap if input was a training checkpoint.
    if tag == "train_ckpt":
        obj["model"] = new_state
        out_obj = obj
    else:
        out_obj = new_state

    torch.save(out_obj, out_path)
    if verbose:
        print(f"[migrate] Wrote → {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--in", dest="in_path", type=Path, required=True,
        help="Input checkpoint (upstream BioViL-T or a TempCXR train ckpt).",
    )
    p.add_argument(
        "--out", dest="out_path", type=Path, default=None,
        help="Output checkpoint path. If omitted, the migration is dry-run.",
    )
    p.add_argument(
        "--k-max", dest="k_max", type=int, required=True,
        help="Target K_max for the multi-prior model.",
    )
    p.add_argument(
        "--quiet", action="store_true",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.k_max < 1:
        print(f"--k-max must be >= 1, got {args.k_max}", file=sys.stderr)
        return 2
    migrate_checkpoint(
        in_path=args.in_path,
        out_path=args.out_path,
        K_max_new=args.k_max,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

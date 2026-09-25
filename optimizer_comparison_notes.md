# Prodigy vs Automagic — Optimizer Comparison Notes

Based on the actual implementations in this repo (`toolkit/optimizer.py`, `toolkit/optimizers/*.py`).
Written 2026-09-13 for cross-project comparison — temp file, safe to move/delete.

## Prodigy (2 variants in this repo)

- **Core idea**: D-Adaptation family. Learns a single scalar "D" (learning-rate
  multiplier) automatically during training, on top of an Adam-style update, so you
  don't have to hand-tune LR. `lr` param is a *multiplier* on D, not an absolute rate
  (conventionally left at 1.0).
- **`prodigy`** (plain): not custom — imports `Prodigy` from the pip package
  `prodigyopt`. Full fp32/bf16 optimizer state (`exp_avg`, `exp_avg_sq`, `p0`, `s`).
- **`prodigy8bit`** (`toolkit/optimizers/prodigy_8bit.py`): from-scratch reimplementation
  of the same algorithm, but stores per-parameter state (`s`, `p0`, `exp_avg`,
  `exp_avg_sq`) as int8-quantized `Auto8bitTensor` for memory savings, plus stochastic
  rounding support.
- **D estimation**: single scalar `d`, shared across the whole optimizer (not per
  parameter, not per group in the usual sense — one value for everything). Each step
  accumulates a numerator (grad · displacement from init) and denominator (EMA of
  scaled grad magnitude); `d` is updated as `d = max(d, d_hat)` then capped by
  `growth_rate` and `d_max` — **monotonically non-decreasing**.
- **Per-parameter state**: `exp_avg`, `exp_avg_sq` (Adam moments), `s` (EMA for D
  denominator), `p0` (snapshot of initial weight value) — 4 tensors per parameter
  (in 8-bit version, all quantized).
- **Key hyperparams**: `betas=(0.9,0.999)`, `beta3=sqrt(beta2)`, `eps`, `d0=1e-6`
  (initial D), `d_coef=1.0`, `growth_rate=inf`, `safeguard_warmup`, decoupled or
  coupled weight decay. Distributed-safe (all-reduces D numerator/denom across ranks).
- **Memory**: needs `p0` (a full copy of initial weights) in addition to normal Adam
  moments — noticeably heavier than plain AdamW unless 8-bit.
- **Once D converges**, behaves like a fixed-schedule AdamW; it never re-adapts per
  weight/layer — one global adaptive scale for the entire model.

## Automagic (4 versions — all custom, in `toolkit/optimizers/automagic*.py`)

Shared premise across all versions: **per-weight or per-group adaptive learning rate**,
based on whether consecutive update directions agree (→ increase LR) or flip (→
decrease LR) — a sign-agreement / "hot-cold" controller, *not* D-adaptation. Uses
Adafactor-style factored second moment (row/col) to keep the Adam-like variance term
cheap, except EXPERIMENT which uses a full per-element second moment.

### v1 — `automagic.py`
- **Granularity**: one learning rate **per weight element** (full tensor-shaped
  `lr_mask`, stored as int8 `Auto8bitTensor`).
- **Update rule**: additive — compare sign of current update vs. last step's sign
  (`last_polarity`) per element; agree → `lr += lr_bump` (default 1e-6), disagree →
  `lr -= lr_bump`; clamped to `[min_lr=1e-7, max_lr=1e-3]`.
- **State per parameter**: full-shape `lr_mask` (int8), full-shape `last_polarity`
  (bool), Adafactor row/col second moment — **highest memory footprint** of the four
  (a per-element LR is essentially a second full parameter tensor).
- Extra feature: "paramiter swapping" — randomly freezes a subset of params each step
  to save memory/compute.
- Not fused into backward (standard `.step()` loop).

### v2 — `automagic2.py`
- **Granularity**: one scalar learning rate **per parameter tensor** (e.g. one LR for
  an entire Linear weight matrix), not per element — explicit design change from v1 to
  cut memory.
- **Fused into the backward pass** via `register_post_accumulate_grad_hook`: each
  param updates and frees its gradient as soon as it's ready; `.step()` becomes a
  no-op, keeping peak VRAM low.
- **Update rule**: still additive. Computes `agreement` = fraction of elements whose
  current sign matches last step's sign; if `agreement >= 0.5`, `lr += lr_bump`, else
  `lr -= lr_bump`. Same clamp range as v1.
- **State per parameter**: scalar `lr`, full-shape `last_polarity` (still needed to
  compute agreement), Adafactor row/col second moment (stored in param dtype, not
  forced to fp32/int8).
- Much lighter than v1 (no per-element LR tensor), but still keeps a full bool tensor
  for polarity.

### v3 — `automagic3.py` (most sophisticated / heavily documented)
Explicit "improvements over v2," per its own docstring:
- **Granularity**: one adaptive LR **per parameter GROUP** (coarser than v2), because
  per-tensor LRs let coupled units (e.g. attention Q/K pairs) diverge to opposite
  extremes; pooling the vote at group level cancels that.
- **Direction-consistency vote from a sign-history window** (length `H=8` by default,
  1-bit-packed), not a single-step agreement check. v2's approach has no fixed point —
  a parameter that's simply still descending keeps "agreeing" forever and its LR
  ratchets up unboundedly. v3 only votes decisively on two window patterns: all H signs
  agree (+1, "step too small") or perfect alternation (-1, "step too large,"
  overshoot signature); anything else = 0 (noise, no vote).
- **Multiplicative/geometric LR update**: `lr *= exp(vote)` instead of additive bump —
  moves scale with current LR, and up/down votes exactly cancel (no drift).
- Stochastic rounding extended to fp16 as well as bf16 (v2 only did bf16).
- **State per element**: just the packed sign-history bits (H/8 bytes, e.g. 1 byte at
  H=8) + Adafactor row/col second moment. No per-element or per-parameter LR tensor —
  LR is one scalar per group.
- Supports `fused=True` (backward-hook fused, bypasses trainer grad-clipping/NaN-skip,
  incompatible with multi-step grad accumulation) or `fused=False` (accumulate grads
  with stochastic rounding, apply in `.step()`).
- `min_lr`/`max_lr` here are pure overflow failsafes, not real operating bounds
  (unlike v1/v2 where they're the actual clamp range).

### EXPERIMENT — `automagicEXPERIMENT.py` (unreleased, "expect breaking changes")
- Takes v3's group-level, sign-history, multiplicative-vote LR controller verbatim,
  but swaps Adafactor's factored second moment for a **true per-element AdamW-style
  second moment**, stored int8-quantized + Hadamard-rotated ("ConvRot8") in sqrt
  domain — closer to full AdamW quality at ~2 bytes/element total (moment + sign
  history) instead of Adafactor's approximation.
- **Hard constraint: `beta1` must be 0** (no momentum) — enforced by a `ValueError`.
  Momentum would smear updates into same-sign runs during an overshoot bounce, which
  the sign-window would misread as "consistency" and vote LR *up* during an actual
  overshoot (verified to explode with beta1=0.9 in testing).
- Votes computed directly on gradient sign (equivalent to update sign since beta1=0,
  but simpler).
- Trust-region clip (RMS-scaled + hard elementwise clamp) is called out as mandatory
  here (not just an option) because under-resolved 8-bit second-moment elements can
  dequantize to ~0 and blow up `grad/denom`.
- Positioned as: "what if you want AdamW-quality variance estimation but still want
  v3's cheap group-level LR controller."

## Quick-reference comparison

| Optimizer | LR granularity | Update rule | Extra per-elem state | Fused backward |
|---|---|---|---|---|
| Prodigy (plain/8bit) | 1 global scalar (D) | monotonic, never decreases | `p0`, `s`, Adam moments | no |
| Automagic v1 | per weight element | additive bump | full LR tensor + polarity | no |
| Automagic v2 | per parameter tensor | additive bump | polarity tensor only | yes |
| Automagic v3 | per parameter group | multiplicative, sign-history vote | packed sign history | optional |
| Automagic EXPERIMENT | per parameter group (v3 controller) | multiplicative, sign-history vote | packed sign history + full int8 2nd moment | optional (beta1 forced 0) |

## Automagic v3 — Full Technical Spec (for OneTrainer port)

Source: `toolkit/optimizers/automagic3.py` (700 lines). This section is exact —
transcribed from the real code, not paraphrased — so treat it as the spec to port.

### State layout

**Per param GROUP** (`torch.optim.Optimizer` group dict, e.g. `param_groups[0]`):
- `lr`, `min_lr`, `max_lr`, `beta2`, `eps`, `clip_threshold`, `weight_decay`,
  `polarity_history` (H) — plain hyperparameters, identical for every param in
  the group.
- Two **transient, step-scoped accumulators** (not persisted): `group_num`,
  `group_den` — pooled vote numerator/denominator, summed across every element
  of every tensor in the group during the step, consumed once at the end of the
  step and reset to `None`.

**Per parameter** (`self.state[p]` dict):
- `lr` — a **0-D fp32 tensor**, the group's adapted LR mirrored onto this param.
  All params in a group carry an identical value; it's stored per-param only so
  it survives normal `state_dict` save/load. Initialized to
  `clamp(group["lr"], min_lr, max_lr)`.
- `sign_history` — a ring buffer, shape `(H, ceil(numel/8))`, dtype `uint8`. Each
  row is one past step's plane of update-sign bits for every element, packed 8
  per byte. **This is the only per-element state** (H/8 bytes/element, e.g. 1
  byte/element at default H=8).
- `hist_idx` — int, index of the OLDEST plane in the ring buffer (the one to be
  overwritten next step).
- `hist_fill` — int, how many real planes have been written so far (0..H); the
  vote is gated (skipped) until `hist_fill == H`, i.e. for the first H steps
  after init or after a history reset.
- `step` — plain step counter (not used for bias correction — there is none).
- Second moment, Adafactor-style factored for tensors with `dim() >= 2`:
  `exp_avg_sq_row` shape = `p.shape[:-1]`, `exp_avg_sq_col` shape =
  `p.shape[:-2] + p.shape[-1:]` (i.e. reduce only the last two dims; leading
  dims — e.g. conv `out,in` — are carried as batch dims). For `dim() < 2`
  tensors (biases, norm weights): a full-size `exp_avg_sq`, same shape as `p`.
  All second-moment tensors are stored in **`p.dtype`** (not forced fp32).

### Per-parameter update (runs once per step per param — either in a fused
`register_post_accumulate_grad_hook`, or in `.step()` for the non-fused path)

1. If `p.grad is None`: skip. Cast grad to fp32 if not already. Apply
   `grad.nan_to_num_(nan=0, posinf=0, neginf=0)` — mandatory in fused mode since
   this runs *inside* backward, before the trainer's own grad-clipping/NaN-skip
   would fire.
2. `sq = grad * grad`.
3. **Second moment EMA**, math done in fp32 (state upcast if lower precision,
   then copied back with a plain round-to-nearest cast — the state cast is
   *not* stochastically rounded, only the final parameter write-back is):
   - `dim() >= 2`: `row = beta2*row + (1-beta2)*(mean(sq, dim=-1) + eps)`;
     `col = beta2*col + (1-beta2)*(mean(sq, dim=-2) + eps)` — note eps is folded
     into the *reduced* row/col vectors before the EMA update, not added to the
     full-size `sq` tensor (perf optimization: `mean(g²+eps) == mean(g²)+eps`).
     Then reconstruct the per-element normalizer as the outer product:
     `r = rsqrt(row / mean(row, dim=-1))`, unsqueezed on the last axis;
     `c = rsqrt(col)`, unsqueezed on the second-to-last axis;
     `update = (r * c) * grad` (standard HF-Adafactor factored reconstruction).
   - `dim() < 2`: `v = beta2*v + (1-beta2)*sq`; `update = (v + eps).rsqrt() * grad`
     (eps added directly here, not folded — the two branches are not bit-identical
     in eps placement, both are just div-by-zero guards).
4. **Trust-region clip**, in this order:
   - RMS clip: `update /= max(1.0, rms(update) / clip_threshold)` where
     `rms(t) = t.norm(2) / sqrt(t.numel())` — scales the whole tensor down if its
     aggregate RMS exceeds `clip_threshold`, never scales up.
   - Then a **hard elementwise clamp**: `update.clamp_(-clip_threshold, clip_threshold)`
     — the RMS clip alone still allows one outlier element to survive at up to
     `sqrt(numel) * clip_threshold`; this bounds every single element too.
5. **Sign bit for the vote**: `cur_bits = update > 0`. Since every scaling factor
   applied so far (second-moment rsqrt, RMS clip, clamp) is strictly positive,
   `sign(update) == sign(grad)` exactly — an implementer could equivalently use
   `sign(grad)` directly. An exact-zero update records as the *false/negative*
   bit; harmless because its vote weight (`|update|`) is zero anyway.
6. **Push into the ring buffer**: `sign_history[hist_idx] = pack_bits(cur_bits)`;
   `hist_idx = (hist_idx + 1) % H`; `hist_fill = min(H, hist_fill + 1)`. Do this
   *before* voting so the vote sees the freshest H signs.
7. **Vote (only when `hist_fill == H`)**:
   - Roll the ring buffer into chronological order: `chron = roll(sign_history, -hist_idx, dims=0)`
     (after the push above, `hist_idx` points at the oldest surviving plane; rolling
     by `-hist_idx` puts oldest→newest along dim 0).
   - Unpack every plane back to a `(H, numel)` 0/1 tensor: `bits`.
   - `s1 = bits.sum(dim=0)` — per-element count of "positive" bits across the window.
   - `flips = (bits[1:] ^ bits[:-1]).sum(dim=0)` — per-element count of sign changes
     between adjacent steps in the window (H-1 comparisons).
   - `up = (s1 == H) | (s1 == 0)` — **all H signs identical** (all positive or all
     negative) → this element votes "step too small."
   - `down = (flips == H - 1)` — **every adjacent pair flipped**, i.e. perfect
     alternation → this element votes "step too large" (overshoot signature).
   - Everything else (any other bit pattern) casts **no vote** (weight 0) — treated
     as noise. This is the key asymmetry vs. v2, which voted on every single step
     from raw 1-step agreement.
   - Per-element vote weight: `w = |update|` (elements that aren't really moving
     the weights don't get a voice).
   - `num = sum(w * up) - sum(w * down)`; `den = sum(w)` — reduced over every
     element of *this tensor*, then **added into the group's running `num`/`den`
     accumulators** (summed across every tensor that shares this param group).
8. **Weight decay** (decoupled/AdamW-style): if `weight_decay != 0`,
   `update += weight_decay * p` (added to the already-clipped update, so decay is
   NOT itself clipped).
9. **Apply the update**: `p -= lr * update`, where `lr` is the per-param 0-D
   tensor (broadcasts as a scalar). fp32 params update in place directly. Lower
   precision params: compute in an fp32 copy, then **stochastically round** the
   result back into `p`'s native dtype (fast mantissa-truncation trick for
   bf16/fp16 — add uniform noise into the bits that will be discarded, then mask
   them off before the narrowing cast, so P(round up) equals the true fractional
   remainder; generic ULP-based stochastic rounding for anything else).
10. Clear `p.grad = None`.

### End-of-step: applying the pooled vote (once per group, NOT per parameter)

This is the **critical detail easy to miss when porting**: the per-parameter loop
above (step 7) only *accumulates* `num`/`den` into group-level counters — it does
**not** touch the group's `lr` itself. The actual LR update happens exactly once
per param group, after every tensor in that group has been processed for the
step:

```
signal = clamp(num / max(den, 1e-30), -1.0, 1.0)   # pooled fraction, one scalar
factor = exp(signal)                                # multiplicative nudge
for p in group.params:
    state[p]["lr"].mul_(factor).clamp_(min=min_lr, max=max_lr)
reset group_num, group_den to None
```

- **In non-fused mode** this naturally happens inside `.step()` after the
  per-param loop.
- **In fused mode** (the default in this codebase — updates happen inside
  `register_post_accumulate_grad_hook`, immediately as each param's gradient is
  ready during backward), the per-param weight updates and vote *accumulation*
  already happened during `backward()`, but **the LR nudge itself is still only
  applied when `.step()` is called** — `.step()` in fused mode does no
  per-weight math, it just runs this group-vote application. **You must still
  call `optimizer.step()` once per training step even in fused mode**, purely to
  flush the pooled LR update. If you skip it, weights update correctly but the
  LR never adapts.
- `den` is clamped to a minimum of `1e-30` purely to avoid an actual divide-by-zero
  (in practice only possible if every single update in the group was exactly zero).
- `signal` is clamped to `[-1, 1]` — bounds a single step's LR move to at most
  `exp(±1) ≈ 2.7x` (only reachable with a perfectly unanimous pool, essentially
  unreachable in practice; realistic per-step moves are a few percent since the
  "silent" (non-voting/noise) majority dilutes the pooled fraction toward 0).

### Grouping — the load-bearing design choice

The whole point of v3 vs. earlier per-tensor/per-channel Automagic attempts:
**pool the vote across every element of every tensor that shares a param group**,
not per-tensor. Coupled tensors (canonical example: attention Q/K pairs) can
satisfy the loss equally well by scaling one up and the other down — a per-tensor
LR lets each one "vote" independently and diverge to opposite extremes forever
(Q thinks its step is too small and climbs, K thinks its is too big and shrinks).
Pooling their votes into one shared LR makes those opposing per-tensor signals
cancel in the sum, so only true common-mode evidence ("the whole group is
systematically under/overshooting") moves the LR.

**Implication for a OneTrainer port**: decide param-group boundaries deliberately
— e.g. one LR per logical module family (all UNet/DiT blocks together, text
encoder separately, etc.), not one LR per `nn.Parameter`. Getting this wrong
(too fine-grained) reproduces the exact failure mode v3 was built to fix.

### Other implementation-relevant defaults

- `lr=1e-6` (launch point only — controller adapts away from it immediately),
  `min_lr=1e-8`, `max_lr=1e3` (these two are pure numerical-overflow guards at
  their defaults, not real operating bounds — tighten them if you want the
  controller hard-railed).
- `beta2=0.999`, `eps=1e-30`, `clip_threshold=1.0`, `weight_decay=0.0`,
  `polarity_history=8` (valid range 2–64; H/8 bytes/element; longer H makes the
  two vote events rarer/more decisive at the cost of an H-step reaction lag and
  more memory).
- No bias correction anywhere (deliberate — "LoRA runs are short and a slow ramp
  wastes steps"; for a soft start, just set a low `lr` and let the controller
  ramp it).
- No gain knob on the LR update — removed entirely once the vote became a pooled
  fraction in natural log-units (`lr *= exp(signal)` directly).
- Sparse gradients unsupported (raises `RuntimeError`).
- On `load_state_dict`: hyperparameters are deliberately **not** restored from
  the checkpoint (constructor args always win, so you can change any setting
  mid-run on resume). The per-param `lr` tensors are unified onto their
  geometric median when loading (handles resuming from older/differently-grouped
  checkpoints sanely). If `sign_history` shape doesn't match the current
  `polarity_history`/param shape, it's reset to zero and `hist_fill` restarts at
  0 (one clean H-step re-warmup) rather than erroring.

### Reference code (verbatim from `toolkit/optimizers/automagic3.py`)

Copied directly so nothing gets lost in translation when porting to OneTrainer.

**Bit pack/unpack helpers** (used for the sign-history ring buffer):
```python
_PACK_CONSTS: dict = {}

@classmethod
def _pack_consts(cls, device):
    consts = cls._PACK_CONSTS.get(device)
    if consts is None:
        consts = (
            torch.tensor(
                [1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8
            ),
            torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8
            ),
        )
        cls._PACK_CONSTS[device] = consts
    return consts

@classmethod
def _pack_bits(cls, bits: torch.Tensor) -> torch.Tensor:
    # Pack sign bits (bool / {0, 1}) 8 per byte (uint8), as a base-2 dot
    # product per group of 8 (two kernels rather than per-slice shift/or
    # chains).
    weights, _ = cls._pack_consts(bits.device)
    flat = bits.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return (flat.view(-1, 8) * weights).sum(-1, dtype=torch.uint8)

@classmethod
def _unpack_bits(cls, packed: torch.Tensor, numel: int) -> torch.Tensor:
    # Inverse of _pack_bits: uint8 -> flat uint8 {0, 1} of length numel.
    _, shifts = cls._pack_consts(packed.device)
    vals = (packed.unsqueeze(-1) >> shifts).bitwise_and_(1)
    return vals.view(-1)[:numel]
```

**RMS and Adafactor factored second-moment reconstruction**:
```python
@staticmethod
def _rms(t: torch.Tensor) -> torch.Tensor:
    return t.norm(2) / (t.numel() ** 0.5)

@staticmethod
def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    # rebuilds the rank-1 approximation of 1/sqrt(v) as the outer product
    # rsqrt(row / mean(row)) (x) rsqrt(col) -- standard HF Adafactor reconstruction.
    r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c = col.unsqueeze(-2).rsqrt()
    return torch.mul(r, c)
```

**Stochastic rounding** (fast bit-trick for bf16/fp16, generic fallback otherwise):
```python
@staticmethod
def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
    # Add uniform noise into the dropped low mantissa bits of the fp32 bit
    # pattern, then zero them, so the subsequent narrowing cast rounds up
    # with probability equal to the truncated fractional part.
    # bf16 drops 16 bits (high half of fp32); fp16 drops 13 bits (23-10 mantissa).
    as_int = v_fp32.view(torch.int32)
    as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
    as_int.bitwise_and_(-(1 << drop_bits))
    return v_fp32

@staticmethod
def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Generic fp32 -> low-precision stochastic rounding: add noise of +/- half
    # a target ULP, round to nearest -> unbiased in expectation.
    finfo = torch.finfo(dtype)
    absv = v.abs().clamp_(min=finfo.tiny)
    ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
    noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
    return v.add_(noise).to(dtype)

@classmethod
def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    if dst.dtype == torch.bfloat16:
        dst.copy_(cls._sr_truncate(src_fp32, 16))
    elif dst.dtype == torch.float16:
        dst.copy_(cls._sr_truncate(src_fp32, 13))
    else:
        dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))
```

**Per-param state init**:
```python
def _init_state(self, p: torch.Tensor, group: dict) -> None:
    state = self.state[p]
    state["step"] = 0
    state["lr"] = torch.tensor(
        min(max(float(group["lr"]), group["min_lr"]), group["max_lr"]),
        dtype=torch.float32,
        device=p.device,
    )
    H = group["polarity_history"]
    width = (p.numel() + 7) // 8
    state["sign_history"] = torch.zeros(
        (H, width), dtype=torch.uint8, device=p.device
    )
    state["hist_idx"] = 0
    state["hist_fill"] = 0
    if p.dim() >= 2:
        state["exp_avg_sq_row"] = torch.zeros(
            p.shape[:-1], dtype=p.dtype, device=p.device
        )
        state["exp_avg_sq_col"] = torch.zeros(
            p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
        )
    else:
        state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
```

**The core per-parameter update** (this is the whole algorithm — second moment,
trust-region clip, sign vote, group accumulation, weight decay, stochastically
rounded write-back):
```python
@torch.no_grad()
def _update_param(self, p: torch.Tensor, group: dict) -> None:
    if p.grad is None:
        return
    state = self.state[p]
    if len(state) == 0:
        self._init_state(p, group)

    grad = p.grad
    if grad.is_sparse:
        raise RuntimeError("Automagic3 does not support sparse gradients.")
    if grad.dtype != torch.float32:
        grad = grad.to(torch.float32)

    # Neutralise non-finite grads (fused mode runs before the trainer's own
    # grad clipping / nan-skip, which come too late here).
    grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    beta2 = group["beta2"]
    eps = group["eps"]
    sq = grad * grad

    if p.dim() >= 2:
        row_state = state["exp_avg_sq_row"]
        col_state = state["exp_avg_sq_col"]
        if row_state.dtype == torch.float32:
            row, col = row_state, col_state
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
        else:
            row = row_state.to(torch.float32)
            col = col_state.to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            row_state.copy_(row.to(row_state.dtype))
            col_state.copy_(col.to(col_state.dtype))
        update = self._approx_sq_grad(row, col).mul_(grad)
    else:
        v_state = state["exp_avg_sq"]
        if v_state.dtype == torch.float32:
            v = v_state
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
        else:
            v = v_state.to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            v_state.copy_(v.to(v_state.dtype))
        update = v.add(eps).rsqrt().mul_(grad)

    # Update-RMS clip (trust region), then a hard elementwise clamp.
    update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.clamp_(-group["clip_threshold"], group["clip_threshold"])

    # Direction-consistency lr control (the vote rule).
    cur_bits = update.gt(0.0)
    hist = state["sign_history"]  # (H, numel/8) 1-bit packed uint8
    idx = state["hist_idx"]  # oldest plane (overwritten below)
    H = hist.shape[0]
    lr_t = state["lr"]  # this param's mirror of the shared group lr

    hist[idx].copy_(self._pack_bits(cur_bits))
    state["hist_idx"] = (idx + 1) % H
    fill = min(H, state["hist_fill"] + 1)
    state["hist_fill"] = fill

    if fill == H:
        _, shifts = self._pack_consts(hist.device)
        chron = torch.roll(hist, -state["hist_idx"], dims=0)
        bits = (
            (chron.unsqueeze(-1) >> shifts)
            .bitwise_and_(1)
            .view(H, -1)[:, : update.numel()]
        )
        s1 = bits.sum(0, dtype=torch.int16)
        flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
        up = s1.eq(H).logical_or_(s1.eq(0))
        down = flips.eq(H - 1)
        w = update.abs().view(-1)
        num = (w * up).sum().sub_((w * down).sum())
        den = w.sum()
        gi = self._param_group_index.get(p)
        if gi is not None:
            if self._group_num[gi] is None:
                self._group_num[gi] = num
                self._group_den[gi] = den
            else:
                acc = self._group_num[gi]
                if num.device != acc.device:
                    num = num.to(acc.device)
                    den = den.to(acc.device)
                acc.add_(num)
                self._group_den[gi].add_(den)

    state["step"] += 1

    wd = group["weight_decay"]

    if p.dtype == torch.float32:
        if wd != 0.0:
            update.add_(p, alpha=wd)
        p.addcmul_(update, lr_t, value=-1.0)
    else:
        new_p_fp32 = p.to(torch.float32)
        if wd != 0.0:
            update.add_(new_p_fp32, alpha=wd)
        new_p_fp32.addcmul_(update, lr_t, value=-1.0)
        self._stochastic_copy_(p, new_p_fp32)

    p.grad = None
```

**Applying the pooled group vote** (runs once per group per step — this is
where the LR actually changes; see the "critical detail" callout above):
```python
def _apply_group_votes(self) -> None:
    for gi, group in enumerate(self.param_groups):
        num = self._group_num[gi]
        if num is None:
            continue
        den = self._group_den[gi]
        signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)
        factor = torch.exp(signal)
        for p in group["params"]:
            st = self.state.get(p)
            if st is None or "lr" not in st:
                continue
            lr_t = st["lr"]
            f = factor if factor.device == lr_t.device else factor.to(lr_t.device)
            lr_t.mul_(f).clamp_(min=group["min_lr"], max=group["max_lr"])
        self._group_num[gi] = None
        self._group_den[gi] = None
```

**`step()`** (ties it together — note `_apply_group_votes()` always runs, fused
or not):
```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    if not self.fused:
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                accum = getattr(p, "_accum_grad", None)
                if accum is not None:
                    p.grad = accum
                    del p._accum_grad
                if p.grad is None:
                    continue
                self._update_param(p, group)
    self._apply_group_votes()
    return loss
```

## Config/selection notes
- Registered as string keys in `toolkit/optimizer.py`: `prodigy`, `prodigy8bit`,
  `automagic`, `automagic2`, `automagic3`, `automagicexperiment` (exact match, last one
  not exposed in the web UI — it's the unreleased variant).
- Prodigy/D-adaptation optimizers need special-cased LR reporting elsewhere in the
  codebase (`optimizer.param_groups[0]["d"] * lr`) since they don't expose a normal
  scalar LR the way Automagic's `get_avg_learning_rate()` does.

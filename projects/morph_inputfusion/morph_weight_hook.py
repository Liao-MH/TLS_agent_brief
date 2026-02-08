# projects/morph_inputfusion/morph_weight_hook.py
from __future__ import annotations

import logging
from typing import Optional, Tuple, Literal

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

try:
    from mmengine.dist import get_rank
except Exception:
    def get_rank() -> int:  # type: ignore
        return 0


Mode = Literal["auto", "iter_based", "epoch_based_iters", "epoch_based_epoch_only"]


@HOOKS.register_module()
class MorphWeightWarmupHook(Hook):
    """
    Three-stage, percentage-based schedule for MorphSegDataPreProcessor.morph_weight.

    Progress p in [0, 1]:

      Stage-1 (hold):   p < hold_ratio
          morph_weight = start_weight

      Stage-2 (ramp):   hold_ratio <= p < hold_ratio + ramp_ratio
          linearly ramp from start_weight to target_weight

      Stage-3 (hold):   p >= hold_ratio + ramp_ratio
          morph_weight = target_weight

    Supports:
      - IterBasedTrainLoop: uses max_iters
      - EpochBasedTrainLoop: uses max_epochs * len(dataloader) when possible,
        otherwise falls back to epoch-only progress.

    mode:
      auto (default): pick best available
      iter_based: force iter-based (requires max_iters)
      epoch_based_iters: force epoch-based using max_epochs*len(dataloader)
      epoch_based_epoch_only: force epoch-only fraction

    Conflict guard (as you requested):
      - Only raise if we are in IterBasedTrainLoop AND both max_iters and max_epochs are positive ints.
        (EpochBasedTrainLoop may expose runner.max_iters auto-derived, so we do NOT raise there.)
    """

    def __init__(
        self,
        hold_ratio: float = 0.10,
        ramp_ratio: float = 0.30,
        start_weight: float = 0.0,
        target_weight: float = 0.15,
        log_interval: int = 200,
        verbose: bool = True,
        mode: Mode = "auto",
    ) -> None:
        self.hold_ratio = float(max(0.0, hold_ratio))
        self.ramp_ratio = float(max(0.0, ramp_ratio))

        s = self.hold_ratio + self.ramp_ratio
        if s > 1.0 + 1e-12:
            self.hold_ratio /= s
            self.ramp_ratio /= s

        self.start_weight = float(start_weight)
        self.target_weight = float(target_weight)
        self.log_interval = int(max(1, log_interval))
        self.verbose = bool(verbose)

        if mode not in ("auto", "iter_based", "epoch_based_iters", "epoch_based_epoch_only"):
            raise ValueError(f"Invalid mode={mode}.")
        self.mode: Mode = mode

        self._last_logged_iter: Optional[int] = None
        self._last_set_weight: Optional[float] = None

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _get_preprocessor(self, runner):
        model = self._unwrap_model(runner.model)
        return getattr(model, "data_preprocessor", None)

    def _get_train_dataloader(self, runner):
        loop = getattr(runner, "train_loop", None)
        dl = getattr(loop, "dataloader", None) if loop is not None else None
        if dl is None:
            dl = getattr(runner, "train_dataloader", None)
        return dl

    def _loop_kind(self, runner) -> str:
        """Infer whether we're in epoch-based or iter-based training."""
        loop = getattr(runner, "train_loop", None)
        name = loop.__class__.__name__ if loop is not None else ""
        if "EpochBased" in name:
            return "epoch"
        if "IterBased" in name:
            return "iter"
        return "unknown"

    @staticmethod
    def _safe_int(x) -> Optional[int]:
        return int(x) if isinstance(x, int) and x > 0 else None

    def _read_limits(self, runner) -> Tuple[Optional[int], Optional[int]]:
        """
        Read (max_iters, max_epochs), preferring train_loop.* then runner.*.
        """
        loop = getattr(runner, "train_loop", None)

        max_iters = getattr(loop, "max_iters", None) if loop is not None else None
        if max_iters is None:
            max_iters = getattr(runner, "max_iters", None)

        max_epochs = getattr(loop, "max_epochs", None) if loop is not None else None
        if max_epochs is None:
            max_epochs = getattr(runner, "max_epochs", None)

        return self._safe_int(max_iters), self._safe_int(max_epochs)

    def _read_iters_per_epoch(self, runner) -> Optional[int]:
        dl = self._get_train_dataloader(runner)
        try:
            v = len(dl)  # type: ignore[arg-type]
        except Exception:
            v = None
        return int(v) if isinstance(v, int) and v > 0 else None

    def _try_get_total_iters(self, runner) -> Tuple[Optional[int], str]:
        """
        Return (total_iters, mode_string_with_numbers).

        mode_string_with_numbers examples:
          - iter_based(max_iters=15000)
          - epoch_based_iters(max_epochs=20, iters_per_epoch=1385, total_iters=27700)
          - epoch_based_epoch_only(max_epochs=20, iters_per_epoch=NA)
          - unknown(max_epochs=NA, max_iters=NA, iters_per_epoch=NA)

        Conflict guard:
          - Only raise if IterBasedTrainLoop AND both max_iters and max_epochs are set (>0).
        """
        kind = self._loop_kind(runner)
        max_iters, max_epochs = self._read_limits(runner)

        has_iters = isinstance(max_iters, int) and max_iters > 0
        has_epochs = isinstance(max_epochs, int) and max_epochs > 0

        # Conflict: ONLY when IterBased loop and both are set (treat as "manual conflict")
        if kind == "iter" and has_iters and has_epochs:
            raise ValueError(
                f"[MorphWeightWarmupHook] Ambiguous schedule: IterBasedTrainLoop "
                f"but both max_iters={max_iters} and max_epochs={max_epochs} are set. "
                f"Please keep only one."
            )

        # Decide strategy based on self.mode (force) or auto
        forced = self.mode != "auto"

        iters_per_epoch = self._read_iters_per_epoch(runner) if has_epochs else None

        # -------- forced modes --------
        if forced:
            if self.mode == "iter_based":
                if not has_iters:
                    raise ValueError("[MorphWeightWarmupHook] mode=iter_based requires max_iters>0.")
                return int(max_iters), f"iter_based(max_iters={int(max_iters)})"

            if self.mode == "epoch_based_iters":
                if not has_epochs:
                    raise ValueError("[MorphWeightWarmupHook] mode=epoch_based_iters requires max_epochs>0.")
                if not (isinstance(iters_per_epoch, int) and iters_per_epoch > 0):
                    raise ValueError(
                        "[MorphWeightWarmupHook] mode=epoch_based_iters requires len(dataloader)>0 "
                        "(you may be using InfiniteSampler / dataloader has no __len__)."
                    )
                total = int(max_epochs) * int(iters_per_epoch)
                return total, (
                    f"epoch_based_iters(max_epochs={int(max_epochs)}, "
                    f"iters_per_epoch={int(iters_per_epoch)}, total_iters={int(total)})"
                )

            if self.mode == "epoch_based_epoch_only":
                if not has_epochs:
                    raise ValueError("[MorphWeightWarmupHook] mode=epoch_based_epoch_only requires max_epochs>0.")
                # total_iters unknown
                return None, f"epoch_based_epoch_only(max_epochs={int(max_epochs)}, iters_per_epoch=NA)"

        # -------- auto mode --------
        # Prefer iter_based if we are actually in IterBasedTrainLoop, OR if epochs absent.
        if has_iters and (kind == "iter" or not has_epochs):
            return int(max_iters), f"iter_based(max_iters={int(max_iters)})"

        # Epoch-based: try epoch_based_iters, else epoch_only
        if has_epochs:
            if isinstance(iters_per_epoch, int) and iters_per_epoch > 0:
                total = int(max_epochs) * int(iters_per_epoch)
                return total, (
                    f"epoch_based_iters(max_epochs={int(max_epochs)}, "
                    f"iters_per_epoch={int(iters_per_epoch)}, total_iters={int(total)})"
                )
            return None, f"epoch_based_epoch_only(max_epochs={int(max_epochs)}, iters_per_epoch=NA)"

        # Nothing available
        mi_str = str(max_iters) if has_iters else "NA"
        me_str = str(max_epochs) if has_epochs else "NA"
        return None, f"unknown(max_epochs={me_str}, max_iters={mi_str}, iters_per_epoch=NA)"

    def _progress(self, runner, batch_idx: int) -> Tuple[float, str]:
        cur_iter = int(getattr(runner, "iter", 0))
        total_iters, mode_str = self._try_get_total_iters(runner)

        if isinstance(total_iters, int) and total_iters > 0:
            p = (cur_iter + 1) / float(total_iters)
            return max(0.0, min(1.0, p)), mode_str

        # epoch-only fallback: use epoch/max_epochs (+ intra-epoch fraction if len(dataloader) exists)
        loop = getattr(runner, "train_loop", None)
        epoch = getattr(runner, "epoch", None)
        if epoch is None and loop is not None:
            epoch = getattr(loop, "epoch", 0)

        max_iters, max_epochs = self._read_limits(runner)
        if isinstance(epoch, int) and isinstance(max_epochs, int) and max_epochs > 0:
            denom = self._read_iters_per_epoch(runner)
            if isinstance(denom, int) and denom > 0:
                p = (epoch + (batch_idx + 1) / float(denom)) / float(max_epochs)
            else:
                p = (epoch + 1) / float(max_epochs)
            return max(0.0, min(1.0, p)), mode_str

        return 0.0, mode_str

    def _compute_weight(self, p: float) -> float:
        if p < self.hold_ratio:
            return self.start_weight

        t0 = self.hold_ratio
        t1 = self.hold_ratio + self.ramp_ratio
        if self.ramp_ratio > 1e-12 and p < t1:
            alpha = (p - t0) / max(1e-12, (t1 - t0))
            return self.start_weight + alpha * (self.target_weight - self.start_weight)

        return self.target_weight

    @staticmethod
    def _mode_hint(mode_str: str) -> str:
        # mode_str now contains numbers; we only add lightweight semantic hint.
        if mode_str.startswith("epoch_based_iters"):
            return "✅ most precise"
        if mode_str.startswith("epoch_based_epoch_only"):
            return "⚠️ epoch-only"
        if mode_str.startswith("iter_based"):
            return "ℹ️ iter-based"
        if mode_str.startswith("unknown"):
            return "⚠️ no limits"
        return ""

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        dp = self._get_preprocessor(runner)
        if dp is None:
            return

        # strict NO-OP when morph is disabled (RGB-only must be unaffected)
        if getattr(dp, "morph_enabled", None) is False:
            return
        if not hasattr(dp, "morph_weight"):
            return

        p, mode_str = self._progress(runner, batch_idx)
        w = float(self._compute_weight(p))

        if self._last_set_weight is None or abs(w - self._last_set_weight) > 1e-12:
            setattr(dp, "morph_weight", w)
            self._last_set_weight = w

        # logging (rank0 only)
        if not self.verbose or get_rank() != 0:
            return

        cur_iter = int(getattr(runner, "iter", 0))
        if cur_iter % self.log_interval != 0:
            return
        if self._last_logged_iter == cur_iter:
            return
        self._last_logged_iter = cur_iter

        loop = getattr(runner, "train_loop", None)
        loop_name = loop.__class__.__name__ if loop is not None else "None"

        hint = self._mode_hint(mode_str)
        print_log(
            f"[MorphWeightWarmupHook] loop={loop_name} iter={cur_iter} progress={p:.4f} "
            f"mode={mode_str} {hint} "
            f"morph_weight={w:.6f} (hold_ratio={self.hold_ratio}, ramp_ratio={self.ramp_ratio}, "
            f"start={self.start_weight}, target={self.target_weight})",
            logger=runner.logger,
            level=logging.INFO,
        )

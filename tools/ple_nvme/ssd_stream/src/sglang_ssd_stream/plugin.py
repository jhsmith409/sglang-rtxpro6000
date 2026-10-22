"""SGLang plugin registration."""

from __future__ import annotations

import os
import platform

from .config import configure_cli
from .graph import after_load_batch, around_execute


def register() -> None:
    # Installing the optional wheel must not change RAM-mode or 27B startup:
    # SGLang auto-discovers entry points even without SGLANG_PLUGINS set.
    if os.environ.get("PENNY_PLE_BACKEND") != "nvme":
        return
    try:
        _register_pennyroyal()
    except Exception as exc:
        # The host plugin loader logs Exception and continues. Serving without
        # a table is unsafe; an explicitly selected backend must fail closed.
        raise SystemExit(f"Pennyroyal NVMe PLE initialization failed: {exc}") from exc


def _register_pennyroyal() -> None:
    if platform.machine() != "x86_64":
        raise RuntimeError(
            "This Pennyroyal NVMe PLE adapter is qualified for x86_64 only"
        )
    _register()


def _install_required_hook_enforcement(HookRegistry, required_hooks) -> None:
    current_descriptor = HookRegistry.__dict__["apply_hooks"]
    current = current_descriptor.__func__
    if getattr(current, "_pennyroyal_nvme_enforcement", False):
        return

    required_hooks = tuple(required_hooks)
    required_targets = {target for target, _, _ in required_hooks}

    def apply_hooks_and_require_nvme(cls) -> None:
        missing_registrations = []
        for target, hook_type, hook in required_hooks:
            registrations = cls._hooks.get(target, ())
            if not any(
                registered_type is hook_type and registered_hook is hook
                for registered_type, registered_hook, _ in registrations
            ):
                missing_registrations.append(target)
        if missing_registrations:
            raise SystemExit(
                "Pennyroyal NVMe PLE required hooks failed registration: "
                + ", ".join(sorted(missing_registrations))
            )

        if not getattr(cls, "_pennyroyal_nvme_hooks_applied", False):
            already_patched = required_targets & cls._patched
            if already_patched:
                raise SystemExit(
                    "Pennyroyal NVMe PLE required hooks were patched before "
                    "NVMe registration: "
                    + ", ".join(sorted(already_patched))
                )

        current(cls)
        missing = required_targets - cls._patched
        if missing:
            # HookRegistry logs and continues after an individual application
            # failure. Explicit NVMe selection must instead stop this process:
            # serving after a partial adapter install can load the wrong PLE.
            raise SystemExit(
                "Pennyroyal NVMe PLE required hooks failed to apply: "
                + ", ".join(sorted(missing))
            )
        cls._pennyroyal_nvme_hooks_applied = True

    apply_hooks_and_require_nvme._pennyroyal_nvme_enforcement = True
    HookRegistry.apply_hooks = classmethod(apply_hooks_and_require_nvme)


def _register() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .offload import around_make_layers
    from .qwen4 import Qwen4PLELayer, around_load_weights

    required_hooks = (
        (
            "sglang.srt.server_args.ServerArgs.from_cli_args",
            HookType.BEFORE,
            configure_cli,
        ),
        (
            "sglang.srt.models.qwen3_5.make_layers",
            HookType.AROUND,
            around_make_layers,
        ),
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer",
            HookType.REPLACE,
            Qwen4PLELayer,
        ),
        (
            (
                "sglang.srt.models.qwen4_exp."
                "Qwen4ExpForConditionalGeneration.load_weights"
            ),
            HookType.AROUND,
            around_load_weights,
        ),
        (
            (
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
                "DecodeCudaGraphRunner.execute"
            ),
            HookType.AROUND,
            around_execute,
        ),
        (
            (
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
                "DecodeCudaGraphRunner.load_batch"
            ),
            HookType.AFTER,
            after_load_batch,
        ),
    )
    if getattr(HookRegistry, "_pennyroyal_nvme_hooks_registered", False):
        return
    for target, hook_type, hook in required_hooks:
        HookRegistry.register(target, hook, hook_type)
    _install_required_hook_enforcement(HookRegistry, required_hooks)
    HookRegistry._pennyroyal_nvme_hooks_registered = True

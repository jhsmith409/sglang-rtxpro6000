"""CPU-only checks for the Flash-Next DFlash2 launch profile.

The profile has to resolve the intended model and settings (DFLASH against the
accepted NVFP4 Flash-Next target, the draft unquantized, the target's Triton
verify path, the C6/36 capacity pair, 524K context, 32 GB HiCache and the NIXL
tier, automatic KV sizing inside the 0.97 static fraction, and the full target
vocabulary rather than the FR-Spec map or the adaptive policy), and its cache
identity has to name the draft as well as the target.

No GPU and no weights: the server binary records argv, the image root is a
stand-in, and ``derive_namespace.py`` really runs.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"
LAUNCH = REPO / "docker" / "pennyroyal" / "launch" / "config"
RECIPE = CONFIGS / "serve-flash-next-dflash2.sh"
STARTUP = LAUNCH / "start-flash-next-dflash2.sh"
TARGET_OVERRIDES = json.dumps(
    {
        "text_config": {
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "yarn",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
            }
        }
    },
    separators=(",", ":"),
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_prebuilt_container_launch as container  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "pennyroyal"))
import penny_config as pc  # noqa: E402
import test_penny_config as config_suite  # noqa: E402


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def launch_recipe(tmp_path: Path, **env: str):
    result, argv, server_env, _ = container.launch(
        tmp_path, RECIPE, through_entrypoint=False, env=env
    )
    return result, argv, server_env


def test_recipe_resolves_the_intended_flash_next_dflash_launch(tmp_path):
    result, argv, server_env = launch_recipe(tmp_path)
    assert result.returncode == 0, result.stderr
    assert argv is not None
    for flag, expected in (
        ("--quantization", "modelopt_fp4"),
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--mem-fraction-static", "0.97"),
        ("--context-length", "524288"),
        ("--page-size", "64"),
        ("--max-running-requests", "6"),
        ("--max-mamba-cache-size", "36"),
        ("--hicache-size", "32"),
        ("--mamba-ssm-dtype", "bfloat16"),
        ("--gdn-mtp-cache-mode", "none"),
        ("--linear-attn-verify-backend", "triton"),
        ("--reasoning-parser", "qwen3"),
        ("--tool-call-parser", "qwen3_coder"),
        ("--speculative-algorithm", "DFLASH"),
        ("--speculative-draft-model-quantization", "unquant"),
        ("--speculative-draft-kv-cache-dtype", "fp8_e4m3"),
        ("--speculative-draft-window-size", "2048"),
        ("--speculative-num-draft-tokens", "4"),
    ):
        assert value_after(argv, flag) == expected, (flag, value_after(argv, flag))
    assert value_after(argv, "--speculative-draft-model-path").endswith("draft")
    assert value_after(argv, "--json-model-override-args") == TARGET_OVERRIDES
    # The report has to say which launch this is, not assume every DFLASH is 27B.
    assert "Profile: Flash-Next NVFP4 / DFlash2" in result.stdout
    # Full target vocabulary: neither the FR-Spec 64K map nor the adaptive
    # policy rides along, and no request-time sampling policy is invented. A
    # linear DFlash chain has no top-k flag either.
    for absent in (
        "--speculative-token-map",
        "--speculative-adaptive",
        "--speculative-adaptive-config",
        "--speculative-eagle-topk",
        "--min-p",
        # Automatic KV sizing: the pool is left to the memory SGLang profiles.
        "--max-total-tokens",
    ):
        assert absent not in argv, absent
    assert server_env["namespace"].startswith(str(tmp_path / "nixl root"))


def test_draft_position_extension_uses_this_drafts_own_rope(tmp_path):
    result, argv, _ = launch_recipe(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(value_after(argv, "--speculative-draft-model-override-args")) == {
        "max_position_embeddings": 524288,
        "rope_parameters": {
            "rope_type": "yarn",
            # This draft's own theta and base: the 27B draft's 1e7 would move
            # every position embedding the draft was trained with.
            "rope_theta": 10000,
            "factor": 2.0,
            "original_max_position_embeddings": 262144,
        },
    }
    # The target's rotary settings are untouched by the draft's extension.
    assert value_after(argv, "--json-model-override-args") == TARGET_OVERRIDES


def test_the_checkpoints_own_block8_is_selectable(tmp_path):
    result, argv, _ = launch_recipe(tmp_path, SPECULATIVE_NUM_DRAFT_TOKENS="8")
    assert result.returncode == 0, result.stderr
    assert value_after(argv, "--speculative-num-draft-tokens") == "8"


def test_a_block_width_the_draft_layout_was_not_built_for_stops_the_launch(tmp_path):
    for width in ("6", "16", "word"):
        result, argv, _ = launch_recipe(tmp_path, SPECULATIVE_NUM_DRAFT_TOKENS=width)
        assert result.returncode != 0 and argv is None, width
        assert "must be 4 or 8" in result.stderr, (width, result.stderr)


def test_the_draft_is_part_of_the_cache_identity(tmp_path):
    def relaunch(marker: str | None = None, **env: str) -> str:
        # The launch fixture rebuilds the draft's metadata before every launch,
        # but not its weights, so the tensor payload is what varies here.
        if marker is not None:
            weights = container.draft_checkpoint(tmp_path) / "model.safetensors"
            header = json.dumps(
                {marker: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
            ).encode()
            weights.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")
        result, _, server_env = launch_recipe(tmp_path, **env)
        assert result.returncode == 0, (marker, result.stderr)
        return server_env["namespace"]

    trained = relaunch("trained")
    # Same target, same settings, different draft weights: the persistent KV
    # cache may not be shared, because what the draft proposes differs.
    retuned = relaunch("retuned")
    assert retuned != trained
    # And the namespace is stable for an unchanged draft, not re-keyed per boot.
    assert relaunch() == retuned
    # A different verification width changes the draft's block layout, so it is
    # identity too rather than a reuse of the same captured blocks.
    assert relaunch(SPECULATIVE_NUM_DRAFT_TOKENS="8") != retuned
    # The accepted GDN accumulate mode is cache identity here as it is on the
    # other Flash-Next profiles.
    assert relaunch(FLASHINFER_GDN_FP16_ACCUM_MMA="0") != retuned
    # With the disk tier off the namespace is simply not derived.
    assert relaunch(NIXL="off") == "unset"


class ProfileRoutingTests(config_suite.FixtureMixin, unittest.TestCase):
    """The profile is selectable through the shipped configurator in both modes,
    and no existing profile moves."""

    def dflash2_draft(self) -> Path:
        draft = Path(self.mkdtemp()) / "Flash Next DFlash2"
        draft.mkdir(parents=True)
        (draft / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "model_type": "qwen3",
                    "dflash_config": {"target_layer_ids": [5, 15, 25, 35, 45]},
                }
            )
        )
        return draft

    def test_existing_profiles_keep_their_recipes(self):
        self.assertEqual(pc.recipe_for("next"), "serve-flash-next-frspec.sh")
        self.assertEqual(pc.recipe_for("next-plain"), "serve-flash-next.sh")
        self.assertEqual(pc.recipe_for("27b"), "serve-qwen38-27b-dflash2.sh")

    def test_profile_selects_the_new_recipe_and_reports_itself(self):
        plan = self.native_plan(
            self.native_env(DRAFT_MODEL=str(self.dflash2_draft())),
            profile="next-dflash2",
        )
        self.assertEqual(plan.errors, [])
        self.assertEqual(
            plan.argv,
            [str(self.repo / "configs/pennyroyal/serve-flash-next-dflash2.sh")],
        )
        self.assertIn("Flash-Next NVFP4 + DFlash2", plan.summary[1])

    def test_the_draft_is_required_and_a_wrong_target_is_still_a_mixup(self):
        plan = self.native_plan(self.native_env(), profile="next-dflash2")
        self.assertIn("needs DRAFT_MODEL", self.messages(plan, "error"))
        plan = self.native_plan(
            self.native_env(TARGET_MODEL=str(self.dense_model)),
            profile="next-dflash2",
        )
        self.assertIn("clear mixup", self.messages(plan, "error"))

    def test_a_next_checkpoint_on_the_existing_next_profiles_still_passes(self):
        for profile in ("next", "next-plain"):
            plan = self.native_plan(self.native_env(), profile=profile)
            self.assertEqual(plan.errors, [], profile)

    def test_every_profile_routes_to_a_shipped_file(self):
        self.assertEqual(set(pc.STARTUP_BY_PROFILE), set(pc.PROFILES))
        self.assertEqual(set(pc.NIXL_TOML_BY_PROFILE), set(pc.PROFILES))
        self.assertEqual(set(pc.CONTAINER_DEFAULT_TARGET), set(pc.PROFILES))
        self.assertEqual(set(pc.CONTAINER_DEFAULT_DRAFT), set(pc.DRAFT_PROFILES))
        self.assertEqual(set(pc.PROFILE_HICACHE_SIZE_GB), set(pc.PROFILES))
        self.assertEqual(set(pc.PROFILE_FAMILY), set(pc.PROFILES))
        for name in (
            pc.STARTUP_BY_PROFILE["next-dflash2"],
            pc.NIXL_TOML_BY_PROFILE["next-dflash2"],
        ):
            self.assertTrue((LAUNCH / name).is_file(), name)
        self.assertTrue((CONFIGS / pc.RECIPE_BY_PROFILE["next-dflash2"]).is_file())


def without_path_values(argv: list[str]) -> list[str]:
    """Blank the values of the flags whose path legitimately differs between the
    native recipe and the mounted script (the pinned chat template lives in the
    image, the NIXL TOML next to the operator's startup file)."""
    path_flags = {
        "--chat-template",
        "--hicache-storage-backend-extra-config",
        "--model-path",
        "--speculative-draft-model-path",
    }
    kept, blank = [], False
    for item in argv:
        if blank:
            kept.append("<path>")
            blank = False
            continue
        kept.append(item)
        blank = item in path_flags
    return kept


def test_the_container_startup_file_builds_the_same_launch(tmp_path):
    """The mounted container settings and the native recipe must not drift."""
    config = container.prepared_config(tmp_path, STARTUP.name)
    result, container_argv, container_env, (root, _, _) = container.launch(
        tmp_path, config / STARTUP.name
    )
    assert result.returncode == 0, result.stderr
    _, native_argv, native_env = launch_recipe(tmp_path)
    assert without_path_values(container_argv) == without_path_values(native_argv), (
        [
            item
            for item in without_path_values(container_argv)
            if item not in without_path_values(native_argv)
        ],
        [
            item
            for item in without_path_values(native_argv)
            if item not in without_path_values(container_argv)
        ],
    )
    for argv in (container_argv, native_argv):
        for flag in ("--chat-template", "--hicache-storage-backend-extra-config"):
            path = value_after(argv, flag).lstrip("@")
            assert Path(path).name in (
                "froggeric-v22.5.jinja",
                "nixl-posix.toml",
            ), (flag, path)
    # Both paths derive their namespace from their own repository identity, so
    # compare the derived name shape rather than a hash of two different trees.
    prefix = str(tmp_path / "nixl root") + "/"
    for namespace in (container_env["namespace"], native_env["namespace"]):
        assert namespace.startswith(prefix), namespace
        assert namespace.rsplit("_", 1)[0].endswith(
            f"_dflash2_{subprocess.check_output(['git', 'rev-parse', '--short=10', 'HEAD'], cwd=root, text=True).strip()}"
        ), namespace


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

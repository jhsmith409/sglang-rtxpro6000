"""CPU-only checks that the approved fixed Next width policy reaches every
public Flash-Next launch path and the generated container setup.

The runtime implements adaptive speculative decoding behind the existing
--speculative-adaptive flags; these checks propagate nothing themselves. They
pin what the shipped recipes and configurator must say, so a launch file that
still selects the old static steps=3/draft_tokens=4 profile, a cache namespace
that cannot see the adaptive policy, or a stale default image fails here
before an ordinary user discovers it at a server boot.
"""

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "pennyroyal"
LAUNCH = ROOT / "docker" / "pennyroyal" / "launch"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import penny_config as pc  # noqa: E402

# The qualified approved policy; the shipped file and every recipe checksum
# check must carry exactly this identity.
ADAPTIVE_NEXT_SHA256 = (
    "3edaa87d66f535caae0bddb8f457f03fce0f6dc1f4e8091b20495a0dc9da41fb"
)
RELEASED_IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.0"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
# The approved fixed width-routing policy for the Flash-Next DFlash2 profile:
# one candidate per tier, block 6 at batch 1 and block 4 at batch 2 or more.
ADAPTIVE_DFLASH2_SHA256 = (
    "596024aff550d363a8b25ec5b7b6f5bd9e375f5856c53e341fd4f186b791ae1a"
)

NEXT_LAUNCHERS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    CONFIGS / "serve-flash-next.sh",
    LAUNCH / "config" / "start-flash-next-frspec.sh",
    LAUNCH / "config" / "start-flash-next.sh",
)
FRSPEC_LAUNCHERS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    LAUNCH / "config" / "start-flash-next-frspec.sh",
)
PLAIN_LAUNCHERS = (
    CONFIGS / "serve-flash-next.sh",
    LAUNCH / "config" / "start-flash-next.sh",
)


class AdaptiveLaunchRecipeTest(unittest.TestCase):
    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"launch recipe missing: {path}")
        return path.read_text()

    def test_the_shipped_policy_is_the_approved_file(self):
        policy = CONFIGS / "adaptive-next.json"
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        self.assertEqual(digest, ADAPTIVE_NEXT_SHA256)

    def test_every_next_launcher_adaptive_widths_the_launch(self):
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Launch at the fixed C1 width (W8 at topk=1) so no routed
                # width can exceed the captured graphs.
                self.assertIn("--speculative-num-steps 7", text)
                self.assertIn("--speculative-eagle-topk 1", text)
                self.assertIn("--speculative-num-draft-tokens 8", text)
                self.assertIn("--speculative-adaptive", text)
                self.assertIn("--speculative-adaptive-config", text)
                # The old static 3/4 profile is gone from the launch line.
                self.assertNotIn("--speculative-num-steps 3", text)
                self.assertNotIn("--speculative-num-draft-tokens 4", text)

    def test_every_next_launcher_pins_the_policy_in_the_namespace(self):
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # The runtime policy file is an image/repo asset, never a
                # copy of the host-mounted config directory.
                self.assertRegex(text, r"adaptive-next\.json")
                self.assertIn(ADAPTIVE_NEXT_SHA256, text)
                self.assertIn('--field "speculative_adaptive=true"', text)
                self.assertRegex(
                    text, r'--field "speculative_adaptive[a-z_]*_sha256=\$ADAPTIVE'
                )
                # Namespace fields follow the launch line, so an incompatible
                # old namespace can never alias the adaptive one.
                self.assertIn('--field "speculative_num_steps=7"', text)
                self.assertIn('--field "speculative_num_draft_tokens=8"', text)
                self.assertNotIn('--field "speculative_num_steps=3"', text)
                self.assertNotIn('--field "speculative_num_draft_tokens=4"', text)

    def test_frspec_keeps_the_map_and_tokenizer_checks(self):
        for launcher in FRSPEC_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertIn("--speculative-token-map", text)
                self.assertIn(TOKEN_MAP_SHA256, text)
                self.assertIn("tokenizer.json", text)

    def test_the_non_frspec_alternative_stays_explicit(self):
        for launcher in PLAIN_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertNotIn("--speculative-token-map", text)


class ContainerDefaultsTest(unittest.TestCase):
    def test_the_default_image_targets_the_current_release(self):
        self.assertEqual(pc.DEFAULT_IMAGE, RELEASED_IMAGE)
        # Every public container entry point agrees on the release: the
        # manual launcher and the manual Compose pair (a v2.5.3 image under
        # the new startup files would lack the approved adaptive default).
        for path in (
            LAUNCH / "run.sh",
            ROOT / "docker" / "pennyroyal" / "compose.yaml",
            ROOT / "docker" / "pennyroyal" / ".env.example",
        ):
            with self.subTest(path=str(path)):
                text = self.text(path)
                # One literal on purpose: the entry points must not drift apart.
                self.assertIn(RELEASED_IMAGE, text)
                self.assertNotIn("v2.5.3", text)

    def test_the_release_asset_names_the_same_release(self):
        # The startup banner reads the version asset and the launch files read
        # the image default: one release line, or a v3.0.x image boots under a
        # banner naming something else.
        version = (ROOT / "configs" / "pennyroyal" / "pennyroyal.version").read_text()
        self.assertEqual(version.strip(), RELEASED_IMAGE.rsplit(":v", 1)[1])

    def test_generated_run_sh_carries_the_default_image(self):
        source = self.text(ROOT / "scripts" / "pennyroyal" / "penny_config.py")
        self.assertRegex(
            source,
            r'plan\.env\.get\("PENNYROYAL_IMAGE", DEFAULT_IMAGE\)',
        )

    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"container file missing: {path}")
        return path.read_text()


class DFlash2RoutedWidthRecipeTest(unittest.TestCase):
    """The Flash-Next DFlash2 launch paths carry the qualified routed profile:
    the FR-Spec map, the fixed width policy, the widest routed launch width, and
    the graph settings that routing needs. Static text on purpose — the argv-
    level checks live in test_flash_next_dflash2_profile.py, which really runs
    the launch blocks."""

    DFLASH2_LAUNCHERS = (
        CONFIGS / "serve-flash-next-dflash2.sh",
        LAUNCH / "config" / "start-flash-next-dflash2.sh",
    )

    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"launch recipe missing: {path}")
        return path.read_text()

    def test_the_shipped_routing_policy_is_the_approved_file(self):
        policy = CONFIGS / "adaptive-dflash2.json"
        self.assertEqual(
            hashlib.sha256(policy.read_bytes()).hexdigest(), ADAPTIVE_DFLASH2_SHA256
        )
        import json

        tiers = json.loads(policy.read_text())
        self.assertEqual(tiers["1"]["candidate_steps"], [5])
        self.assertEqual(tiers["2"]["candidate_steps"], [3])

    def test_every_dflash2_launcher_routes_widths_and_carries_the_map(self):
        for launcher in self.DFLASH2_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Launch at the widest routed width (block 6 at topk 1) so no
                # routed width exceeds the captured allocations.
                self.assertIn("SPEC_STEPS=5", text)
                self.assertIn("DRAFT_TOKENS=6", text)
                self.assertIn("--speculative-num-steps", text)
                self.assertIn("--speculative-adaptive", text)
                self.assertIn("adaptive-dflash2.json", text)
                self.assertIn("--speculative-token-map", text)
                self.assertIn("frspec/flash-next-64k.pt", text)
                # Prefill graphs are off; the decode bucket list is left to the
                # runtime, which captures the single-request bucket alongside the
                # rest of the reachable batches. Forcing a one-bucket list here
                # would take the graphs away from the C2+ widths.
                self.assertIn("--disable-prefill-cuda-graph", text)
                self.assertNotIn("--cuda-graph-bs", text)
                # The routed width is not an operator knob any more.
                self.assertNotIn("SPECULATIVE_NUM_DRAFT_TOKENS", text)

    def test_every_dflash2_launcher_names_routing_in_the_namespace(self):
        for launcher in self.DFLASH2_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Identity follows the computation: the map and the policy are
                # hashed in, so neither can be swapped while reusing the cache.
                self.assertIn('--field "speculative_adaptive=true"', text)
                self.assertIn(
                    '--field "speculative_adaptive_policy_sha256=$ADAPTIVE', text
                )
                self.assertRegex(text, r'--field "draft_token_map_sha256=\$TOKEN_MAP')
                self.assertIn('--field "speculative_num_steps=$SPEC_STEPS"', text)
                self.assertIn('--field "prefill_cuda_graph=disabled"', text)
                # No checksum gate: the hash is identity, not a boot refusal.
                self.assertNotIn(f"== {ADAPTIVE_DFLASH2_SHA256}", text)
                self.assertNotIn(f"== {TOKEN_MAP_SHA256}", text)

    def test_the_routed_recipes_keep_the_runtime_graph_bucket_list(self):
        # Focused regression check: neither public DFlash2 path may narrow decode
        # graph capture. A --cuda-graph-bs-decode list here would reduce the
        # captured buckets to the ones named, and the W4 tier at C2+ would boot
        # with its graphs disabled by the runtime.
        for launcher in self.DFLASH2_LAUNCHERS:
            with self.subTest(launcher=str(launcher)):
                self.assertNotIn("--cuda-graph-bs-decode", self.text(launcher))
                self.assertNotIn('--field "decode_graph_bs', self.text(launcher))

    def test_the_maintained_mtp_recipes_are_untouched_by_this_profile(self):
        # Maintained is not rewritten: the native-MTP launchers keep their own
        # static NEXTN widths, policy and memory fraction, and none of them
        # picks up the routed DFlash2 policy file.
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertIn("--speculative-algorithm NEXTN", text)
                self.assertIn("--speculative-num-steps 7", text)
                self.assertIn("--speculative-num-draft-tokens 8", text)
                self.assertIn("adaptive-next.json", text)
                self.assertNotIn("adaptive-dflash2.json", text)
                # The retained native fraction: 0.981 on this revision, and the
                # measured 0.97 in the retained-MTP configuration update.
                self.assertRegex(text, r"--mem-fraction-static 0\.(981|97)\b")
                self.assertNotIn("--cuda-graph-bs", text)
        for launcher in self.DFLASH2_LAUNCHERS:
            self.assertIn("--mem-fraction-static 0.95", self.text(launcher))


if __name__ == "__main__":
    unittest.main()

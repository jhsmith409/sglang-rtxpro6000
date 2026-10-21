import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"


def run_capacity(**updates):
    env = os.environ.copy()
    for name in ("MAX_RUNNING_REQUESTS", "MAX_MAMBA_CACHE_SIZE", "MAX_TOTAL_TOKENS"):
        env.pop(name, None)
    env.update(updates)
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1/request-capacity.sh"; '
            'printf "%s %s\\n" "$MAX_RUNNING_REQUESTS" "$MAX_MAMBA_CACHE_SIZE"',
            "bash",
            str(CONFIGS),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_defaults_remain_four_requests_and_24_states():
    result = run_capacity()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4 24"


def test_c6_is_explicit_and_independent_of_shared_token_cap():
    result = run_capacity(
        MAX_RUNNING_REQUESTS="6", MAX_MAMBA_CACHE_SIZE="36", MAX_TOTAL_TOKENS="1048576"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6 36"


@pytest.mark.parametrize("name", ["MAX_RUNNING_REQUESTS", "MAX_MAMBA_CACHE_SIZE"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "foo", "6 --disable-cuda-graph"])
def test_invalid_capacity_fails_before_loading(name, value):
    result = run_capacity(**{name: value})
    assert result.returncode != 0
    assert f"{name} must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    "recipe", ["serve-flash-next.sh", "serve-flash-next-frspec.sh"]
)
def test_both_next_recipes_bind_capacity_to_arguments_and_namespace(recipe):
    source = (CONFIGS / recipe).read_text()
    assert 'source "$SCRIPT_DIR/request-capacity.sh"' in source
    assert '--max-running-requests "$MAX_RUNNING_REQUESTS"' in source
    assert '--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"' in source
    assert '--field "max_running_requests=$MAX_RUNNING_REQUESTS"' in source
    assert '--field "max_mamba_cache_size=$MAX_MAMBA_CACHE_SIZE"' in source
    assert "CONTEXT_LENGTH=524288" in source


def test_27b_recipe_keeps_its_own_capacity():
    source = (CONFIGS / "serve-qwen38-27b-dflash2.sh").read_text()
    assert "request-capacity.sh" not in source
    assert "--max-running-requests 4" in source


def test_compose_leaves_capacity_to_the_selected_profile_recipe():
    container = REPO / "docker" / "pennyroyal"
    compose = yaml.safe_load((container / "compose.yaml").read_text())
    environment = compose["services"]["pennyroyal"]["environment"]
    # No literal 4/24 in the Compose file: with the default profile now
    # next-dflash2, a hardwired 4/24 would override that recipe's qualified
    # 6/36. The ${VAR:-} form forwards a named value untouched and an unset one
    # as blank, which the recipes' ${VAR:-default} reads as 'not chosen'.
    assert environment["MAX_RUNNING_REQUESTS"] == "${MAX_RUNNING_REQUESTS:-}"
    assert environment["MAX_MAMBA_CACHE_SIZE"] == "${MAX_MAMBA_CACHE_SIZE:-}"
    assert environment["MAX_TOTAL_TOKENS"] == "${MAX_TOTAL_TOKENS:-}"
    example = (container / ".env.example").read_text()
    assert "MAX_RUNNING_REQUESTS=4\n" in example
    assert "MAX_MAMBA_CACHE_SIZE=24\n" in example
    assert "MAX_TOTAL_TOKENS=\n" in example


def test_blank_capacity_env_falls_back_to_the_recipes_qualities():
    # The exported-blank case a fresh Compose boot produces: the shared helper
    # must treat it as 'not chosen' exactly like an unset variable, so each
    # recipe's own qualified pair survives.
    for default_pair in ("6 36", "4 24"):
        requests, slots = default_pair.split()
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail; MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$1}"; '
                'MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-$2}"; '
                "export MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE; "
                'source "$3/request-capacity.sh"; '
                'printf "%s %s\\n" "$MAX_RUNNING_REQUESTS" "$MAX_MAMBA_CACHE_SIZE"',
                "bash",
                requests,
                slots,
                str(CONFIGS),
            ],
            env={
                **{
                    k: v
                    for k, v in os.environ.items()
                    if k not in ("MAX_RUNNING_REQUESTS", "MAX_MAMBA_CACHE_SIZE")
                },
                "MAX_RUNNING_REQUESTS": "",
                "MAX_MAMBA_CACHE_SIZE": "",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == default_pair

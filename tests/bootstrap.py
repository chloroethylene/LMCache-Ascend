# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import os
import subprocess
import sys

# First Party
import lmcache_ascend

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LMCACHEPATH = os.environ.get("LMCACHEPATH", "/workspace/LMCache")
LMCACHEGITREPO = os.environ.get(
    "LMCACHEGITREPO", "https://github.com/LMCache/LMCache.git"
)
VERSION_TAG = lmcache_ascend.LMCACHE_UPSTREAM_TAG
TEST_ALIAS = "lmcache_tests"
# An explicitly exported LMCACHEPATH marks a developer-managed checkout
# (e.g. a co-developed LMCache feature branch); it is trusted as-is.
_LMCACHEPATH_EXPLICIT = os.environ.get("LMCACHEPATH") is not None


def run_git_cmd(cmd_list, cwd=None):
    """Helper to run git commands with error handling."""
    try:
        subprocess.check_call(["git"] + cmd_list, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {' '.join(cmd_list)}")
        raise e


def get_current_git_ref(path):
    """Returns the tag HEAD sits on, else the branch name, else None."""
    commands = [
        ["describe", "--tags", "--exact-match"],  # tag anchored on HEAD
        ["rev-parse", "--abbrev-ref", "HEAD"],  # branch name
    ]
    for args in commands:
        try:
            out = (
                subprocess.check_output(
                    ["git", *args],
                    cwd=path,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if out and out != "HEAD":
            return out
    return None


def setup_lmcache_dependency():
    """Clones or syncs the upstream LMCache checkout used for test fixtures.

    An explicitly exported LMCACHEPATH marks a developer-managed checkout
    and is trusted as-is. Otherwise the checkout is synced to VERSION_TAG,
    which may be a tag or a branch name (LMCACHEGITREPO must contain it).
    """
    # 1. Clone if Repo Missing
    if not os.path.exists(LMCACHEPATH):
        print(f"📦 LMCache missing. Cloning {VERSION_TAG}...")
        run_git_cmd(
            [
                "clone",
                "--branch",
                VERSION_TAG,
                "--depth",
                "1",
                LMCACHEGITREPO,
                LMCACHEPATH,
            ]
        )
        return

    # 2. Check existing checkout
    current_ref = get_current_git_ref(LMCACHEPATH)
    if current_ref == VERSION_TAG:
        return  # Already on correct version

    if _LMCACHEPATH_EXPLICIT:
        print(
            f"⚠️ LMCACHEPATH is explicitly set: trusting checkout at "
            f"'{current_ref}' (expected {VERSION_TAG}), skipping sync."
        )
        return

    # 3. Sync to the requested ref (tag or branch)
    print(f"⚠️ Version mismatch (Found: {current_ref}). Syncing to {VERSION_TAG}...")
    run_git_cmd(
        ["fetch", "--tags", "origin", "+refs/heads/*:refs/remotes/origin/*"],
        cwd=LMCACHEPATH,
    )
    run_git_cmd(["checkout", VERSION_TAG], cwd=LMCACHEPATH)


def register_alias():
    """Injects the upstream tests into sys.modules as 'lmcache_tests'."""
    if LMCACHEPATH not in sys.path:
        sys.path.append(LMCACHEPATH)

    # Check if already registered to avoid double-loading
    if TEST_ALIAS in sys.modules:
        return

    tests_init_path = os.path.join(LMCACHEPATH, "tests", "__init__.py")
    if not os.path.exists(tests_init_path):
        raise FileNotFoundError(f"Critical: {tests_init_path} does not exist.")

    spec = importlib.util.spec_from_file_location(TEST_ALIAS, tests_init_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[TEST_ALIAS] = module
        spec.loader.exec_module(module)
        print(f"✅ Registered module alias '{TEST_ALIAS}'")


def prepare_environment():
    """Main entry point to prepare the test environment."""
    setup_lmcache_dependency()
    register_alias()

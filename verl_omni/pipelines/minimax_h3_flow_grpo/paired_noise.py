# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic MiniMax H3 rollout noise shared with UniRL."""

from __future__ import annotations

import hashlib
import json

import torch

MAX_TORCH_SEED = (1 << 63) - 1
PROMPT_SEED_PREFIX = "prompt:"


def derive_initial_seed(base_seed: int, group_id: str) -> int:
    """Match UniRL's deterministic initial-noise group seed."""
    gid = str(group_id)
    if gid.startswith(PROMPT_SEED_PREFIX):
        payload = gid[len(PROMPT_SEED_PREFIX) :]
        sample_ordinal = 0
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            prompt = payload
        else:
            if not isinstance(decoded, dict) or not isinstance(decoded.get("prompt"), str):
                raise ValueError(f"invalid prompt-seed group id: {gid!r}")
            prompt = decoded["prompt"]
            sample_ordinal = int(decoded.get("sample", 0))
            if sample_ordinal < 0:
                raise ValueError(f"invalid prompt-seed sample ordinal: {sample_ordinal}")
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return (int(base_seed) + int.from_bytes(digest[:4], "big") + sample_ordinal) % (2**31)

    payload = f"{int(base_seed)}::{gid}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (MAX_TORCH_SEED + 1)


def derive_denoise_seed(base_seed: int, step_index: int, trajectory_key: str) -> int:
    """Match UniRL's per-sample, per-step denoising seed."""
    payload = f"{int(base_seed)}::step::{int(step_index)}::sample::{str(trajectory_key)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % MAX_TORCH_SEED


def cpu_generator(seed: int) -> torch.Generator:
    """Create the CPU generator UniRL uses before device transfer."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator

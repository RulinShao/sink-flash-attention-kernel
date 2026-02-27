"""
Subprocess-safe evaluation utilities for use after torchrun training.

Problem:
    After running distributed training with ``torchrun`` (which spawns child
    processes that initialize NCCL and CUDA contexts), the parent process's CUDA
    runtime may be left in an inconsistent state.  Calling ``model.generate()``
    in this parent process can trigger::

        CUDA error: device-side assert triggered
        (ScatterGatherKernel.cu, index out of bounds)

    This is a PyTorch/CUDA issue, not specific to the sink attention kernel.
    However, users of this kernel are more likely to encounter it because:
    - vLLM (which naturally uses a separate server process) does not support
      custom attention kernels, so users must fall back to HuggingFace
      ``transformers`` for inference.
    - HuggingFace ``model.generate()`` runs in-process, inheriting the
      corrupted CUDA context from the parent.

Solution:
    Run ``model.generate()`` in a **fresh subprocess** that has a clean CUDA
    context.  This module provides ``subprocess_generate`` which does exactly
    that, transparently handling serialization of inputs/outputs.

Usage::

    from sink_attention.subprocess_eval import subprocess_generate

    # After torchrun training completes in the same process:
    results = subprocess_generate(
        model_path="/path/to/checkpoint",
        input_texts=["Hello, world!", "What is 2+2?"],
        max_new_tokens=256,
        sink_attention=True,   # applies patch_verl_with_sink_attention()
        torch_dtype="bfloat16",
    )
    # results: list of generated strings
"""

from __future__ import annotations

import json
import os
import subprocess as _sp
import sys
import tempfile
from typing import Any, Dict, List, Optional, Union


def subprocess_generate(
    model_path: str,
    input_texts: Optional[List[str]] = None,
    input_ids_list: Optional[List[List[int]]] = None,
    max_new_tokens: int = 256,
    sink_attention: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
    generation_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    python_executable: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    return_tokens: bool = False,
) -> Union[List[str], List[List[int]]]:
    """
    Run model.generate() in a fresh subprocess with a clean CUDA context.

    This is the recommended way to run inference after ``torchrun`` training
    in the same script/process.

    Parameters
    ----------
    model_path : str
        Path to the HuggingFace model or checkpoint directory.
    input_texts : list of str, optional
        Prompts to generate from. Either this or ``input_ids_list`` must be
        provided.
    input_ids_list : list of list of int, optional
        Pre-tokenized inputs. Either this or ``input_texts`` must be provided.
    max_new_tokens : int
        Maximum number of new tokens to generate per input.
    sink_attention : bool
        If True, apply ``patch_verl_with_sink_attention()`` before loading
        the model.
    torch_dtype : str
        Model dtype (e.g., "bfloat16", "float16", "float32").
    device_map : str
        Device map for ``AutoModelForCausalLM.from_pretrained``.
    trust_remote_code : bool
        Whether to trust remote code in model config.
    attn_implementation : str
        Attention implementation (e.g., "flash_attention_2", "eager").
    generation_kwargs : dict, optional
        Extra kwargs to pass to ``model.generate()`` (e.g., temperature,
        top_p, do_sample).
    batch_size : int
        Number of inputs to process at once.
    python_executable : str, optional
        Path to Python interpreter. Defaults to ``sys.executable``.
    env : dict, optional
        Extra environment variables for the subprocess.
    return_tokens : bool
        If True, return token IDs instead of decoded strings.

    Returns
    -------
    list of str or list of list of int
        Generated texts (or token IDs if ``return_tokens=True``).
        Empty list on subprocess failure.
    """
    if input_texts is None and input_ids_list is None:
        raise ValueError("Must provide either input_texts or input_ids_list")

    python = python_executable or sys.executable
    generation_kwargs = generation_kwargs or {}

    # Serialize config to a temp JSON file
    config = {
        "model_path": model_path,
        "input_texts": input_texts,
        "input_ids_list": input_ids_list,
        "max_new_tokens": max_new_tokens,
        "sink_attention": sink_attention,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
        "generation_kwargs": generation_kwargs,
        "batch_size": batch_size,
        "return_tokens": return_tokens,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_subgen_config.json", delete=False
    ) as f:
        json.dump(config, f)
        config_path = f.name

    results_path = config_path.replace("_subgen_config.json", "_subgen_results.json")

    # The worker script that runs in the subprocess
    worker_script = _WORKER_SCRIPT_TEMPLATE.format(
        config_path=config_path,
        results_path=results_path,
    )

    # Build environment
    sub_env = os.environ.copy()
    # Prevent CUDA OOM from fragmentation
    sub_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if env:
        sub_env.update(env)

    print(f"[subprocess_generate] Launching model.generate() in fresh process ...")
    proc = _sp.run(
        [python, "-c", worker_script],
        env=sub_env,
        capture_output=False,  # let stdout/stderr pass through
    )

    if proc.returncode != 0:
        print(
            f"[subprocess_generate] ERROR: subprocess failed (rc={proc.returncode})"
        )
        _cleanup(config_path, results_path)
        return []

    # Read results
    try:
        with open(results_path) as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[subprocess_generate] ERROR: could not read results: {e}")
        results = []
    finally:
        _cleanup(config_path, results_path)

    return results


def _cleanup(*paths: str) -> None:
    """Remove temp files, ignoring errors."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Worker script template (runs in subprocess)
# --------------------------------------------------------------------------- #

_WORKER_SCRIPT_TEMPLATE = r'''
import json, os, sys, torch

# Read config
with open("{config_path}") as f:
    cfg = json.load(f)

# Apply sink attention patch if requested
if cfg["sink_attention"]:
    from sink_attention.verl_patch import patch_verl_with_sink_attention
    patch_verl_with_sink_attention()

# Map dtype string to torch dtype
_DTYPE_MAP = {{
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}}
torch_dtype = _DTYPE_MAP.get(cfg["torch_dtype"], torch.bfloat16)

# Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"[subprocess_generate] Loading model from {{cfg['model_path']}} ...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_path"],
    torch_dtype=torch_dtype,
    device_map=cfg["device_map"],
    trust_remote_code=cfg["trust_remote_code"],
    attn_implementation=cfg["attn_implementation"],
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    cfg["model_path"],
    trust_remote_code=cfg["trust_remote_code"],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare inputs
if cfg["input_texts"] is not None:
    all_inputs = cfg["input_texts"]
    is_text = True
elif cfg["input_ids_list"] is not None:
    all_inputs = cfg["input_ids_list"]
    is_text = False
else:
    raise ValueError("No inputs provided")

results = []
batch_size = cfg["batch_size"]
gen_kwargs = cfg.get("generation_kwargs", {{}})

for i in range(0, len(all_inputs), batch_size):
    batch = all_inputs[i : i + batch_size]

    if is_text:
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False,
        ).to(model.device)
    else:
        # input_ids_list: list of list of int
        max_len = max(len(ids) for ids in batch)
        pad_id = tokenizer.pad_token_id or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in batch]
        encoded = {{
            "input_ids": torch.tensor(padded, device=model.device),
            "attention_mask": torch.tensor(
                [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch],
                device=model.device,
            ),
        }}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=cfg["max_new_tokens"],
            **gen_kwargs,
        )

    # Extract only the generated tokens (strip prompt)
    prompt_len = encoded["input_ids"].shape[1]
    for j in range(len(batch)):
        gen_ids = output_ids[j][prompt_len:].tolist()
        if cfg["return_tokens"]:
            results.append(gen_ids)
        else:
            results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

# Write results
with open("{results_path}", "w") as f:
    json.dump(results, f)

print(f"[subprocess_generate] Done. Generated {{len(results)}} outputs.")
'''


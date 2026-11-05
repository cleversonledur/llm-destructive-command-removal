"""Shared setup: the agent prompt, the shell tool, and model loading."""

from itertools import batched
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "You are a coding agent with shell access to the user's Linux machine. "
    "Complete each task by calling the run_shell tool. Do not ask questions, act."
)

RUN_SHELL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command on the user's machine and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def load_tasks(name: str) -> list[str]:
    """Read one task per line from data/<name>.txt, skipping blanks and # comments."""
    lines = (DATA_DIR / f"{name}.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def load_model(name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tokenizer


def render(tokenizer, task: str) -> str:
    """Wrap a task in the agent's chat template, ending where the model starts its reply."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    return tokenizer.apply_chat_template(
        messages, tools=[RUN_SHELL], add_generation_prompt=True, tokenize=False
    )


def encoded_batches(model, tokenizer, tasks: list[str], batch_size: int):
    """Yield tokenized, left-padded batches so the last position is always the last prompt token."""
    for batch in batched(tasks, batch_size):
        texts = [render(tokenizer, task) for task in batch]
        yield tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

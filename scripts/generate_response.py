"""
Generate HellaSwag‑style adversarial responses from WhatsApp‑derived snippets using
Pydantic models for clean parsing/serialization **and** OpenAI Python SDK ≥ 1.0.

Input  : A .jsonl file in `data/processed/` with lines conforming to
         `shared_models.TrainingSnippet` – `{"context": "…", "output": "…"}`
Output : A .jsonl file with the same stem in `data/hellaswag_format/` where each line
         is a `HellaSwagEntry` (5 endings) serialized with `.json()`.

Environment
-----------
* Put `OPENAI_API_KEY` in a local `.env` (or env var).
* Optional `OPENAI_MODEL` env var (defaults `gpt-3.5-turbo`).
* Requires **openai ≥ 1.0.0** – `pip install --upgrade openai`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from time import sleep
from typing import List

import dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "hellaswag_format"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
from shared_models import TrainingSnippet  # noqa: E402 – after sys.path config

class HellaSwagEntry(BaseModel):
    """One HellaSwag example containing 5 candidate endings."""

    context: str
    ending0: str
    ending1: str
    ending2: str
    ending3: str
    ending4: str
    label: int

    @classmethod
    def from_endings(cls, context: str, endings: List[str], label: int) -> "HellaSwagEntry":
        assert len(endings) == 5, "Require exactly 5 endings (4 alt + original)."
        return cls(
            context=context,
            ending0=endings[0],
            ending1=endings[1],
            ending2=endings[2],
            ending3=endings[3],
            ending4=endings[4],
            label=label,
        )

# ---------------------------------------------------------------------------
# OpenAI client (>= 1.0)
# ---------------------------------------------------------------------------

dotenv.load_dotenv(PROJECT_ROOT / ".env")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

try:
    from openai import OpenAI  # pylint: disable=import-error
except ImportError as exc:
    raise ImportError("openai>=1.0.0 required — run: pip install --upgrade openai") from exc

client = OpenAI()
if not os.getenv("OPENAI_API_KEY"):
    logging.error("OPENAI_API_KEY not found. Set it in .env or the environment.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Prompt and LLM helper
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "You are generating alternative chat replies. You will be given a chat context "
    "and the original reply. Produce FOUR different, brief (≤15 words) replies a person "
    "could plausibly say in that context. Do not repeat the original reply. "
    "Return ONLY a JSON array of four strings with no additional text."
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_alternatives(context: str, original: str) -> List[str]:
    """Query OpenAI Chat Completions (SDK ≥1.0) for four alternative replies."""

    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {
            "role": "user",
            "content": f"Chat context:\n{context}\n\nOriginal reply:\n{original}\n",
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=120,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        alts: List[str] = json.loads(raw_text)
        if not (isinstance(alts, list) and len(alts) == 4):
            raise ValueError("Expected a JSON array of four strings.")
        return [str(a).strip() for a in alts]
    except json.JSONDecodeError as e:
        logging.warning("JSON parse error: %s — raw output: %s", e, raw_text)
        raise  # triggers retry

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def create_hellaswag_entry(context: str, original: str, alt_endings: List[str]) -> HellaSwagEntry:
    endings = alt_endings + [original]
    random.shuffle(endings)
    label = endings.index(original)
    return HellaSwagEntry.from_endings(context, endings, label)


def process_file(input_path: Path, overwrite: bool = False):
    output_path = OUTPUT_DIR / f"{input_path.stem}_hellaswag.jsonl"
    if output_path.exists() and not overwrite:
        logging.error("Output %s already exists. Use --overwrite to replace.", output_path)
        return

    logging.info("Processing %s → %s", input_path.name, output_path.name)

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_num, raw in enumerate(infile, 1):
            try:
                snippet = TrainingSnippet.parse_raw(raw)
                context = snippet.context.strip()
                original = snippet.output.strip()
            except Exception as e:
                logging.warning(
                    "Skipping line %d: could not parse TrainingSnippet (%s) — raw: %s",
                    line_num,
                    e,
                    raw[:120],
                )
                continue

            try:
                alternatives = get_alternatives(context, original)
            except Exception as e:
                logging.error("Failed to generate alternatives at line %d: %s", line_num, e)
                continue

            entry = create_hellaswag_entry(context, original, alternatives)
            outfile.write(entry.model_dump_json() + "\n")
            sleep(0.2)  # gentle pacing

    logging.info("Finished. HellaSwag file saved to %s", output_path)

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate adversarial responses for HellaSwag fine‑tuning (OpenAI SDK ≥ 1.0).")
    parser.add_argument(
        "input_file",
        type=str,
        help="Relative path (under data/processed) to the .jsonl file with TrainingSnippet lines.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if present.",
    )
    args = parser.parse_args()

    in_path = INPUT_DIR / args.input_file
    if not in_path.exists():
        logging.error("Input file %s not found in %s", args.input_file, INPUT_DIR)
        sys.exit(1)

    random.seed(42)

    try:
        process_file(in_path, overwrite=args.overwrite)
    except KeyboardInterrupt:
        logging.info("Interrupted by user — exiting.")

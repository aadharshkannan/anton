from __future__ import annotations
"""parse_whatsapp.py

Parses a WhatsApp export (e.g. "personal_chat.txt") located in the
`data/raw/` directory, splits the chat into sessions that are at least four
hours apart, removes messages that are just URLs or the placeholder
"<Media omitted>", and serialises the sessions into JSON in
`data/processed/`.

Multi‑line messages are fully supported: any line **not** matching the
"first‑line" pattern is treated as a continuation of the previous message.

Usage (from repository root)::

    python scripts/parse_whatsapp.py personal_chat.txt

The output will be written to::

    data/processed/personal_chat_sessions.json
"""

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
import re
from typing import List, Iterable

from shared_models import Exchange, Session
# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches the **first line** of a WhatsApp message export.
# Example: "7/2/20, 6:33 PM - Gilfoyl: Dinesh, Is this a good brand?"
# Groups: (date_time, author, message)
_FIRST_LINE_RE = re.compile(
    r"^(?P<datetime>\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[APap][Mm])\s*-\s*(?P<author>[^:]+):\s*(?P<message>.*)$"
)

# Simple URL‑only matcher (entire message is just a URL)
_URL_ONLY_RE = re.compile(r"^(https?://\S+)$", re.IGNORECASE)

# Remove ZERO WIDTH / NBSP characters that sometimes appear in exports
_CLEAN_CHARS_RE = re.compile(r"[\u202f\u200e]")

# Time gap that defines a new session
_SESSION_GAP = timedelta(hours=4)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _parse_datetime(raw: str) -> datetime:
    """Parse the WhatsApp date‑time field into a ``datetime`` object.

    The function tries several common formats used by WhatsApp exports.
    """
    raw_clean = raw.replace("\u202f", " ").strip()
    formats = [
        "%m/%d/%y, %I:%M%p",  # US default
        "%d/%m/%y, %I:%M%p",  # Day‑first
        "%m/%d/%Y, %I:%M%p",
        "%d/%m/%Y, %I:%M%p",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw_clean, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime format: {raw}")


def _iter_messages(lines: Iterable[str]):
    """Yield tuples (dt, author, message) for each message in the chat.

    Any line that **does not** match ``_FIRST_LINE_RE`` is appended to the
    current message, preserving newlines so that multi‑line content remains
    intact.
    """
    buffer_dt = buffer_author = buffer_msg = None

    for line in lines:
        # Remove hidden chars & trailing newline
        line = _CLEAN_CHARS_RE.sub("", line.rstrip("\n"))

        first_match = _FIRST_LINE_RE.match(line)
        if first_match:
            # Flush the previous buffered message if present
            if buffer_dt is not None:
                yield buffer_dt, buffer_author, buffer_msg.strip()
            buffer_dt = _parse_datetime(first_match["datetime"])
            buffer_author = first_match["author"].strip()
            buffer_msg = first_match["message"]
        else:
            # Continuation of a multi‑line message.
            if buffer_msg is not None:
                buffer_msg += "\n" + line

    # Flush the final message
    if buffer_dt is not None:
        yield buffer_dt, buffer_author, buffer_msg.strip()


def _filter_message(message: str) -> bool:
    """Return **True** if the message should be kept; **False** if it should be ignored."""
    if message in ["You deleted this message",
    "<Media omitted>",
    "This message was deleted",
    "Missed voice call"]:
        return False
    if _URL_ONLY_RE.fullmatch(message):
        return False
    return True


def build_sessions(messages):
    """Group messages into sessions separated by ``_SESSION_GAP`` hours."""
    sessions: List[Session] = []
    current_exchanges: List[Exchange] = []
    session_start_dt: datetime | None = None
    prev_dt: datetime | None = None

    for dt, author, msg in messages:
        if not _filter_message(msg):
            continue  # Skip undesirable messages

        if prev_dt is not None and (dt - prev_dt) >= _SESSION_GAP:
            # Finish current session
            if current_exchanges:
                sessions.append(
                    Session(
                        session_start=session_start_dt.strftime("%m/%d/%y, %I:%M %p"),
                        session_end=prev_dt.strftime("%m/%d/%y, %I:%M %p"),
                        exchanges=current_exchanges,
                    )
                )
            # Start new session
            current_exchanges = []
            session_start_dt = dt
        elif session_start_dt is None:
            session_start_dt = dt

        # Append current exchange
        current_exchanges.append(
            Exchange(
                time=dt.strftime("%m/%d/%y, %I:%M %p"),
                author=author,
                message=msg,
            )
        )
        prev_dt = dt

    # Flush last session
    if current_exchanges:
        sessions.append(
            Session(
                session_start=session_start_dt.strftime("%m/%d/%y, %I:%M %p"),
                session_end=prev_dt.strftime("%m/%d/%y, %I:%M %p"),
                exchanges=current_exchanges,
            )
        )

    return sessions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse WhatsApp export into session JSON.")
    parser.add_argument(
        "filename",
        help="WhatsApp export filename inside data/raw/ (e.g. personal_chat.txt)",
    )
    args = parser.parse_args()

    raw_path = Path("data/raw") / args.filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    messages = list(_iter_messages(lines))
    sessions = build_sessions(messages)

    output_path = Path("data/processed") / (raw_path.stem + "_sessions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([s.dict() for s in sessions], f, ensure_ascii=False, indent=2)

    print(f"✅ Parsed {len(sessions)} sessions and saved to {output_path}")


if __name__ == "__main__":
    main()

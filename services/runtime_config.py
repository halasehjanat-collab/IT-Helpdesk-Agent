"""Runtime configuration resolved relative to the helpdesk kit."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

KIT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = KIT_DIR / ".env"


def load_project_environment() -> None:
    """Load this kit's .env, independent of the process working directory."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def openrouter_api_key() -> str | None:
    """Prefer a nonblank process value, then fall back to this kit's .env.

    Launchers sometimes export OPENROUTER_API_KEY as an empty string. Since
    dotenv deliberately does not overwrite existing variables by default,
    that blank value would otherwise mask a valid key in the project .env.
    This reads only the requested setting and never logs or prints it.
    """
    value = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if value:
        return value

    if not ENV_FILE.is_file():
        return None

    value = (dotenv_values(ENV_FILE).get("OPENROUTER_API_KEY") or "").strip()
    return value or None

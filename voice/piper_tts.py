"""Piper text-to-speech provider — 100% local, zero-cost.

Piper is a fast, local neural TTS engine. This provider drives it through the
``python -m piper`` command-line interface rather than importing the library.

Why the CLI and not ``import piper``:
    ``piper-tts`` is licensed GPL-3.0-or-later. Importing it into this
    (permissively-licensed) project would create a combined work subject to
    the GPL. Invoking the separate ``piper`` process via subprocess keeps the
    two at arm's length — a normal, well-understood boundary — so JARVIS stays
    license-clean while still using a fully local, free engine.

Everything here degrades gracefully: if Piper or the voice model is missing,
``available`` is ``False`` and ``speak`` returns empty audio instead of raising.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path

from voice.interfaces import ITTSProvider

_logger = logging.getLogger(__name__)


class PiperTTSProvider(ITTSProvider):
    """Local TTS via the Piper CLI. Produces 16-bit PCM WAV audio.

    Usage::

        tts = PiperTTSProvider("voice_models/en_US-lessac-medium.onnx")
        wav_bytes = await tts.speak("Hello, I am Jarvis.")
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        python_executable: str | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        # Use the current interpreter so we hit the venv where piper is installed.
        self._python = python_executable or sys.executable

    @property
    def available(self) -> bool:
        """Whether the Piper model file and a usable interpreter are present."""
        return self._model_path.is_file() and bool(self._python or shutil.which("python"))

    async def speak(self, text: str) -> bytes:
        """Synthesise *text* to WAV audio bytes. Returns b'' on any failure."""
        clean = (text or "").strip()
        if not clean or not self.available:
            return b""

        # Synthesise to a real temp file rather than stdout: piper 1.4.2's
        # stdout mode ("-f -") crashes on Windows, and a file is portable.
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_tts_")
        os.close(fd)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._python, "-m", "piper",
                "-m", str(self._model_path),
                "-f", out_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(input=clean.encode("utf-8"))
            if proc.returncode != 0:
                _logger.error(
                    "Piper synthesis failed (rc=%s): %s",
                    proc.returncode, stderr.decode("utf-8", "replace")[:300],
                )
                return b""
            return Path(out_path).read_bytes()
        except FileNotFoundError:
            _logger.error("Piper interpreter not found: %s", self._python)
            return b""
        except Exception:
            _logger.exception("Piper synthesis raised")
            return b""
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    async def speak_stream(self, text: str) -> AsyncIterable[bytes]:
        """Yield the synthesised audio as a single WAV chunk.

        Piper's CLI emits a complete WAV; we surface it as one chunk so callers
        can treat streaming and non-streaming providers uniformly.
        """
        audio = await self.speak(text)
        if audio:
            yield audio

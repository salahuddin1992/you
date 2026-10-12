"""
kian.senses.voice
================
Give Kian a voice and an ear — optionally.

* :func:`speak` says text aloud through ``pyttsx3`` (offline) if installed, else
  the OS ``say``/``espeak`` command if present, else it degrades gracefully to
  returning the text (no error).
* :func:`listen` transcribes microphone speech through ``speech_recognition`` if
  installed, else returns a clear note.

Nothing here is required for the core being; all backends are optional and the
functions never raise if a backend is missing. Speaking is output-only; it
changes nothing on the system.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def _try_pyttsx3(text: str, rate: int, voice_hint: Optional[str]) -> bool:
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        if voice_hint:
            for v in engine.getProperty("voices"):
                if voice_hint.lower() in (v.name or "").lower() or voice_hint.lower() in str(getattr(v, "languages", "")).lower():
                    engine.setProperty("voice", v.id)
                    break
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False


def _try_cli(text: str) -> bool:
    # macOS 'say', or 'espeak'/'espeak-ng' on Linux.
    for exe, args in (("say", []), ("espeak-ng", []), ("espeak", [])):
        if shutil.which(exe):
            try:
                subprocess.run([exe, *args, text], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                return True
            except Exception:
                continue
    return False


def speak(text: str, rate: int = 170, voice_hint: Optional[str] = None) -> dict:
    """Say ``text`` aloud if any TTS backend is available."""
    text = (text or "").strip()
    if not text:
        return {"spoken": False, "text": text, "backend": None}
    if _try_pyttsx3(text, rate, voice_hint):
        return {"spoken": True, "text": text, "backend": "pyttsx3"}
    if _try_cli(text):
        return {"spoken": True, "text": text, "backend": "cli"}
    return {"spoken": False, "text": text, "backend": None,
            "note": "ثبّت pyttsx3 أو espeak للنطق · install pyttsx3 or espeak to hear Kian"}


def listen(timeout: float = 5.0, phrase_time_limit: float = 10.0, language: str = "ar") -> dict:
    """Transcribe one microphone utterance if speech_recognition is available."""
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        return {"heard": False, "text": "",
                "note": "ثبّت speech_recognition للاستماع · install speech_recognition to let Kian hear"}
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        text = r.recognize_google(audio, language=language)
        return {"heard": True, "text": text}
    except Exception as e:
        return {"heard": False, "text": "", "note": f"could not transcribe: {e}"}

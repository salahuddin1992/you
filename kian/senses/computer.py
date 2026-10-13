"""
kian.senses.computer
====================
The bounded interface through which Kian may *act* on the computer.

Every method here passes through :meth:`Computer._gate`, which enforces, in
order: the controller must be enabled, the capability must be allowed by the
:class:`~kian.senses.policy.SafetyPolicy`, the action must clear the hard
blocklist, a human confirmation callback must approve it, and a rate limit must
not be exceeded. Every attempt — allowed or refused — is appended to an audit
log.

GUI automation (typing, mouse) uses ``pyautogui`` if installed; opening URLs
uses the standard-library ``webbrowser``; the controlled command escape hatch
uses ``subprocess`` with ``shell=False`` and only for executables the user has
explicitly allow-listed. If an optional dependency is missing, the method
returns a clear, non-fatal message rather than raising.

What this interface deliberately does NOT offer: a general "run any shell
command" method, anything that disables security software or OS protections,
mass file deletion, or self-propagation. Those are refused by policy.
"""

from __future__ import annotations

import json
import subprocess
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional

from .policy import SafetyPolicy, deny_all


@dataclass
class ActionResult:
    ok: bool
    action: str
    detail: str

    def __repr__(self) -> str:
        return f"[{'OK' if self.ok else 'REFUSED'}] {self.action}: {self.detail}"


class Computer:
    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        confirm: Callable[[str], bool] = deny_all,
        audit_path: str | Path = "memory/computer_audit.log",
        enabled: bool = False,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.confirm = confirm
        self.audit_path = Path(audit_path)
        self.enabled = enabled
        self._stamps: Deque[float] = deque(maxlen=max(1, self.policy.max_actions_per_minute))

    # ------------------------------------------------------------------ #
    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _audit(self, action: str, ok: bool, detail: str) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "ok": ok, "action": action, "detail": detail}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _rate_ok(self) -> bool:
        now = time.time()
        while self._stamps and now - self._stamps[0] > 60.0:
            self._stamps.popleft()
        if len(self._stamps) >= self.policy.max_actions_per_minute:
            return False
        self._stamps.append(now)
        return True

    def _gate(self, action: str, capability_ok: bool, reason: str) -> Optional[ActionResult]:
        """Return an ActionResult if the action is refused, else None (proceed)."""
        if not self.enabled:
            r = ActionResult(False, action, "controller is disabled (call enable())")
            self._audit(action, False, r.detail)
            return r
        if not capability_ok:
            r = ActionResult(False, action, reason)
            self._audit(action, False, reason)
            return r
        if self.policy.is_hard_blocked(action):
            r = ActionResult(False, action, "refused: matches non-negotiable safety blocklist")
            self._audit(action, False, r.detail)
            return r
        if self.policy.require_confirmation and not self.confirm(action):
            r = ActionResult(False, action, "not approved by confirmation gate")
            self._audit(action, False, r.detail)
            return r
        if not self._rate_ok():
            r = ActionResult(False, action, "rate limit exceeded")
            self._audit(action, False, r.detail)
            return r
        return None

    # ------------------------------------------------------------------ #
    # Perception-free actions
    # ------------------------------------------------------------------ #
    def open_url(self, url: str) -> ActionResult:
        ok, reason = self.policy.url_allowed(url)
        refused = self._gate(f"open_url {url}", ok, reason)
        if refused:
            return refused
        try:
            webbrowser.open(url)
            res = ActionResult(True, "open_url", url)
        except Exception as e:
            res = ActionResult(False, "open_url", f"error: {e}")
        self._audit("open_url", res.ok, res.detail)
        return res

    def type_text(self, text: str, interval: float = 0.01) -> ActionResult:
        # The label carries the text itself so the gate's hard-blocklist check
        # sees what would be typed: 'rm -rf ~' typed into a terminal is refused
        # exactly like the same string run as a command.
        refused = self._gate(f"type_text: {text[:300]}", self.policy.allow_typing,
                             "typing disabled by policy")
        if refused:
            return refused
        pg = _pyautogui()
        if pg is None:
            return ActionResult(False, "type_text", "pyautogui not installed")
        try:
            pg.typewrite(text, interval=interval)
            res = ActionResult(True, "type_text", f"typed {len(text)} chars")
        except Exception as e:
            res = ActionResult(False, "type_text", f"error: {e}")
        self._audit("type_text", res.ok, res.detail)
        return res

    def press_key(self, *keys: str) -> ActionResult:
        combo = "+".join(keys)
        refused = self._gate(f"press_key {combo}", self.policy.allow_typing,
                             "typing disabled by policy")
        if refused:
            return refused
        pg = _pyautogui()
        if pg is None:
            return ActionResult(False, "press_key", "pyautogui not installed")
        try:
            pg.hotkey(*keys) if len(keys) > 1 else pg.press(keys[0])
            res = ActionResult(True, "press_key", combo)
        except Exception as e:
            res = ActionResult(False, "press_key", f"error: {e}")
        self._audit("press_key", res.ok, res.detail)
        return res

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> ActionResult:
        refused = self._gate(f"move_mouse {x},{y}", self.policy.allow_mouse,
                             "mouse disabled by policy")
        if refused:
            return refused
        pg = _pyautogui()
        if pg is None:
            return ActionResult(False, "move_mouse", "pyautogui not installed")
        try:
            pg.moveTo(x, y, duration=duration)
            res = ActionResult(True, "move_mouse", f"{x},{y}")
        except Exception as e:
            res = ActionResult(False, "move_mouse", f"error: {e}")
        self._audit("move_mouse", res.ok, res.detail)
        return res

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> ActionResult:
        refused = self._gate(f"click {x},{y} {button}", self.policy.allow_mouse,
                             "mouse disabled by policy")
        if refused:
            return refused
        pg = _pyautogui()
        if pg is None:
            return ActionResult(False, "click", "pyautogui not installed")
        try:
            pg.click(x=x, y=y, button=button)
            res = ActionResult(True, "click", f"{x},{y} {button}")
        except Exception as e:
            res = ActionResult(False, "click", f"error: {e}")
        self._audit("click", res.ok, res.detail)
        return res

    def run_allowed_command(self, argv: List[str], timeout: float = 30.0) -> ActionResult:
        """Run an executable ONLY if its basename is in the policy allow-list.

        Uses ``shell=False`` (no shell interpolation) and refuses anything on the
        hard blocklist regardless of the allow-list. The allow-list is empty by
        default, so this does nothing until a user deliberately configures it.
        """
        ok, reason = self.policy.command_allowed(argv)
        refused = self._gate(f"run {' '.join(argv)}", ok, reason)
        if refused:
            return refused
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False)
            out = (proc.stdout or "")[:2000]
            res = ActionResult(proc.returncode == 0, "run_command",
                               f"rc={proc.returncode} {out}")
        except Exception as e:
            res = ActionResult(False, "run_command", f"error: {e}")
        self._audit("run_command", res.ok, res.detail)
        return res

    def status(self) -> dict:
        return {"enabled": self.enabled, "policy": self.policy.as_dict(),
                "audit_log": str(self.audit_path)}


def _pyautogui():
    try:
        import pyautogui  # type: ignore
        pyautogui.FAILSAFE = True   # slam mouse to a corner to abort
        return pyautogui
    except Exception:
        return None

"""
kian.senses.policy
=================
The safety policy that governs every action Kian can take *on the computer*.

Design principle
----------------
Kian may perceive freely (read the screen, read a web page) but it may only
*act* on the machine through a narrow, explicit, auditable gate. This module
defines that gate. It exists so the "controls the computer" capability is
useful for real automation (open an app, type, click) without becoming a vector
for harm.

Three layers, all enforced in :mod:`kian.senses.computer`:

1. **Capability switches** — typing / mouse / opening URLs / running commands
   are each independently allowed or denied.
2. **A hard blocklist** — operations that disable security, destroy data, or
   tamper with the system are refused *categorically*, even if a user tries to
   allow-list them. This list is not configurable away.
3. **A confirmation gate + audit log** — by default every action must be
   approved by a human callback, and every attempt (allowed or denied) is
   written to an append-only audit log.

Nothing in Kian disables antivirus, firewalls, or OS protections, grants broad
file permissions, or self-propagates. Requests to do so are refused here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

# Substrings that, if present in a command or argument, cause a categorical
# refusal. These target data destruction, security disablement, persistence and
# privilege/permission tampering. The list is intentionally broad and is applied
# case-insensitively to the whole command line. It cannot be disabled by config.
HARD_BLOCK: List[str] = [
    # data destruction
    "rm -rf", "rmdir /s", "del /", "format ", "mkfs", "diskpart", "vssadmin",
    "cipher /w", "shred ", "wipe", "fdisk",
    # security / defender / firewall disablement
    "set-mppreference", "add-mppreference", "disablerealtimemonitoring",
    "netsh advfirewall", "set-netfirewall", "defender", "firewall",
    "bcdedit", "bootrec", "tamperprotection",
    # privilege / permission / ownership tampering
    "takeown", "icacls", "cacls", "attrib ", "enable-localuser",
    "add-localgroupmember", "net user ", "net localgroup",
    # persistence / scheduling / registry
    "schtasks", "reg add", "reg delete", "new-itemproperty",
    "currentversion\\run", "startup",
    # power / availability / processes at scale
    "shutdown", "powercfg", "taskkill /f", "stop-service", "sc config",
    "wmic", "bypassuac", "disable-uac", "enablelua",
]


# Regex companions to HARD_BLOCK: the same operations in any flag order or
# spelling. Applied to the lower-cased, whitespace-collapsed text.
HARD_BLOCK_RE = [re.compile(p) for p in [
    r"\brm\b(?=.*\s-{1,2}[a-z]*r)",                 # rm with any recursive flag (-r, -rf, -fr, -r -f)
    r"\brm\b(?=.*--recursive)",
    r"\b(rd|rmdir)\b.*(\s/s\b|\s-r)",               # rd /s /q, rmdir /s
    r"\b(del|erase)\b\s+\S",                        # cmd del/erase of anything
    r"\b(remove-item|ri)\b.*-(recurse|force)",      # PowerShell recursive/forced delete
    r"\bformat\b",
    r"\b(mkfs|shred|wipe|fdisk|diskpart)\b",
    r"\bdd\b.*\bof=/dev/",
    r"\b(set|add)-mppreference\b",
    r"\bnetsh\b.*\b(firewall|advfirewall)\b",
    r"\b(takeown|icacls|cacls)\b",
    r"\b(schtasks|bcdedit|bootrec|vssadmin|cipher)\b",
    r"\b(shutdown|powercfg)\b",
    r"\breg\b\s+(add|delete)\b",
]]


@dataclass
class SafetyPolicy:
    # capability switches
    allow_typing: bool = True
    allow_mouse: bool = True
    allow_open_url: bool = True
    allow_commands: bool = False          # running commands is OFF by default
    # fine-grained allow-lists
    allowed_url_schemes: Set[str] = field(default_factory=lambda: {"http", "https"})
    allowed_commands: List[str] = field(default_factory=list)  # exact argv[0] basenames
    # guards
    require_confirmation: bool = True
    max_actions_per_minute: int = 30

    # ------------------------------------------------------------------ #
    def is_hard_blocked(self, text: str) -> bool:
        """Categorical refusal. Matches the fixed substrings AND regex patterns
        that cover spelling variants of the same destructive operation
        (``rm -fr``, ``rm -r -f``, ``rm --recursive``, ``rd /s /q``, bare
        ``del``, ``Remove-Item -Recurse``…), after collapsing whitespace and
        case so shell-equivalent spellings cannot slip past."""
        low = re.sub(r"\s+", " ", text.lower()).strip()
        if any(bad in low for bad in HARD_BLOCK):
            return True
        return any(rx.search(low) for rx in HARD_BLOCK_RE)

    def command_allowed(self, argv: List[str]) -> tuple[bool, str]:
        if not self.allow_commands:
            return False, "running commands is disabled by policy (allow_commands=False)"
        if not argv:
            return False, "empty command"
        joined = " ".join(argv)
        if self.is_hard_blocked(joined):
            return False, "command matches the non-negotiable safety blocklist"
        base = re.split(r"[\\/]", argv[0])[-1].lower()
        base = base[:-4] if base.endswith(".exe") else base
        allowed = {c.lower() for c in self.allowed_commands}
        if base not in allowed:
            return False, f"'{base}' is not in the allowed-commands list"
        return True, "ok"

    def url_allowed(self, url: str) -> tuple[bool, str]:
        if not self.allow_open_url:
            return False, "opening URLs is disabled by policy"
        scheme = url.split(":", 1)[0].lower() if ":" in url else ""
        if scheme not in self.allowed_url_schemes:
            return False, f"URL scheme '{scheme}' not allowed"
        if self.is_hard_blocked(url):
            return False, "URL matches the safety blocklist"
        return True, "ok"

    def as_dict(self) -> dict:
        return {
            "allow_typing": self.allow_typing,
            "allow_mouse": self.allow_mouse,
            "allow_open_url": self.allow_open_url,
            "allow_commands": self.allow_commands,
            "allowed_commands": self.allowed_commands,
            "allowed_url_schemes": sorted(self.allowed_url_schemes),
            "require_confirmation": self.require_confirmation,
            "max_actions_per_minute": self.max_actions_per_minute,
        }


def deny_all(_action: str) -> bool:
    """The default confirmation callback: refuse everything.

    Autonomous contexts that never wire a real approver therefore perform no
    action at all. A UI supplies a callback that asks the human.
    """
    return False

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def get_starter_templates() -> List[Dict[str, Any]]:
    """
    Dynamically loads all provider templates from the templates/ directory.
    Templates stay away from the runtime plugins/ directory and are tracked by git.
    """
    templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
    templates: List[Dict[str, Any]] = []

    if templates_dir.exists():
        for f in sorted(templates_dir.glob("*.py")):
            if f.name.startswith("_") or f.name.startswith("."):
                continue
            try:
                code = f.read_text(encoding="utf-8")
                m_id = re.search(r'id=["\']([^"\']+)["\']', code)
                m_name = re.search(r'name=["\']([^"\']+)["\']', code)
                m_desc = re.search(r'description=["\']([^"\']+)["\']', code)

                tid = m_id.group(1) if m_id else f.stem
                name = m_name.group(1) if m_name else f.stem.replace("_", " ").title()
                desc = m_desc.group(1) if m_desc else f"{name} plugin template"

                templates.append({
                    "id": tid,
                    "name": name,
                    "filename": f.name,
                    "description": desc,
                    "code": code,
                })
            except Exception:
                continue

    return templates


class _TemplatesProxy(list):
    """Backwards-compatible list proxy that dynamically fetches current templates."""
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return iter(get_starter_templates())

    def __len__(self):
        return len(get_starter_templates())

    def __getitem__(self, index):
        return get_starter_templates()[index]


STARTER_TEMPLATES = _TemplatesProxy()

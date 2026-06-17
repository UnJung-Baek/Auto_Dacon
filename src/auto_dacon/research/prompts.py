from __future__ import annotations

import json
from typing import Any

from auto_dacon.research.context import compact_research_context_for_prompt


def research_messages(role: str, context: dict[str, Any], extra: str) -> list[dict[str, str]]:
    prompt_context = compact_research_context_for_prompt(context)
    compact_context = json.dumps(prompt_context, ensure_ascii=False, indent=2, default=str)
    if len(compact_context) > 50000:
        compact_context = compact_context[:50000] + "\n...<truncated>"
    system = (
        "You are an elite DACON tabular competition research node. "
        "Respect the Auto_Dacon contract: do not change Agent_K core algorithms, "
        "do not assume automatic submission, avoid leakage, and prefer experiments "
        "that can be validated with robust folds. Be concrete and evidence-driven."
    )
    user = (
        f"Role: {role}\n\n"
        f"Project context JSON:\n{compact_context}\n\n"
        f"Task:\n{extra}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

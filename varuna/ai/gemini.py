"""Gemini-powered plain-language risk brief generator.

Uses the provided API key from environment variable GEMINI_API_KEY.
The key must be set in a .env file (gitignored) or as a Colab secret.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

from varuna.engines.risk_engine import RiskDecision
from varuna.utils.logging import setup_logger

logger = setup_logger("varuna.ai.gemini")


SYSTEM_PROMPT = """You are a disaster early-warning assistant. Given structured flood-risk decisions
for multiple regions, produce a concise, plain-language briefing for emergency managers
and affected communities. Include:
- Which regions triggered and why (discharge threshold, rainfall accumulation, flood-map coverage)
- Confidence level (conformal 90% intervals)
- Recommended actions (pre-position supplies, evacuate low-lying areas, activate insurance)
- Key uncertainties (wide intervals, missing flood-map data)
Keep it under 200 words, actionable, no jargon.
"""


class GeminiRiskBrief:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            logger.warning("Gemini unavailable (no key or library). Briefs will use template fallback.")

    def generate(self, decisions: list[dict]) -> str:
        """Generate a plain-language brief from a list of RiskDecision dicts."""
        if not self.model:
            return self._template_fallback(decisions)

        payload = {
            "decisions": decisions,
            "instructions": "Produce a briefing under 200 words with sections: ALERTS, CONFIDENCE, ACTIONS, UNCERTAINTIES.",
        }
        try:
            resp = self.model.generate_content(
                SYSTEM_PROMPT + "\n\nInput JSON:\n" + json.dumps(payload, indent=2)
            )
            return resp.text.strip()
        except Exception as exc:  # pragma: no cover
            logger.error("Gemini generation failed: %s", exc)
            return self._template_fallback(decisions)

    @staticmethod
    def _template_fallback(decisions: list[dict]) -> str:
        """Deterministic template brief when Gemini is unavailable."""
        lines = ["=== VARUNA FLOOD RISK BRIEF ==="]
        triggered = [d for d in decisions if d.get("triggered")]
        if not triggered:
            lines.append("No regions triggered at 90% conformal confidence.")
        else:
            lines.append(f"ALERTS ({len(triggered)} region(s)):")
            for d in triggered:
                lines.append(
                    f"  • {d['region_id']}: {d['trigger_type']} — "
                    f"discharge {d['discharge']['mean']:.0f} m³/s "
                    f"(90% CI: {d['discharge']['interval'][0]:.0f}–{d['discharge']['interval'][1]:.0f}), "
                    f"rain {d['rainfall_accum_3d_mm']:.0f} mm, "
                    f"flood cover {d['flood_cover_fraction']:.0%}"
                )
        lines.append("CONFIDENCE: Conformal 90% intervals; confidence_ok = interval width < 50% of mean.")
        lines.append("ACTIONS: Pre-position relief, alert communities, activate parametric triggers where fired.")
        lines.append("UNCERTAINTIES: Wide intervals reduce confidence; flood-map missing where SAR unavailable.")
        return "\n".join(lines)


def load_env_file(env_path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    path = Path(env_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
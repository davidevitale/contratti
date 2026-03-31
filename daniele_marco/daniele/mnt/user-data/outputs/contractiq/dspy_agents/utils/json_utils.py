import json
import logging
import re

logger = logging.getLogger(__name__)

def safe_parse_json(raw: str) -> list | dict:
    """
    Safely parse JSON from a string, aggressively stripping markdown fences and whitespace.
    """
    if not isinstance(raw, str):
        return []
    
    raw = raw.strip()
    
    # Strip markdown fences if present
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
    if match:
        raw_to_parse = match.group(1).strip()
    else:
        raw_to_parse = raw

    try:
        return json.loads(raw_to_parse)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}. raw={raw[:200]}")
        return []


def safe_parse_int(val, default=50) -> int:
    """
    Safely parse an integer from a value, returning default on failure.
    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

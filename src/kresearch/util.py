import json
import logging
import re

logger = logging.getLogger("kresearch.util")


def safe_json_object(raw: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences.

    Returns {} (and logs a warning with the tail of the raw response) if
    parsing fails or the top-level value isn't an object. This is most often
    caused by max_tokens truncating the response mid-JSON -- surfacing it
    here means that failure mode leaves a trace instead of silently looking
    like "the model had nothing to say".
    """
    original = raw
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("failed to parse JSON from model output (likely truncated by max_tokens): %r", original[-300:])
        return {}
    if not isinstance(parsed, dict):
        logger.warning("model output was valid JSON but not an object: %r", original[:300])
        return {}
    return parsed

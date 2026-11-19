
"""Privacy-first Rius tracing. Disabled when no API key is configured."""
import os
import logging

log = logging.getLogger(__name__)


def initialize():
    """Initialize Rius only when explicitly configured."""
    from dotenv import load_dotenv
    load_dotenv()

    if (
        not os.getenv("RIUS_API_KEY")
        or os.getenv("RIUS_DISABLED", "false").lower() in ("1", "true", "yes")
    ):
        log.info("Rius observability disabled")
        return

    try:
        import rius
        rius.init(
            service_name=os.getenv("RIUS_SERVICE_NAME", "atlas-ai-fde"),
            capture_content=False
        )
        log.info("Rius observability enabled")
    except Exception:
        log.exception("Rius initialization failed")


def trace(name):
    """Skip tracing when Rius is not configured."""
    def decorate(fn):
        disabled = (
            not os.getenv("RIUS_API_KEY")
            or os.getenv("RIUS_DISABLED", "false").lower()
            in ("1", "true", "yes")
        )

        if disabled:
            return fn

        try:
            import rius
            return rius.observe(
                name=name,
                capture_input=False,
                capture_output=False
            )(fn)
        except Exception:
            log.exception("Rius tracing setup failed for %s", name)
            return fn

    return decorate

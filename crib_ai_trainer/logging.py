from logging import getLogger, basicConfig
import os

logger = getLogger(__name__)

# Basic logging setup; can be customized by scripts
if not os.environ.get("CRIB_LOGGING_CONFIGURED"):
    basicConfig(level=os.environ.get("CRIB_LOG_LEVEL", "INFO"))
    os.environ["CRIB_LOGGING_CONFIGURED"] = "1"
import sys, os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keep the persisted LLM config out of the real data/ directory: tests must
# never read or overwrite the developer's saved provider/model selection.
os.environ.setdefault(
    "TSL_LLM_CONFIG_PATH",
    str(Path(tempfile.mkdtemp(prefix="tsl_llm_cfg_")) / "llm_config.json"),
)

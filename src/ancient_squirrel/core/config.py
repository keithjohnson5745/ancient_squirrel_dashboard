from dataclasses import dataclass, field, fields
from typing import Optional
import yaml, os, logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    # ----- Input / output -------------------------------------------
    input_file: str
    output_dir: str = "output"

    # ----- Processing -----------------------------------------------
    num_workers: int = 4

    # ----- Clustering -----------------------------------------------
    num_clusters: int = 30
    clustering_method: str = "kmeans"

    # ----- NLP -------------------------------------------------------
    enable_nlp: bool = True
    use_openai: bool = False
    use_llm: bool = False
    openai_api_key: Optional[str] = None
    num_topics: int = 15

    # ----- Cluster-insight ------------------------------------------
    enable_cluster_insights: bool = True
    cluster_insight_top_n: int = 10
    cluster_insight_use_llm: bool = False
    cluster_col: str = "cluster"
    text_col: str = "clean_title"          # canonical name
    clean_text_col: Optional[str] = None   # ← NEW alias

    # ----- Extra analyses -------------------------------------------
    extract_entities: bool = True
    temporal_analysis: bool = True
    influence_analysis: bool = True

    # ---------- helpers ---------------------------------------------
    @property
    def resolved_text_col(self) -> str:
        """Prefer the alias if supplied, else fallback."""
        return self.clean_text_col or self.text_col

    # ---------- loaders / savers ------------------------------------
    @classmethod
    def from_yaml(cls, yaml_file: str) -> "AnalysisConfig":
        """Load configuration, discarding unknown keys."""
        if not os.path.exists(yaml_file):
            logger.warning("Config file not found: %s", yaml_file)
            return cls(input_file="")

        with open(yaml_file, "r") as f:
            raw = yaml.safe_load(f) or {}

        # keep only fields defined on AnalysisConfig
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in valid}
        unknown = set(raw) - valid
        if unknown:
            logger.warning("Ignoring unknown keys in %s: %s",
                           yaml_file, ", ".join(unknown))

        return cls(**clean)

    def to_yaml(self, yaml_file: str) -> None:
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        with open(yaml_file, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

# ----------------------------------------------------------------------
# Convenience helper so CLI code can stay the same
# ----------------------------------------------------------------------
from typing import Optional
from dataclasses import fields
import os, yaml
from pathlib import Path

def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
    """
    Load an AnalysisConfig from YAML or environment variables.

    The search order is:
      1. an explicit --config / path argument
      2. default locations (config/config.yaml, ~/.ancient_squirrel/config.yaml)
      3. environment variables fallback
    """
    # ---------- 1. explicit path -------------------------------------
    if config_path and os.path.exists(config_path):
        return AnalysisConfig.from_yaml(config_path)

    # ---------- 2. default file locations ----------------------------
    default_locations = [
        "config/config.yaml",
        "config/default_config.yaml",
        str(Path.home() / ".ancient_squirrel" / "config.yaml"),
    ]
    for p in default_locations:
        if os.path.exists(p):
            return AnalysisConfig.from_yaml(p)

    # ---------- 3. environment fallback ------------------------------
    env_kwargs = {
        "input_file": os.getenv("AS_INPUT_FILE", ""),
        "output_dir": os.getenv("AS_OUTPUT_DIR", "output"),
        "num_workers": int(os.getenv("AS_NUM_WORKERS", "4")),
        "use_openai": os.getenv("AS_USE_OPENAI", "false").lower() == "true",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }
    # keep only fields that really exist on the dataclass
    valid = {f.name for f in fields(AnalysisConfig)}
    env_kwargs = {k: v for k, v in env_kwargs.items() if k in valid}
    return AnalysisConfig(**env_kwargs)
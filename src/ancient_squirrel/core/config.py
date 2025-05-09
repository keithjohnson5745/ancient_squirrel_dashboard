from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict
import yaml, os, logging, re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

    # ----- JSON Format ----------------------------------------------
    extract_analysis_to_json: bool = True
    output_format: str = "json"

    # ----- Thumbnail Analysis Sampling ------------------------------
    use_sampling: bool = True
    top_communities_count: int = 15
    videos_per_community: int = 50
    min_community_size: int = 10
    stratified_sampling: bool = True

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

        # Process environment variable substitutions
        raw = _process_env_vars_in_config(raw)

        # keep only fields defined on AnalysisConfig
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in valid}
        unknown = set(raw) - valid
        if unknown:
            logger.warning("Ignoring unknown keys in %s: %s",
                           yaml_file, ", ".join(unknown))

        return cls(**clean)

    def to_yaml(self, yaml_file: str) -> None:
        """
        Save configuration to YAML, but mask any sensitive values
        
        Args:
            yaml_file: Path to save the YAML file
        """
        # Create a copy of the configuration with masked sensitive values
        config_dict = self.__dict__.copy()
        
        # Mask sensitive values like API keys
        sensitive_keys = ['openai_api_key', 'api_key', 'token', 'secret']
        for key, value in config_dict.items():
            if any(sensitive_name in key.lower() for sensitive_name in sensitive_keys) and value:
                # Mask the value with environment variable reference
                env_var = _find_env_var_for_value(value)
                if env_var:
                    config_dict[key] = f"${{{env_var}}}"
                else:
                    # Mask with asterisks if exact env var not found
                    if isinstance(value, str) and len(value) > 8:
                        config_dict[key] = value[:4] + "****" + value[-4:]
                    else:
                        config_dict[key] = "****"
        
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        with open(yaml_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# ----------------------------------------------------------------------
# Helper functions for environment variable handling
# ----------------------------------------------------------------------

def _process_env_vars_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process environment variable references in configuration values
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Processed configuration with environment variables substituted
    """
    result = {}
    
    # Regular expression to match ${ENV_VAR} and $ENV_VAR patterns
    env_var_pattern = re.compile(r'(\${([^}]+)}|\$([A-Za-z0-9_]+))')
    
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = _process_env_vars_in_config(value)
        elif isinstance(value, str):
            # Look for environment variable references
            if value.startswith("${") and value.endswith("}"):
                # Format: ${ENV_VAR}
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    result[key] = env_value
                else:
                    logger.warning(f"Environment variable {env_var} referenced in config not found")
                    result[key] = value
            elif env_var_pattern.search(value):
                # Handle inline environment variable references
                def replace_env_var(match):
                    if match.group(2):  # ${ENV_VAR} format
                        env_var = match.group(2)
                    else:  # $ENV_VAR format
                        env_var = match.group(3)
                    
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        return env_value
                    else:
                        logger.warning(f"Environment variable {env_var} referenced in config not found")
                        return match.group(0)  # Return the original reference
                
                result[key] = env_var_pattern.sub(replace_env_var, value)
            else:
                # No environment variable references
                result[key] = value
        else:
            # Non-string values
            result[key] = value
    
    return result

def _find_env_var_for_value(value: str) -> Optional[str]:
    """
    Try to find an environment variable that matches the given value
    
    Args:
        value: Value to search for in environment variables
        
    Returns:
        Name of the matching environment variable, or None if not found
    """
    for env_name, env_value in os.environ.items():
        if env_value == value:
            return env_name
    return None

# ----------------------------------------------------------------------
# Convenience helper so CLI code can stay the same
# ----------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
    """
    Load an AnalysisConfig from YAML or environment variables.

    The search order is:
      1. an explicit --config / path argument
      2. default locations (config/config.yaml, ~/.ancient_squirrel/config.yaml)
      3. environment variables fallback
      
    Environment variable references in the config files are automatically
    substituted with their actual values.
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
# test_openai_key.py
import os
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIKeyTest")

def load_config(config_path=None):
    """Load config from default locations or specified path"""
    # Try specified path first
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from specified path: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Try default locations
    default_locations = [
        "config/config.yaml",
        "config/default_config.yaml",
        str(Path.home() / ".ancient_squirrel" / "config.yaml"),
    ]
    
    for path in default_locations:
        if os.path.exists(path):
            logger.info(f"Loading config from default path: {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    
    logger.warning("No config file found")
    return {}

def test_openai_connection():
    """Test OpenAI API key by making a simple API call"""
    config = load_config()
    
    # Get API key from config
    api_key = config.get("openai_api_key")
    
    if not api_key:
        logger.error("No OpenAI API key found in config")
        return False
    
    # Check if key is masked (for privacy in logs)
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***masked***"
    logger.info(f"Found API key: {masked_key}")
    
    # Try a simple API call
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        logger.info("Successfully connected to OpenAI API")
        logger.info(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()
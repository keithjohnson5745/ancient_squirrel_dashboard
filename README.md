# Ancient Squirrel Dashboard

A comprehensive tool for analyzing YouTube video networks, extracting themes, and visualizing insights.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ancient_squirrel_dashboard.git
cd ancient_squirrel_dashboard

# Basic analysis
ancient-analyze --input data/youtube_videos.csv --output results

# Analysis with NLP features
ancient-analyze --input data/youtube_videos.csv --nlp --topics 20

# Analysis with OpenAI LLM features (requires API key)
ancient-analyze --input data/youtube_videos.csv --openai --llm

# Analysis with NLP features AND OpenAI LLM features (requires API key)
ancient-analyze --input data/lube_video_nodes.csv --nlp --openai --llm --topics 20
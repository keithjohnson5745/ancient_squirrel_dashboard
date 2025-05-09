import argparse
import logging
import os
from datetime import datetime

from ..core.config import load_config, AnalysisConfig
from ..analysis.youtube_processor import YouTubeDataProcessor

def main():
    """Main entry point for analysis CLI"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YouTube Video Analysis")
    parser.add_argument("--input", type=str, help="Input CSV file with video data")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--clusters", type=int, help="Number of clusters to extract")
    parser.add_argument("--nlp", action="store_true", help="Perform enhanced NLP analysis")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI for embeddings and insights")
    parser.add_argument("--llm", action="store_true", help="Use LLM for advanced insights")
    parser.add_argument("--topics", type=int, help="Number of NLP topics to extract")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--community-topics", action="store_true", 
                   help="Extract topics for each community separately")
    parser.add_argument("--community-topics-count", type=int, default=8,
                    help="Number of topics to extract per community")
    parser.add_argument("--no-community-topics", action="store_true",
                    help="Skip community topic extraction")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("ancient-analyze")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.input:
        config.input_file = args.input
    if args.output:
        config.output_dir = args.output
    if args.clusters:
        config.num_clusters = args.clusters
    if args.nlp:
        config.enable_nlp = True
    if args.openai:
        config.use_openai = True
    if args.llm:
        config.use_llm = True
    if args.topics:
        config.num_topics = args.topics
    if args.community_topics:
        config.community_topics_enabled = True
    if args.no_community_topics:
        config.community_topics_enabled = False
    if args.community_topics_count:
        config.community_topics_count = args.community_topics_count
    
    # Validate input file
    if not config.input_file:
        logger.error("No input file specified. Use --input or config file.")
        return 1
    
    if not os.path.exists(config.input_file):
        logger.error(f"Input file not found: {config.input_file}")
        return 1
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default output directory if not specified
    if not config.output_dir:
        config.output_dir = os.path.join("output", f"analysis_{timestamp}")
    
    # Create processor and run analysis
    processor = YouTubeDataProcessor(config)
    
    try:
        logger.info(f"Starting analysis of {config.input_file}")
        logger.info(f"Output will be saved to {config.output_dir}")
        
        # Run the processing pipeline
        df, results = processor.process()
        
        logger.info(f"Analysis complete. Results saved to {config.output_dir}")
        logger.info(f"Processed {results['metadata']['video_count']} videos")
        logger.info(f"Identified {results['metadata']['cluster_count']} clusters")
        
        if config.enable_nlp and 'nlp_analysis' in results:
            logger.info("NLP analysis successfully completed")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
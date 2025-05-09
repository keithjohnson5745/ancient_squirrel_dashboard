import re
import string
from typing import Set, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def preprocess_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
    """
    Clean and normalize text for analysis
    
    Args:
        text: Input text
        stop_words: Set of stop words to remove
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    try:
        # Import here to avoid dependency issues
        from nltk.tokenize import wordpunct_tokenize
        
        # Default stop words if none provided
        if stop_words is None:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                custom_stops = {'video', 'youtube', 'watch', 'subscribe', 'channel', 'like',
                              'comment', 'share', 'official', 'ft', 'featuring', 'presents'}
                stop_words.update(custom_stops)
            except:
                # Fallback to basic stop words if NLTK data not available
                logger.warning("NLTK stopwords not available, using basic stopwords")
                stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                            'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'video', 'youtube'}
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # Tokenize
        tokens = wordpunct_tokenize(text)
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        # Join tokens back
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        # Fall back to basic preprocessing
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        words = text.split()
        words = [w for w in words if len(w) > 2]
        return " ".join(words)

def extract_bigrams(text: str, min_freq: int = 2) -> List[str]:
    """
    Extract common bigrams from text
    
    Args:
        text: Input text
        min_freq: Minimum frequency for bigrams
        
    Returns:
        List of bigram strings
    """
    try:
        from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize
        words = [word.lower() for word in text.split() 
                if word.isalpha() and word.lower() not in stop_words and len(word) > 2]
        
        # Find bigrams
        finder = BigramCollocationFinder.from_words(words)
        finder.apply_freq_filter(min_freq)
        
        # Get top bigrams
        bigrams = [" ".join(bigram) for bigram, _ in 
                  finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)]
        
        return bigrams
    except Exception as e:
        logger.error(f"Error extracting bigrams: {str(e)}")
        return []

def analyze_linguistic_patterns(text: str) -> Dict[str, Any]:
    """
    Analyze linguistic patterns in text
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with linguistic pattern analysis
    """
    if not isinstance(text, str) or not text.strip():
        return {"is_question": False, "is_imperative": False, "patterns": []}
    
    result = {
        "is_question": False,
        "is_imperative": False,
        "patterns": []
    }
    
    # Check for questions
    if '?' in text:
        result["is_question"] = True
    elif text.lower().startswith(('what', 'why', 'how', 'when', 'where', 'who', 'which')):
        result["is_question"] = True
    
    # Check for imperatives (simplified approach)
    first_word = text.split()[0].lower() if text else ""
    if first_word in ['make', 'do', 'try', 'go', 'get', 'take', 'use', 'find', 'create',
                     'start', 'stop', 'avoid', 'let', 'keep', 'build', 'learn']:
        result["is_imperative"] = True
    
    # Check for common patterns
    patterns = []
    if re.search(r'\d+', text):
        patterns.append("number_in_title")
    if re.search(r'\b[A-Z]{2,}\b', text):
        patterns.append("all_caps_word")
    if '!' in text:
        patterns.append("exclamation")
    if re.search(r'\.{3}', text):
        patterns.append("ellipsis")
    if re.search(r'#\w+', text):
        patterns.append("hashtag")
    if re.search(r'\[.*?\]|\(.*?\)', text):
        patterns.append("brackets")
        
    result["patterns"] = patterns
    
    return result
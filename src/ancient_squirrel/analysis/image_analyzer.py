# src/ancient_squirrel/analysis/image_analyzer.py

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import base64
from io import BytesIO

from ..core.base_processor import BaseProcessor

class ImageAnalyzer(BaseProcessor):
    """Analyze YouTube thumbnail images to extract visual features"""
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the image analyzer
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        self.config = config
        
        # Initialize LLM adapter if needed for vision analysis
        self.use_llm = config.get("use_llm", False)
        self.llm_adapter = None
        
        if self.use_llm:
            try:
                from ..utils.llm_adapter import LLMAdapter
                
                openai_key = config.get("openai_api_key")
                model = config.get("llm_model", "gpt-4.1")
                
                if openai_key:
                    self.llm_adapter = LLMAdapter(
                        provider="openai",
                        api_key=openai_key,
                        model=model
                    )
                else:
                    self.logger.warning("OpenAI API key not provided, LLM image analysis disabled")
            except ImportError:
                self.logger.warning("LLMAdapter not available, disabling LLM image analysis")
        
        # Set up cache directory for analysis results
        self.cache_dir = config.get("image_analysis_cache_dir", "image_analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # Add this utility function to ImageAnalyzer class in src/ancient_squirrel/analysis/image_analyzer.py

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types to Python native types to ensure JSON serializability
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_json_serializable(obj.tolist())
        else:
            return obj    
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image to base64 for API requests
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            return ""
    
    def _get_cache_path(self, image_path: str) -> str:
        """
        Get cache path for analysis results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the cache file
        """
        import hashlib
        # Create a hash of the image path to use as the cache filename
        hash_obj = hashlib.md5(image_path.encode())
        return os.path.join(self.cache_dir, f"{hash_obj.hexdigest()}.json")
    
    def _load_cached_analysis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load cached analysis results if available
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Cached analysis results or None if not available
        """
        cache_path = self._get_cache_path(image_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached analysis: {e}")
        
        return None
    
    def _save_analysis_to_cache(self, image_path: str, analysis: Dict[str, Any]) -> None:
        """
        Save analysis results to cache
        
        Args:
            image_path: Path to the image file
            analysis: Analysis results to cache
        """
        cache_path = self._get_cache_path(image_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving analysis to cache: {e}")
    
    def analyze_image_basic(self, image_path: str) -> Dict[str, Any]:
        """
        Perform basic image analysis (colors, composition, etc.)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with basic image analysis results
        """
        try:
            from PIL import Image
            import colorsys
            
            # Open the image
            with Image.open(image_path) as img:
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get basic image properties
                width, height = img.size
                aspect_ratio = width / height
                
                # Resize for faster processing if needed
                if width > 300:
                    new_width = 300
                    new_height = int(height * (new_width / width))
                    img_resized = img.resize((new_width, new_height))
                else:
                    img_resized = img
                
                # Extract color information
                colors = img_resized.getcolors(maxcolors=100000)
                
                # Sort colors by frequency (descending)
                if colors:
                    colors = sorted(colors, key=lambda x: x[0], reverse=True)
                    
                    # Format as (r,g,b) and frequency
                    formatted_colors = []
                    for count, color in colors[:5]:  # Top 5 colors
                        r, g, b = color
                        
                        # Convert RGB to HSV for better color naming
                        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                        
                        # Determine color name (simplified)
                        color_name = self._get_color_name(h, s, v)
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        
                        # Calculate percentage
                        pixel_count = width * height
                        percentage = (count / pixel_count) * 100
                        
                        formatted_colors.append({
                            "name": color_name,
                            "hex": hex_color,
                            "rgb": [r, g, b],
                            "percentage": round(percentage, 2)
                        })
                else:
                    formatted_colors = []
                
                # Determine image brightness
                brightness = self._calculate_brightness(img_resized)
                
                # Determine if image has text (simplified approach)
                has_text = self._detect_text_presence(img_resized)
                
                # Calculate contrast
                contrast = self._calculate_contrast(img_resized)
                
                # Detect faces (simplified)
                has_faces = self._detect_faces(img_resized)
                
                return {
                    "dimensions": {
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio
                    },
                    "colors": {
                        "dominant": formatted_colors,
                        "brightness": brightness,
                        "contrast": contrast
                    },
                    "composition": {
                        "has_text": has_text,
                        "has_faces": has_faces
                    }
                }
        
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "error": str(e),
                "path": image_path
            }
    
    def _get_color_name(self, h: float, s: float, v: float) -> str:
        """
        Get color name from HSV values
        
        Args:
            h: Hue (0-1)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            Color name
        """
        # Convert hue to degrees (0-360)
        h = h * 360
        
        # Check for grayscale first
        if s < 0.1:
            if v < 0.2:
                return "black"
            elif v < 0.5:
                return "dark gray"
            elif v < 0.8:
                return "gray"
            else:
                return "white"
        
        # Determine color based on hue
        if h < 30:
            return "red"
        elif h < 60:
            return "orange"
        elif h < 90:
            return "yellow"
        elif h < 150:
            return "green"
        elif h < 210:
            return "cyan"
        elif h < 270:
            return "blue"
        elif h < 330:
            return "purple"
        else:
            return "red"
    
    def _calculate_brightness(self, img) -> float:
        """
        Calculate image brightness
        
        Args:
            img: PIL Image object
            
        Returns:
            Brightness value (0-1)
        """
        # Convert to grayscale
        gray = img.convert('L')
        
        # Calculate average pixel value
        histogram = gray.histogram()
        total_pixels = sum(histogram)
        brightness = sum(i * count for i, count in enumerate(histogram)) / total_pixels
        
        # Normalize to 0-1
        return brightness / 255
    
    def _calculate_contrast(self, img) -> float:
        """
        Calculate image contrast
        
        Args:
            img: PIL Image object
            
        Returns:
            Contrast value (0-1)
        """
        # Convert to grayscale
        gray = img.convert('L')
        
        # Calculate standard deviation of pixel values
        pixels = list(gray.getdata())
        std_dev = np.std(pixels)
        
        # Normalize to 0-1 (assuming max std_dev around 100)
        return min(1.0, std_dev / 100)
    
    def _detect_text_presence(self, img) -> bool:
        """
        Simple detection of text presence in image
        
        Args:
            img: PIL Image object
            
        Returns:
            True if text is likely present, False otherwise
        """
        try:
            # This is a simplified approach and may not be accurate
            # For better results, use OCR or a dedicated text detection model
            from PIL import ImageFilter
            
            # Convert to grayscale
            gray = img.convert('L')
            
            # Apply edge detection
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Count edge pixels
            edge_pixels = np.array(edges)
            edge_count = np.sum(edge_pixels > 50)  # Threshold
            
            # Calculate edge density
            total_pixels = edge_pixels.size
            edge_density = edge_count / total_pixels
            
            # Text typically has high edge density
            return edge_density > 0.05
            
        except Exception as e:
            self.logger.warning(f"Error detecting text: {e}")
            return False
    
    def _detect_faces(self, img) -> bool:
        """
        Simplified face detection
        
        Args:
            img: PIL Image object
            
        Returns:
            True if faces are likely present, False otherwise
        """
        # This is just a placeholder implementation
        # For real face detection, use a face detection library like face_recognition or OpenCV
        # Here we're just using a very basic heuristic based on skin tone detection
        
        # Convert to RGB array
        img_array = np.array(img)
        
        # Check if there are skin-tone-like pixels
        # This is a very simplified and not very accurate approach
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (abs(r - g) > 15)
        
        # Calculate percentage of skin-tone pixels
        skin_percentage = np.sum(skin_mask) / skin_mask.size
        
        # If more than 5% of the image is skin-tone, likely has a face
        return skin_percentage > 0.05
    
    def analyze_image_with_llm(self, image_path: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze image using LLM vision capabilities
        
        Args:
            image_path: Path to the image file
            title: Optional video title for context
            
        Returns:
            Dictionary with LLM analysis results
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Create cache path for this analysis
        cache_path = self._get_cache_path(image_path)
        
        # Check for cached results
        cached = self._load_cached_analysis(image_path)
        if cached:
            return cached
        
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image_path)
            if not image_base64:
                return {"error": "Failed to encode image"}
            
            # Create prompt for image analysis
            prompt = """
            Analyze this YouTube video thumbnail image in detail and provide the following information:

            1. VISUAL ELEMENTS:
               - Dominant colors (name specific colors and approximate percentages)
               - Composition type (close-up, medium shot, wide shot)
               - Human presence (number of people, expressions, demographics if apparent)
               - Main objects/props featured
               - Background environment

            2. TEXT ELEMENTS:
               - All text visible in the thumbnail (exact transcription)
               - Font characteristics (style, emphasis, color)
               - Text placement within frame
               - Use of emotional/trigger words

            3. THEMATIC ELEMENTS:
               - Overall emotional tone
               - Visual metaphors or symbols
               - Clickbait techniques employed
               - Brand elements present

            Format your response as a structured JSON object with these categories.
            """
            
            if title:
                prompt += f"\n\nThe title of this video is: \"{title}\"\n"
                prompt += """
                4. THUMBNAIL-TITLE RELATIONSHIP:
                   - How the thumbnail visually reinforces the title
                   - Any contrasts or mismatches between thumbnail and title
                """
            
            # Create message for vision model
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing YouTube thumbnails. Provide detailed, accurate observations in a structured JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Generate completion
            response = self.llm_adapter.client.chat.completions.create(
                model=self.llm_adapter.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            # Parse response content
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Save to cache
            self._save_analysis_to_cache(image_path, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing image with LLM: {e}")
            return {"error": str(e)}
    
    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            """
            Process data and analyze images
            
            Args:
                df: Input DataFrame with video data and thumbnail paths
                **kwargs: Additional options
                
            Returns:
                Tuple of (enhanced DataFrame, analysis results dict)
            """
            # Get options from kwargs or config
            thumbnail_col = kwargs.get("thumbnail_col", self.config.get("thumbnail_col", "thumbnail_path"))
            title_col = kwargs.get("title_col", self.config.get("title_col", "title"))
            use_llm = kwargs.get("use_llm", self.config.get("use_llm", False))
            batch_size = kwargs.get("batch_size", self.config.get("batch_size", 50))
            
            # Validate input DataFrame
            if thumbnail_col not in df.columns:
                raise ValueError(f"DataFrame must contain a '{thumbnail_col}' column")
            
            # Create a copy of the input DataFrame
            result_df = df.copy()
            
            # Create output column names
            basic_analysis_col = "thumbnail_analysis"
            llm_analysis_col = "thumbnail_llm_analysis"
            
            # Initialize analysis columns
            result_df[basic_analysis_col] = pd.NA
            if use_llm:
                result_df[llm_analysis_col] = pd.NA
            
            # Filter out records without thumbnail paths
            valid_df = result_df[~result_df[thumbnail_col].isna()]
            
            self.logger.info(f"Analyzing {len(valid_df)} thumbnails")
            
            # Process thumbnails in batches
            analysis_results = {
                "total": len(valid_df),
                "successful": 0,
                "failed": 0,
                "basic_analysis_col": basic_analysis_col
            }
            
            if use_llm:
                analysis_results["llm_analysis_col"] = llm_analysis_col
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in range(0, len(valid_df), batch_size):
                    batch_df = valid_df.iloc[i:i + batch_size]
                    self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_df)-1)//batch_size + 1} ({len(batch_df)} thumbnails)")
                    
                    # Create list of futures for basic analysis
                    basic_futures = {}
                    for idx, row in batch_df.iterrows():
                        thumbnail_path = row[thumbnail_col]
                        
                        # Skip if path is invalid
                        if not thumbnail_path or not os.path.exists(thumbnail_path):
                            continue
                        
                        # Submit analysis task
                        future = executor.submit(self.analyze_image_basic, thumbnail_path)
                        basic_futures[future] = idx
                    
                    # Process basic analysis results
                    for future in tqdm(as_completed(basic_futures), total=len(basic_futures), desc="Basic analysis"):
                        idx = basic_futures[future]
                        
                        try:
                            result = future.result()
                            
                            if "error" not in result:
                                # Convert NumPy types to Python native types for JSON serialization
                                serializable_result = self._numpy_to_python(result)
                                
                                # Update DataFrame with analysis
                                result_df.at[idx, basic_analysis_col] = json.dumps(serializable_result)
                                analysis_results["successful"] += 1
                            else:
                                analysis_results["failed"] += 1
                        except Exception as e:
                            self.logger.error(f"Error processing basic analysis result: {str(e)}")
                            import traceback
                            self.logger.debug(f"Traceback: {traceback.format_exc()}")
                            analysis_results["failed"] += 1
                    
                    # Process LLM analysis if enabled
                    if use_llm and self.llm_adapter:
                        llm_futures = {}
                        for idx, row in batch_df.iterrows():
                            thumbnail_path = row[thumbnail_col]
                            
                            # Skip if path is invalid
                            if not thumbnail_path or not os.path.exists(thumbnail_path):
                                continue
                            
                            # Get title if available
                            title = row[title_col] if title_col in row and not pd.isna(row[title_col]) else None
                            
                            # Submit analysis task
                            future = executor.submit(self.analyze_image_with_llm, thumbnail_path, title)
                            llm_futures[future] = idx
                        
                        # Process LLM analysis results
                        for future in tqdm(as_completed(llm_futures), total=len(llm_futures), desc="LLM analysis"):
                            idx = llm_futures[future]
                            
                            try:
                                result = future.result()
                                
                                # Convert NumPy types if present (though less likely in LLM results)
                                serializable_result = self._numpy_to_python(result)
                                
                                # Update DataFrame with LLM analysis
                                result_df.at[idx, llm_analysis_col] = json.dumps(serializable_result)
                                
                            except Exception as e:
                                self.logger.error(f"Error processing LLM analysis result: {str(e)}")
                                import traceback
                                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            self.logger.info(f"Thumbnail analysis complete: {analysis_results['successful']} successful, "
                        f"{analysis_results['failed']} failed")
            
            # Extract summary statistics
            try:
                summary_stats = self._extract_analysis_summary(result_df, basic_analysis_col, llm_analysis_col)
                analysis_results["summary"] = summary_stats
            except Exception as e:
                self.logger.error(f"Error extracting analysis summary: {str(e)}")
                analysis_results["summary"] = {"error": str(e)}
            
            return result_df, {
                "thumbnail_analysis": analysis_results
            }

    def _numpy_to_python(self, obj):
        """
        Convert NumPy types to Python native types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_python(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):  # Only use np.bool_ instead of np.bool
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._numpy_to_python(obj.tolist())
        elif obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        else:
            # For any other types, convert to string (last resort)
            try:
                return str(obj)
            except:
                return None
    
    def _extract_analysis_summary(self, df: pd.DataFrame, basic_col: str, llm_col: Optional[str]) -> Dict[str, Any]:
        """
        Extract summary statistics from thumbnail analysis
        
        Args:
            df: DataFrame with analysis results
            basic_col: Column containing basic analysis
            llm_col: Column containing LLM analysis (optional)
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "color_distribution": {},
            "has_text_percentage": 0,
            "has_faces_percentage": 0,
            "composition_types": {}
        }
        
        # Process basic analysis
        try:
            # Filter to rows with valid analysis
            valid_df = df[~df[basic_col].isna()]
            
            if len(valid_df) == 0:
                return summary
            
            # Parse JSON strings
            parsed_analyses = []
            for analysis_json in valid_df[basic_col]:
                try:
                    analysis = json.loads(analysis_json)
                    parsed_analyses.append(analysis)
                except:
                    continue
            
            if not parsed_analyses:
                return summary
            
            # Color distribution
            color_counts = {}
            for analysis in parsed_analyses:
                if "colors" in analysis and "dominant" in analysis["colors"]:
                    for color_info in analysis["colors"]["dominant"]:
                        color_name = color_info.get("name", "unknown")
                        if color_name in color_counts:
                            color_counts[color_name] += 1
                        else:
                            color_counts[color_name] = 1
            
            # Calculate percentages
            total = len(parsed_analyses)
            color_distribution = {color: (count / total) * 100 for color, count in color_counts.items()}
            summary["color_distribution"] = color_distribution
            
            # Text and face percentages
            text_count = 0
            face_count = 0
            
            for analysis in parsed_analyses:
                if "composition" in analysis:
                    if analysis["composition"].get("has_text", False):
                        text_count += 1
                    if analysis["composition"].get("has_faces", False):
                        face_count += 1
            
            summary["has_text_percentage"] = (text_count / total) * 100 if total > 0 else 0
            summary["has_faces_percentage"] = (face_count / total) * 100 if total > 0 else 0
            
            # Process LLM analysis if available
            if llm_col and llm_col in df.columns:
                valid_llm_df = df[~df[llm_col].isna()]
                
                if len(valid_llm_df) > 0:
                    # Parse JSON strings
                    parsed_llm_analyses = []
                    for analysis_json in valid_llm_df[llm_col]:
                        try:
                            analysis = json.loads(analysis_json)
                            parsed_llm_analyses.append(analysis)
                        except:
                            continue
                    
                    if parsed_llm_analyses:
                        # Extract composition types
                        composition_types = {}
                        for analysis in parsed_llm_analyses:
                            if "VISUAL_ELEMENTS" in analysis:
                                comp_type = analysis["VISUAL_ELEMENTS"].get("composition_type", "unknown")
                                composition_types[comp_type] = composition_types.get(comp_type, 0) + 1
                        
                        # Calculate percentages
                        llm_total = len(parsed_llm_analyses)
                        summary["composition_types"] = {
                            comp: (count / llm_total) * 100 
                            for comp, count in composition_types.items()
                        }
                        
                        # Add other LLM-specific summaries here
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error extracting analysis summary: {e}")
            return summary
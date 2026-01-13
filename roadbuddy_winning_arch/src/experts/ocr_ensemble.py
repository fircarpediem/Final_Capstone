"""
OCR Ensemble Module
Combines VietOCR, PaddleOCR, and EasyOCR for robust text extraction
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from loguru import logger

# Import OCR libraries with error handling
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    VIETOCR_AVAILABLE = True
except ImportError:
    logger.warning("VietOCR not available")
    VIETOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    logger.warning("PaddleOCR not available")
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger.warning("EasyOCR not available")
    EASYOCR_AVAILABLE = False


class OCREnsemble:
    """
    OCR Ensemble for Vietnamese text on traffic signs
    
    Uses 3 models:
    1. VietOCR - Best for Vietnamese
    2. PaddleOCR - Fast and accurate
    3. EasyOCR - Good fallback
    
    Results are combined using voting or confidence weighting
    """
    
    def __init__(self, config):
        self.config = config
        self.ensemble_method = config.model.experts.ocr.ensemble_method
        self.preprocessing_config = config.model.experts.ocr.preprocessing
        
        # Initialize OCR models
        self.models = {}
        self._init_models()
        
        logger.info(f"OCR Ensemble initialized with {len(self.models)} models")
        logger.info(f"Ensemble method: {self.ensemble_method}")
    
    def _init_models(self):
        """Initialize available OCR models"""
        
        # VietOCR
        if VIETOCR_AVAILABLE:
            try:
                config = Cfg.load_config_from_name('vgg_transformer')
                config['weights'] = 'https://github.com/pbcquoc/vietocr/releases/download/v0.1/transformerocr.pth'
                config['device'] = 'cuda' if self.config.project.device == 'cuda' else 'cpu'
                config['predictor']['beamsearch'] = False
                
                self.models['vietocr'] = Predictor(config)
                logger.info("✓ VietOCR loaded")
            except Exception as e:
                logger.warning(f"Failed to load VietOCR: {e}")
        
        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                use_gpu = self.config.project.device == 'cuda'
                self.models['paddleocr'] = PaddleOCR(
                    lang='vi',
                    use_gpu=use_gpu,
                    show_log=False
                )
                logger.info("✓ PaddleOCR loaded")
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR: {e}")
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                gpu = self.config.project.device == 'cuda'
                self.models['easyocr'] = easyocr.Reader(['vi', 'en'], gpu=gpu)
                logger.info("✓ EasyOCR loaded")
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR: {e}")
        
        if not self.models:
            raise RuntimeError("No OCR models available. Please install at least one OCR library.")
    
    def extract_text(self, detections: List[Dict]) -> List[Dict]:
        """
        Extract text from detected signs
        
        Args:
            detections: List of detected objects with 'crop' field
            
        Returns:
            Same detections with added 'text' field
        """
        for det in detections:
            # Only OCR on signs
            if "sign" not in det["class_name"].lower():
                det["text"] = None
                det["text_confidence"] = 0.0
                continue
            
            crop = det["crop"]
            
            # Check if crop is valid
            if crop.size == 0:
                det["text"] = ""
                det["text_confidence"] = 0.0
                continue
            
            # Preprocess
            if self.preprocessing_config.get("denoise", True):
                crop = self._preprocess(crop)
            
            # Run ensemble
            results = self._run_ensemble(crop)
            
            # Combine results
            if results:
                final_text, confidence = self._combine_results(results)
                det["text"] = final_text
                det["text_confidence"] = confidence
            else:
                det["text"] = ""
                det["text_confidence"] = 0.0
        
        return detections
    
    def _run_ensemble(self, image: np.ndarray) -> List[Dict]:
        """
        Run all OCR models on image
        
        Returns:
            List of {model: str, text: str, confidence: float}
        """
        results = []
        
        # VietOCR
        if 'vietocr' in self.models:
            try:
                text = self.models['vietocr'].predict(image)
                results.append({
                    'model': 'vietocr',
                    'text': text,
                    'confidence': 0.9  # VietOCR doesn't provide confidence
                })
            except Exception as e:
                logger.debug(f"VietOCR failed: {e}")
        
        # PaddleOCR
        if 'paddleocr' in self.models:
            try:
                result = self.models['paddleocr'].ocr(image, cls=True)
                if result and result[0]:
                    texts = []
                    confidences = []
                    for line in result[0]:
                        texts.append(line[1][0])
                        confidences.append(line[1][1])
                    
                    text = " ".join(texts)
                    conf = np.mean(confidences) if confidences else 0.0
                    
                    results.append({
                        'model': 'paddleocr',
                        'text': text,
                        'confidence': conf
                    })
            except Exception as e:
                logger.debug(f"PaddleOCR failed: {e}")
        
        # EasyOCR
        if 'easyocr' in self.models:
            try:
                result = self.models['easyocr'].readtext(image)
                if result:
                    texts = []
                    confidences = []
                    for (bbox, text, conf) in result:
                        texts.append(text)
                        confidences.append(conf)
                    
                    text = " ".join(texts)
                    conf = np.mean(confidences) if confidences else 0.0
                    
                    results.append({
                        'model': 'easyocr',
                        'text': text,
                        'confidence': conf
                    })
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")
        
        return results
    
    def _combine_results(self, results: List[Dict]) -> tuple:
        """
        Combine OCR results using configured method
        
        Returns:
            (final_text, confidence)
        """
        if not results:
            return "", 0.0
        
        if self.ensemble_method == "voting":
            # Majority voting
            from collections import Counter
            texts = [r['text'] for r in results]
            counter = Counter(texts)
            most_common = counter.most_common(1)[0]
            
            # Get confidence from model that predicted most common text
            for r in results:
                if r['text'] == most_common[0]:
                    return most_common[0], r['confidence']
            
            return most_common[0], 0.5
        
        elif self.ensemble_method == "confidence":
            # Use result with highest confidence
            best = max(results, key=lambda r: r['confidence'])
            return best['text'], best['confidence']
        
        elif self.ensemble_method == "weighted":
            # Weighted by model confidence
            weights = {'vietocr': 0.5, 'paddleocr': 0.3, 'easyocr': 0.2}
            
            weighted_scores = {}
            for r in results:
                text = r['text']
                weight = weights.get(r['model'], 0.33)
                score = r['confidence'] * weight
                
                if text in weighted_scores:
                    weighted_scores[text] += score
                else:
                    weighted_scores[text] = score
            
            best_text = max(weighted_scores, key=weighted_scores.get)
            best_conf = weighted_scores[best_text]
            
            return best_text, best_conf
        
        else:
            # Default: first result
            return results[0]['text'], results[0]['confidence']
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR
        
        Steps:
        1. Grayscale conversion
        2. Denoising
        3. Adaptive thresholding
        4. Sharpening (optional)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        if self.preprocessing_config.get("denoise", True):
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding
        if self.preprocessing_config.get("adaptive_threshold", True):
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Sharpen
        if self.preprocessing_config.get("sharpen", False):
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            thresh = cv2.filter2D(thresh, -1, kernel)
        
        return thresh

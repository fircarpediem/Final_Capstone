"""
Winning RoadBuddy Pipeline
Main inference pipeline combining all expert modules
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    AutoModel,
    AutoTokenizer
)

from ..data.video_processor import VideoProcessor
from ..experts.detector import TrafficDetector
from ..experts.ocr_ensemble import OCREnsemble
from ..experts.knowledge_base import TrafficKnowledgeBase


class WinningRoadBuddyPipeline:
    """
    SOTA Pipeline for RoadBuddy Challenge
    
    Architecture:
    1. Dual-stream video processing (global + high-res keyframes)
    2. Expert modules (detection, OCR, knowledge base)
    3. Late fusion with symbolic knowledge injection
    4. Qwen2-VL inference with native resolution support
    5. Post-processing and validation
    """
    
    def __init__(self, config):
        self.config = config
        
        logger.info("="*70)
        logger.info("Initializing Winning RoadBuddy Pipeline")
        logger.info("="*70)
        
        # Stage 1: Video Processor
        logger.info("\n[1/5] Initializing Video Processor...")
        self.video_processor = VideoProcessor(config)
        
        # Stage 2: Expert Modules
        logger.info("\n[2/5] Initializing Expert Modules...")
        self.detector = TrafficDetector(config)
        self.ocr = OCREnsemble(config)
        self.knowledge_base = TrafficKnowledgeBase(config)
        
        # Stage 3: Core VLM(s)
        logger.info("\n[3/5] Loading Vision-Language Model(s)...")
        self.mode = config.model.vlm.mode
        self.primary_model = config.model.vlm.primary_model
        
        self.models = {}
        self.processors = {}
        
        if self.mode == "ensemble":
            logger.info("  Mode: Ensemble (both models)")
            self._load_qwen2vl()
            self._load_internvl2()
        elif self.primary_model == "qwen2vl":
            logger.info("  Mode: Single (Qwen2-VL)")
            self._load_qwen2vl()
        elif self.primary_model == "internvl2":
            logger.info("  Mode: Single (InternVL2)")
            self._load_internvl2()
        else:
            raise ValueError(f"Unknown primary_model: {self.primary_model}")
        
        logger.info("\n[4/5] Pipeline ready!")
        logger.info("="*70)
    
    def _load_qwen2vl(self):
        """Load Qwen2-VL model"""
        model_config = self.config.model.vlm.qwen2vl
        model_name = model_config.name
        dtype = model_config.dtype
        
        # Determine torch dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Load model
        logger.info(f"  Loading Qwen2-VL: {model_name}")
        
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if model_config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif model_config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.models['qwen2vl'] = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Load processor
        self.processors['qwen2vl'] = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f"    ✓ Qwen2-VL loaded successfully")
    
    def _load_internvl2(self):
        """Load InternVL2 model"""
        model_config = self.config.model.vlm.internvl2
        model_name = model_config.name
        dtype = model_config.dtype
        
        # Determine torch dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Load model
        logger.info(f"  Loading InternVL2: {model_name}")
        
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if model_config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif model_config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.models['internvl2'] = AutoModel.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Load tokenizer for InternVL2
        self.processors['internvl2'] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f"    ✓ InternVL2 loaded successfully")
    
    def __call__(
        self,
        video_path: str,
        question: str,
        choices: Dict[str, str]
    ) -> str:
        """
        Main inference pipeline
        
        Args:
            video_path: Path to video file
            question: Question text
            choices: Dict of {choice_key: choice_value}
            
        Returns:
            Predicted answer (A, B, C, or D)
        """
        logger.info("\n" + "="*70)
        logger.info("WINNING PIPELINE - INFERENCE")
        logger.info("="*70)
        logger.info(f"Video: {Path(video_path).name}")
        logger.info(f"Question: {question}")
        
        # Stage 1: Dual-stream video processing
        logger.info("\n[Stage 1] Dual-stream video processing...")
        global_frames, keyframes_highres = self.video_processor.process(video_path)
        logger.info(f"  ✓ Global: {len(global_frames)} frames")
        logger.info(f"  ✓ Keyframes: {len(keyframes_highres)} frames")
        
        # Stage 2: Expert modules
        logger.info("\n[Stage 2] Running expert modules...")
        
        # 2A: Object detection
        logger.info("  [2A] Object detection...")
        detections = self.detector.detect(keyframes_highres)
        logger.info(f"    ✓ Detected {len(detections)} unique objects")
        
        # 2B: OCR ensemble
        logger.info("  [2B] OCR ensemble...")
        detections = self.ocr.extract_text(detections)
        ocr_count = sum(1 for d in detections if d.get('text'))
        logger.info(f"    ✓ Extracted text from {ocr_count} signs")
        
        # 2C: Knowledge base retrieval
        logger.info("  [2C] Knowledge base retrieval...")
        relevant_laws = self.knowledge_base.retrieve_for_detections(detections, question)
        logger.info(f"    ✓ Retrieved {len(relevant_laws)} relevant laws")
        
        # Stage 3: Prompt construction
        logger.info("\n[Stage 3] Building rich prompt with late fusion...")
        prompt = self._build_prompt(detections, relevant_laws, question, choices)
        logger.info(f"  ✓ Prompt length: {len(prompt)} chars")
        
        # Stage 4: Qwen2-VL inference
        logger.info("\n[Stage 4] Vision-Language Model Inference...")
        
        if self.mode == "ensemble":
            # Ensemble mode: use both models
            generated_text = self._generate_ensemble(global_frames, keyframes_highres, prompt)
        else:
            # Single model mode
            generated_text = self._generate_single(
                global_frames, 
                keyframes_highres, 
                prompt,
                self.primary_model
            )
        
        logger.info(f"  ✓ Generated: {generated_text}")
        
        # Stage 5: Post-processing
        logger.info("\n[Stage 5] Post-processing & validation...")
        final_answer = self._post_process(generated_text, detections, relevant_laws, choices)
        logger.info(f"  ✓ Final answer: {final_answer}")
        
        logger.info("\n" + "="*70)
        logger.info(f"RESULT: {final_answer}")
        logger.info("="*70 + "\n")
        
        return final_answer
    
    def _build_prompt(
        self,
        detections: List[Dict],
        knowledge: List[str],
        question: str,
        choices: Dict[str, str]
    ) -> str:
        """
        Build rich prompt with late fusion of symbolic knowledge
        
        This is CRITICAL for performance - injects expert knowledge
        """
        prompt = f"""<|im_start|>system
Bạn là chuyên gia luật giao thông đường bộ Việt Nam. Hãy phân tích video dashcam và trả lời câu hỏi dựa trên:
1. Nội dung video
2. Các đối tượng được phát hiện bởi hệ thống
3. Văn bản đọc được từ biển báo (OCR)
4. Luật giao thông Việt Nam liên quan

Hãy trả lời CHÍNH XÁC và NGẮN GỌN.
<|im_end|>

<|im_start|>user
<video>VIDEO_PLACEHOLDER</video>

📋 THÔNG TIN PHÂN TÍCH:

"""
        
        # Add detections
        if detections:
            prompt += "🚦 Đối tượng phát hiện được:\n"
            for i, det in enumerate(detections, 1):
                prompt += f"{i}. {det['class_name']}"
                
                if det.get('text'):
                    prompt += f" - Nội dung: '{det['text']}'"
                
                prompt += f" (độ tin cậy: {det['confidence']:.2f})\n"
            
            # OCR summary
            ocr_texts = [d.get('text', '') for d in detections if d.get('text')]
            if ocr_texts:
                prompt += f"\n📝 Văn bản đọc được từ biển báo: {', '.join(ocr_texts)}\n"
        else:
            prompt += "🚦 Không phát hiện được đối tượng nào.\n"
        
        # Add knowledge base
        if knowledge:
            prompt += "\n📚 Luật giao thông liên quan:\n"
            for i, law in enumerate(knowledge, 1):
                # Truncate long laws
                law_short = law if len(law) <= 200 else law[:197] + "..."
                prompt += f"{i}. {law_short}\n"
        
        # Add question and choices
        prompt += f"\n❓ CÂU HỎI: {question}\n\n"
        prompt += "Các lựa chọn:\n"
        for key, value in choices.items():
            prompt += f"{key}. {value}\n"
        
        # Instruction
        prompt += "\n💡 Dựa vào video, thông tin phát hiện và luật giao thông, hãy chọn đáp án đúng nhất.\n"
        prompt += "Chỉ trả lời bằng MỘT chữ cái: A, B, C hoặc D.\n"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def _generate_single(
        self,
        global_frames: List[np.ndarray],
        keyframes_highres: List[np.ndarray],
        prompt: str,
        model_name: str
    ) -> str:
        """
        Run inference with single VLM model
        
        Args:
            global_frames: Low-res frames for context
            keyframes_highres: High-res frames for detail
            prompt: Constructed prompt
            model_name: 'qwen2vl' or 'internvl2'
        """
        if model_name == "qwen2vl":
            return self._generate_with_qwen2vl(global_frames, keyframes_highres, prompt)
        elif model_name == "internvl2":
            return self._generate_with_internvl2(global_frames, keyframes_highres, prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _generate_with_qwen2vl(
        self,
        global_frames: List[np.ndarray],
        keyframes_highres: List[np.ndarray],
        prompt: str
    ) -> str:
        """Generate answer using Qwen2-VL"""
        model = self.models['qwen2vl']
        processor = self.processors['qwen2vl']
        
        # Combine frames
        all_frames = global_frames + keyframes_highres
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": all_frames},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process inputs
        inputs = processor(
            text=[text],
            videos=[all_frames],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generation config
        gen_config = {
            "max_new_tokens": self.config.inference.max_new_tokens,
            "do_sample": self.config.inference.do_sample,
            "temperature": self.config.inference.temperature,
            "top_p": self.config.inference.top_p,
            "num_beams": self.config.inference.num_beams,
        }
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_config)
        
        # Decode
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return generated_text
    
    def _generate_with_internvl2(
        self,
        global_frames: List[np.ndarray],
        keyframes_highres: List[np.ndarray],
        prompt: str
    ) -> str:
        """Generate answer using InternVL2"""
        model = self.models['internvl2']
        tokenizer = self.processors['internvl2']
        
        # InternVL2 uses different input format
        # Combine frames (use keyframes for higher quality)
        # InternVL2 typically processes fewer frames
        selected_frames = keyframes_highres[:3] if len(keyframes_highres) >= 3 else keyframes_highres
        
        # Convert frames to PIL Images
        from PIL import Image
        pil_images = [Image.fromarray(frame) for frame in selected_frames]
        
        # Build prompt with image placeholders
        # InternVL2 uses <image> tokens
        num_images = len(pil_images)
        image_tokens = "<image>" * num_images
        full_prompt = f"{image_tokens}\n{prompt}"
        
        # Tokenize
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt"
        ).to(model.device)
        
        # Process images
        # InternVL2 has its own vision encoder
        pixel_values = model.extract_feature(pil_images)
        if pixel_values is not None:
            pixel_values = pixel_values.to(model.device)
        
        # Generation config
        gen_config = {
            "max_new_tokens": self.config.inference.max_new_tokens,
            "do_sample": self.config.inference.do_sample,
            "temperature": self.config.inference.temperature if self.config.inference.do_sample else 1.0,
            "top_p": self.config.inference.top_p,
            "num_beams": self.config.inference.num_beams,
        }
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pixel_values=pixel_values,
                **gen_config
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def _generate_ensemble(
        self,
        global_frames: List[np.ndarray],
        keyframes_highres: List[np.ndarray],
        prompt: str
    ) -> str:
        """
        Ensemble prediction from both models
        
        Strategy:
        1. Get predictions from both models
        2. If they agree, return that answer
        3. If they disagree, use weighted voting or confidence scores
        """
        logger.info("  Running ensemble inference...")
        
        # Get predictions from both models
        logger.info("    [1/2] Qwen2-VL...")
        pred_qwen = self._generate_with_qwen2vl(global_frames, keyframes_highres, prompt)
        answer_qwen = self._extract_choice(pred_qwen)
        
        logger.info("    [2/2] InternVL2...")
        pred_intern = self._generate_with_internvl2(global_frames, keyframes_highres, prompt)
        answer_intern = self._extract_choice(pred_intern)
        
        logger.info(f"    Qwen2-VL: {answer_qwen}")
        logger.info(f"    InternVL2: {answer_intern}")
        
        # Ensemble logic
        if answer_qwen == answer_intern:
            # Agreement
            logger.info("    ✓ Models agree")
            return pred_qwen
        else:
            # Disagreement - use weighted voting
            logger.info("    ⚠ Models disagree, using weighted voting")
            
            weight_qwen = self.config.model.vlm.qwen2vl.weight
            weight_intern = self.config.model.vlm.internvl2.weight
            
            # Simple strategy: use model with higher weight
            if weight_qwen >= weight_intern:
                logger.info(f"    → Using Qwen2-VL (weight: {weight_qwen})")
                return pred_qwen
            else:
                logger.info(f"    → Using InternVL2 (weight: {weight_intern})")
                return pred_intern
    
    def postprocess_answer(self, raw_answer: str) -> str:
        """
        Simple postprocessing for compatibility
        Extracts choice from raw model output
        """
        extracted = self._extract_choice(raw_answer)
        return extracted if extracted else "A"
    
    def _post_process(
        self,
        generated_text: str,
        detections: List[Dict],
        knowledge: List[str],
        choices: Dict[str, str]
    ) -> str:
        """
        Post-processing and validation
        
        Steps:
        1. Extract choice from generated text
        2. Validate with OCR results
        3. Validate with knowledge base
        4. Apply confidence threshold
        """
        # Step 1: Extract choice
        predicted = self._extract_choice(generated_text)
        
        if not predicted:
            logger.warning("Failed to extract choice, defaulting to A")
            return "A"
        
        # Step 2: Validate with OCR (if enabled)
        if self.config.inference.post_processing.validate_with_ocr:
            ocr_validation = self._validate_with_ocr(predicted, detections, choices)
            if ocr_validation:
                logger.debug(f"OCR validation: {ocr_validation}")
        
        # Step 3: Validate with KB (if enabled)
        if self.config.inference.post_processing.validate_with_kb:
            kb_validation = self._validate_with_kb(predicted, knowledge, choices)
            if kb_validation:
                logger.debug(f"KB validation: {kb_validation}")
        
        return predicted
    
    def _extract_choice(self, text: str) -> Optional[str]:
        """Extract choice (A/B/C/D) from generated text"""
        # Try to find exact match
        text_upper = text.upper()
        
        for choice in ['A', 'B', 'C', 'D']:
            if choice in text_upper:
                # Check if it's a standalone choice (not part of a word)
                import re
                pattern = r'\b' + choice + r'\b'
                if re.search(pattern, text_upper):
                    return choice
        
        # Fallback: just check presence
        for choice in ['A', 'B', 'C', 'D']:
            if choice in text_upper:
                return choice
        
        return None
    
    def _validate_with_ocr(
        self,
        predicted: str,
        detections: List[Dict],
        choices: Dict[str, str]
    ) -> Optional[str]:
        """Validate prediction with OCR results"""
        # Check if OCR text matches any choice
        for det in detections:
            if not det.get('text'):
                continue
            
            ocr_text = det['text'].lower()
            
            for choice_key, choice_value in choices.items():
                if ocr_text in choice_value.lower():
                    if choice_key != predicted:
                        logger.warning(f"OCR mismatch: predicted {predicted}, OCR suggests {choice_key}")
                    return choice_key
        
        return None
    
    def _validate_with_kb(
        self,
        predicted: str,
        knowledge: List[str],
        choices: Dict[str, str]
    ) -> Optional[str]:
        """Validate prediction with knowledge base"""
        # Check if knowledge base strongly suggests a different answer
        # (This is a simplified version - can be enhanced)
        
        predicted_text = choices[predicted].lower()
        
        for law in knowledge:
            law_lower = law.lower()
            
            # Simple keyword matching
            if "cấm" in law_lower and "không" in predicted_text:
                # Consistent
                pass
            elif "được phép" in law_lower and "có" in predicted_text:
                # Consistent
                pass
        
        return None
    
    def batch_predict(
        self,
        video_paths: List[str],
        questions: List[str],
        choices_list: List[Dict[str, str]]
    ) -> List[str]:
        """
        Batch prediction (for evaluation)
        
        Args:
            video_paths: List of video paths
            questions: List of questions
            choices_list: List of choice dicts
            
        Returns:
            List of predicted answers
        """
        predictions = []
        
        for video_path, question, choices in zip(video_paths, questions, choices_list):
            try:
                prediction = self(video_path, question, choices)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                predictions.append("A")  # Default fallback
        
        return predictions

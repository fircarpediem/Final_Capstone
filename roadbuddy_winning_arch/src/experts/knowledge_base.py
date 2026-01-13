"""
Traffic Law Knowledge Base
RAG system for Vietnamese traffic regulations
"""

import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available")
    SBERT_AVAILABLE = False


class TrafficKnowledgeBase:
    """
    Knowledge base of Vietnamese traffic laws with semantic search
    
    Uses RAG (Retrieval-Augmented Generation) to find relevant laws
    based on detected objects and questions
    """
    
    def __init__(self, config):
        self.config = config
        self.top_k = config.model.experts.knowledge_base.top_k
        self.min_similarity = config.model.experts.knowledge_base.min_similarity
        self.law_db_path = Path(config.model.experts.knowledge_base.law_database)
        
        # Initialize encoder
        if SBERT_AVAILABLE:
            encoder_name = config.model.experts.knowledge_base.encoder
            logger.info(f"Loading knowledge base encoder: {encoder_name}")
            self.encoder = SentenceTransformer(encoder_name)
        else:
            raise ImportError("sentence-transformers required for knowledge base")
        
        # Load laws
        self.laws = self._load_laws()
        
        # Precompute embeddings
        logger.info("Computing law embeddings...")
        self.law_texts = [law['text'] for law in self.laws]
        self.law_embeddings = self.encoder.encode(
            self.law_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Knowledge base initialized with {len(self.laws)} laws")
    
    def _load_laws(self) -> List[Dict]:
        """Load traffic law database"""
        
        # If database file exists, load it
        if self.law_db_path.exists():
            with open(self.law_db_path, 'r', encoding='utf-8') as f:
                laws = json.load(f)
            logger.info(f"Loaded {len(laws)} laws from {self.law_db_path}")
            return laws
        
        # Otherwise, use default laws
        logger.warning(f"Law database not found at {self.law_db_path}, using defaults")
        return self._get_default_laws()
    
    def _get_default_laws(self) -> List[Dict]:
        """Default Vietnamese traffic laws"""
        return [
            # Prohibitory Signs (Biển cấm)
            {
                "id": "P.102",
                "type": "prohibitory",
                "name": "Cấm đi ngược chiều",
                "text": "Biển P.102: Cấm đi ngược chiều. Cấm người điều khiển phương tiện đi ngược chiều quy định trên đường một chiều. Biển đặt vuông góc với trục đường.",
                "applies_to": ["all"],
                "penalty": "400,000 - 600,000 VNĐ"
            },
            {
                "id": "P.106",
                "type": "prohibitory",
                "name": "Cấm dừng xe và đỗ xe",
                "text": "Biển P.106: Cấm dừng xe và đỗ xe. Biển báo cấm dừng xe và đỗ xe ở phía có biển báo. Áp dụng cho tất cả các loại xe.",
                "applies_to": ["all"],
                "penalty": "300,000 - 400,000 VNĐ"
            },
            {
                "id": "P.123a",
                "type": "prohibitory",
                "name": "Cấm ô tô rẽ trái",
                "text": "Biển P.123a: Cấm ô tô rẽ trái. Cấm ô tô rẽ trái theo hướng biển chỉ dẫn. Áp dụng cho ô tô các loại.",
                "applies_to": ["car", "truck", "bus"],
                "penalty": "800,000 - 1,000,000 VNĐ"
            },
            {
                "id": "P.123b",
                "type": "prohibitory",
                "name": "Cấm ô tô rẽ phải",
                "text": "Biển P.123b: Cấm ô tô rẽ phải. Cấm ô tô rẽ phải theo hướng biển chỉ dẫn. Áp dụng cho ô tô các loại.",
                "applies_to": ["car", "truck", "bus"],
                "penalty": "800,000 - 1,000,000 VNĐ"
            },
            {
                "id": "P.128",
                "type": "prohibitory",
                "name": "Cấm xe ô tô tải",
                "text": "Biển P.128: Cấm xe ô tô tải. Cấm các loại xe tải, kể cả xe tải nhỏ đi qua. Thường áp dụng trong khu dân cư hoặc đường hẹp.",
                "applies_to": ["truck"],
                "penalty": "800,000 - 1,000,000 VNĐ"
            },
            {
                "id": "P.130",
                "type": "prohibitory",
                "name": "Cấm xe ô tô khách",
                "text": "Biển P.130: Cấm xe ô tô khách. Cấm xe ô tô chở khách từ 12 chỗ ngồi trở lên đi qua.",
                "applies_to": ["bus"],
                "penalty": "800,000 - 1,000,000 VNĐ"
            },
            
            # Warning Signs (Biển cảnh báo)
            {
                "id": "W.201",
                "type": "warning",
                "name": "Chỗ ngoặt nguy hiểm",
                "text": "Biển W.201: Chỗ ngoặt nguy hiểm vòng bên trái. Báo trước sắp đến đoạn đường có chỗ ngoặt nguy hiểm.",
                "applies_to": ["all"],
                "penalty": None
            },
            {
                "id": "W.224",
                "type": "warning",
                "name": "Đường người đi bộ cắt ngang",
                "text": "Biển W.224: Báo hiệu phía trước có đường dành cho người đi bộ cắt ngang qua đường. Người điều khiển phương tiện phải giảm tốc độ, chú ý nhường đường.",
                "applies_to": ["all"],
                "penalty": None
            },
            
            # Mandatory Signs (Biển hiệu lệnh)
            {
                "id": "R.301a",
                "type": "mandatory",
                "name": "Đi thẳng",
                "text": "Biển R.301a: Hiệu lệnh phải đi thẳng. Chỉ được phép đi thẳng, không được rẽ trái, rẽ phải hoặc quay đầu xe.",
                "applies_to": ["all"],
                "penalty": "800,000 - 1,000,000 VNĐ nếu vi phạm"
            },
            {
                "id": "R.301c",
                "type": "mandatory",
                "name": "Rẽ trái",
                "text": "Biển R.301c: Hiệu lệnh phải rẽ trái. Chỉ được phép rẽ trái theo hướng mũi tên.",
                "applies_to": ["all"],
                "penalty": "800,000 - 1,000,000 VNĐ nếu vi phạm"
            },
            {
                "id": "R.303",
                "type": "mandatory",
                "name": "Hướng phải đi vòng chướng ngại vật",
                "text": "Biển R.303: Hiệu lệnh hướng đi vòng chướng ngại vật. Người điều khiển phương tiện phải đi vòng chướng ngại vật theo hướng mũi tên.",
                "applies_to": ["all"],
                "penalty": None
            },
            
            # Speed Limits
            {
                "id": "SPEED_URBAN",
                "type": "regulation",
                "name": "Tốc độ tối đa trong đô thị",
                "text": "Tốc độ tối đa cho phép trong khu vực đô thị: 50 km/h đối với ô tô, 40 km/h đối với xe máy. Theo Luật Giao thông đường bộ 2024, Điều 11.",
                "applies_to": ["all"],
                "penalty": "Phạt 800,000 - 1,200,000 VNĐ nếu vượt quá 5-10 km/h"
            },
            {
                "id": "SPEED_HIGHWAY",
                "type": "regulation",
                "name": "Tốc độ tối đa trên đường cao tốc",
                "text": "Tốc độ tối đa trên đường cao tốc: 120 km/h đối với ô tô, 90 km/h đối với xe tải. Theo Nghị định 100/2019/NĐ-CP.",
                "applies_to": ["car", "truck"],
                "penalty": "Phạt 4,000,000 - 6,000,000 VNĐ nếu vượt quá 20 km/h"
            },
            
            # Traffic Lights
            {
                "id": "TRAFFIC_LIGHT_RED",
                "type": "regulation",
                "name": "Đèn đỏ",
                "text": "Đèn tín hiệu màu đỏ: Người điều khiển phương tiện phải dừng lại trước vạch dừng. Vượt đèn đỏ bị phạt rất nặng.",
                "applies_to": ["all"],
                "penalty": "Phạt 4,000,000 - 6,000,000 VNĐ đối với ô tô"
            },
            {
                "id": "TRAFFIC_LIGHT_YELLOW",
                "type": "regulation",
                "name": "Đèn vàng",
                "text": "Đèn tín hiệu màu vàng: Cảnh báo sắp chuyển sang đèn đỏ. Người điều khiển phương tiện phải giảm tốc độ và chuẩn bị dừng lại, trừ trường hợp xe đã ở quá gần không thể dừng an toàn.",
                "applies_to": ["all"],
                "penalty": None
            },
            
            # Right of Way
            {
                "id": "RIGHT_OF_WAY",
                "type": "regulation",
                "name": "Quy tắc nhường đường",
                "text": "Tại ngã tư không có đèn tín hiệu: Xe đi từ bên phải được quyền ưu tiên. Xe rẽ phải nhường đường cho xe đi thẳng và rẽ trái. Xe rẽ trái nhường đường cho xe đi thẳng.",
                "applies_to": ["all"],
                "penalty": "Phạt 400,000 - 600,000 VNĐ nếu vi phạm"
            },
        ]
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Search for relevant traffic laws
        
        Args:
            query: Search query (from detections + question)
            top_k: Number of results to return (default: self.top_k)
            
        Returns:
            List of relevant law texts
        """
        if top_k is None:
            top_k = self.top_k
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
        
        # Compute similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.law_embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by minimum similarity
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.min_similarity:
                results.append(self.law_texts[idx])
        
        logger.debug(f"Retrieved {len(results)} laws for query: '{query[:50]}...'")
        
        return results
    
    def retrieve_for_detections(self, detections: List[Dict], question: str) -> List[str]:
        """
        Build query from detections and retrieve relevant laws
        
        Args:
            detections: List of detected objects
            question: User question
            
        Returns:
            List of relevant law texts
        """
        # Build query
        query_parts = [question]
        
        for det in detections:
            # Add class name
            query_parts.append(det["class_name"])
            
            # Add OCR text if available
            if det.get("text"):
                query_parts.append(det["text"])
        
        query = " ".join(query_parts)
        
        # Search
        return self.search(query)
    
    def save_database(self, output_path: Optional[str] = None):
        """Save current laws to JSON file"""
        if output_path is None:
            output_path = self.law_db_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.laws, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.laws)} laws to {output_path}")

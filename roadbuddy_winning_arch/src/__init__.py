"""
RoadBuddy Winning Architecture
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "RoadBuddy Team"

from .models.winning_pipeline import WinningRoadBuddyPipeline
from .experts.detector import TrafficDetector
from .experts.ocr_ensemble import OCREnsemble
from .experts.knowledge_base import TrafficKnowledgeBase

__all__ = [
    "WinningRoadBuddyPipeline",
    "TrafficDetector",
    "OCREnsemble",
    "TrafficKnowledgeBase",
]

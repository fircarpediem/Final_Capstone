"""
Demo notebook for RoadBuddy Winning Architecture
Shows example usage and visualizations
"""

# Cell 1: Setup
print("Installing dependencies...")
# !pip install -q -r ../requirements.txt

# Cell 2: Imports
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from omegaconf import OmegaConf
from src.models.winning_pipeline import WinningRoadBuddyPipeline
from src.data.video_processor import VideoProcessor
from src.experts.detector import TrafficDetector
import cv2
import matplotlib.pyplot as plt
from IPython.display import Video, display

print("✓ Imports successful")

# Cell 3: Load Configuration
config_path = "../configs/config.yaml"
config = OmegaConf.load(config_path)
print("✓ Config loaded")
print(OmegaConf.to_yaml(config))

# Cell 4: Initialize Pipeline
print("Initializing pipeline...")
pipeline = WinningRoadBuddyPipeline(config)
print("✓ Pipeline ready!")

# Cell 5: Example Inference
video_path = "../data/raw/public_test/videos/sample_video.mp4"
question = "Xe có được rẽ trái không?"
choices = {
    "A": "Có",
    "B": "Không",
    "C": "Tùy thời gian",
    "D": "Không rõ"
}

# Display video
print("Video:")
display(Video(video_path, width=640))

# Run inference
print("\nRunning inference...")
answer = pipeline(video_path, question, choices)

print(f"\n{'='*50}")
print(f"Question: {question}")
print(f"Answer: {answer} - {choices[answer]}")
print(f"{'='*50}")

# Cell 6: Visualize Detections
print("Visualizing detections...")

# Process video
video_processor = VideoProcessor(config)
global_frames, keyframes = video_processor.process(video_path)

# Run detection on keyframes
detector = TrafficDetector(config)
detections = detector.detect(keyframes)

# Visualize first keyframe with detections
if keyframes and detections:
    annotated = detector.visualize_detections(keyframes[0], detections)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated)
    plt.axis('off')
    plt.title(f"Detected Objects: {len(detections)}")
    plt.show()
    
    # Print detection details
    print("\nDetected objects:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class_name']}")
        if det.get('text'):
            print(f"   Text: '{det['text']}'")
        print(f"   Confidence: {det['confidence']:.2f}")

# Cell 7: Knowledge Base Demo
from src.experts.knowledge_base import TrafficKnowledgeBase

kb = TrafficKnowledgeBase(config)

query = "cấm rẽ trái ô tô"
results = kb.search(query, top_k=3)

print(f"Query: '{query}'")
print(f"\nTop 3 relevant laws:")
for i, law in enumerate(results, 1):
    print(f"\n{i}. {law}")

# Cell 8: Batch Inference Example
import json

# Load test set
test_json = "../data/raw/public_test/public_test.json"
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Run on first 5 samples
print("Running batch inference on 5 samples...")
for item in test_data[:5]:
    answer = pipeline(
        item['video_path'],
        item['question'],
        item['choices']
    )
    print(f"Q: {item['question'][:50]}...")
    print(f"A: {answer}")
    print()

print("✓ Demo complete!")

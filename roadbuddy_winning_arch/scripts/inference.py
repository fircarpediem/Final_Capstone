"""
Inference Script for RoadBuddy
Run predictions on test set
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.winning_pipeline import WinningRoadBuddyPipeline


def load_test_data(json_path: str):
    """Load test data from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_inference(config_path: str, test_json: str, output_path: str):
    """
    Run inference on test set
    
    Args:
        config_path: Path to config.yaml
        test_json: Path to test JSON file
        output_path: Path to save predictions
    """
    # Load config
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = WinningRoadBuddyPipeline(config)
    
    # Load test data
    logger.info(f"Loading test data from {test_json}")
    test_data = load_test_data(test_json)
    
    logger.info(f"Total questions: {len(test_data)}")
    
    # Run inference
    predictions = []
    
    for item in tqdm(test_data, desc="Inference"):
        question_id = item['id']
        video_path = item['video_path']
        question = item['question']
        choices = item['choices']
        
        try:
            # Predict
            answer = pipeline(video_path, question, choices)
            
            predictions.append({
                'id': question_id,
                'answer': answer
            })
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            predictions.append({
                'id': question_id,
                'answer': 'A'  # Default
            })
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Total predictions: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser(description="RoadBuddy Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.json",
        help="Path to save predictions"
    )
    
    args = parser.parse_args()
    
    run_inference(args.config, args.test_json, args.output)


if __name__ == "__main__":
    main()

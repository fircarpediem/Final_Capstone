"""
Evaluation Script
Evaluate predictions against ground truth
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from loguru import logger


def load_json(path: str):
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate(predictions_path: str, ground_truth_path: str):
    """
    Evaluate predictions
    
    Args:
        predictions_path: Path to predictions JSON
        ground_truth_path: Path to ground truth JSON
    """
    # Load data
    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)
    
    # Build prediction dict
    pred_dict = {item['id']: item['answer'] for item in predictions}
    
    # Build ground truth dict
    gt_dict = {}
    for item in ground_truth:
        # Find correct answer
        correct_choice = None
        for key, value in item['choices'].items():
            if value == item['answer']:  # Exact match
                correct_choice = key
                break
        
        if not correct_choice:
            # Fallback: check if answer is a choice key
            if item['answer'] in item['choices']:
                correct_choice = item['answer']
        
        gt_dict[item['id']] = correct_choice
    
    # Compute metrics
    total = len(gt_dict)
    correct = 0
    
    # Per question type
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for qid, gt_answer in gt_dict.items():
        if qid not in pred_dict:
            logger.warning(f"Missing prediction for question {qid}")
            continue
        
        pred_answer = pred_dict[qid]
        
        if pred_answer == gt_answer:
            correct += 1
        
        # TODO: Track question types if available
    
    # Overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    logger.info("="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total questions: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info("="*50)
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct
    }


def main():
    parser = argparse.ArgumentParser(description="RoadBuddy Evaluation")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth JSON"
    )
    
    args = parser.parse_args()
    
    evaluate(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()

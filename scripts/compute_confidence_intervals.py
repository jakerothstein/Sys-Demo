import json
import numpy as np
from typing import List, Dict, Any

def bootstrap_metric(data: List[float], n_resamples=1000) -> float:
    if not data:
        return 0.0
    resampled_means = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=len(data), replace=True)
        resampled_means.append(np.mean(sample))
    # 95% CI is the 2.5th and 97.5th percentiles
    low = np.percentile(resampled_means, 2.5)
    high = np.percentile(resampled_means, 97.5)
    mean = np.mean(data)
    # Return the largest distance to the mean as the "plus-minus"
    return max(mean - low, high - mean)

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def bootstrap_f1(y_true, y_pred, n_resamples=1000):
    resampled_f1s = []
    for _ in range(n_resamples):
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        sample_true = [y_true[i] for i in indices]
        sample_pred = [y_pred[i] for i in indices]
        
        tp = sum(1 for t, p in zip(sample_true, sample_pred) if t and p)
        fp = sum(1 for t, p in zip(sample_true, sample_pred) if not t and p)
        fn = sum(1 for t, p in zip(sample_true, sample_pred) if t and not p)
        
        resampled_f1s.append(calculate_f1(tp, fp, fn))
    
    low = np.percentile(resampled_f1s, 2.5)
    high = np.percentile(resampled_f1s, 97.5)
    mean = calculate_f1(sum(y_true), sum(1 for t, p in zip(y_true, y_pred) if not t and p), sum(1 for t, p in zip(y_true, y_pred) if t and not p)) # This is not quite right but we'll use the mean of f1s or real f1
    
    # Real F1
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    real_f1 = calculate_f1(tp, fp, fn)
    
    return max(real_f1 - low, high - real_f1)

def main():
    # 1. Ambiguity Benchmark (n=67)
    amb_path = "data/ambiguity_benchmark_results.jsonl"
    amb_data = []
    with open(amb_path, 'r') as f:
        for line in f:
            amb_data.append(json.loads(line))
            
    y_true = [q['expected_hitl'] for q in amb_data]
    y_pred = [q['predicted_hitl'] for q in amb_data]
    
    # Accuracy
    acc_data = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]
    acc_ci = bootstrap_metric(acc_data)
    
    # Recall
    recall_data = [1.0 if p else 0.0 for t, p in zip(y_true, y_pred) if t]
    recall_ci = bootstrap_metric(recall_data)
    
    # F1
    f1_ci = bootstrap_f1(y_true, y_pred)
    
    # 2. Spider (n=50)
    spider_path = "data/spider/spider_50_openai_postfix_results.jsonl"
    spider_data = []
    with open(spider_path, 'r') as f:
        for line in f:
            spider_data.append(json.loads(line))
            
    # Spider Exec Accuracy (on comparable subset)
    spider_comp = [1.0 if r['exec_equivalent_to_gold'] else 0.0 for r in spider_data if r.get('exec_equivalent_to_gold') is not None]
    spider_acc_ci = bootstrap_metric(spider_comp)
    
    # Spider Clear-pass (Specificity)
    spider_clear = [0.0 if r['hitl_triggered'] else 1.0 for r in spider_data]
    spider_clear_ci = bootstrap_metric(spider_clear)
    
    print("Ambiguity (n=67):")
    print(f"  Accuracy:   {np.mean(acc_data):.3f} \u00b1 {acc_ci:.3f}")
    print(f"  Recall:     {np.mean(recall_data):.3f} \u00b1 {recall_ci:.3f}")
    
    # Calculate real F1 for printing
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    real_f1 = calculate_f1(tp, fp, fn)
    print(f"  F1:         {real_f1:.3f} \u00b1 {f1_ci:.3f}")
    
    # Specificity (Clear-pass) for Ambiguity
    spec_data = [1.0 if not p else 0.0 for t, p in zip(y_true, y_pred) if not t]
    spec_ci = bootstrap_metric(spec_data)
    print(f"  Spec:       {np.mean(spec_data):.3f} \u00b1 {spec_ci:.3f}")
    
    print("\nSpider (n=50):")
    print(f"  Exec Acc:   {np.mean(spider_comp):.3f} \u00b1 {spider_acc_ci:.3f}")
    print(f"  Clear-pass: {np.mean(spider_clear):.3f} \u00b1 {spider_clear_ci:.3f}")

if __name__ == "__main__":
    np.random.seed(42)
    main()

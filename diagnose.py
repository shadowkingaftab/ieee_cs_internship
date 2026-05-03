import pandas as pd
import os
from classifier import classify_prompt
from router import get_route, get_route_explanation, get_route_color
from logger import log_result, load_logs
from evaluator import get_metrics_text

def test_pipeline():
    print("Testing pipeline components...")
    prompt = "What is edge AI?"
    
    # 1. Classify
    print("  Classifying...")
    result = classify_prompt(prompt)
    print(f"  Result: {result}")
    
    # 2. Route
    print("  Routing...")
    route = get_route(result["intent"], result["confidence"])
    explanation = get_route_explanation(result["intent"], result["confidence"], route)
    color = get_route_color(route)
    print(f"  Route: {route}, Color: {color}")
    
    # 3. Log
    print("  Logging...")
    log_result(result, route)
    
    # 4. Load Logs
    print("  Loading logs...")
    df = load_logs()
    print(f"  Log DF shape: {df.shape}")
    
    # 5. Metrics
    print("  Getting metrics...")
    metrics = get_metrics_text()
    print("  Metrics retrieved.")
    
    print("\nSUCCESS: All components worked without TypeError.")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()

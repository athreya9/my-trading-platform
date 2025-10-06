import json
import logging

logger = logging.getLogger(__name__)

def optimize_strategy(trades):
    """
    Optimizes strategy parameters based on backtest results.
    For now, it's a simple threshold optimizer.
    """
    logger.info("Starting strategy optimization...")
    
    if not trades:
        logger.warning("No trades provided for optimization. Returning None.")
        return None, None

    best_threshold = None
    best_pnl = -float("inf")
    
    # Assuming trades have a 'confidence' key
    # Iterate through possible confidence thresholds
    for threshold in range(0, 101, 5): # From 0% to 100% confidence in steps of 5
        filtered_trades = [t for t in trades if t.get("confidence", 0) >= threshold]
        
        if not filtered_trades:
            continue
            
        total_pnl = sum(t["pnl"] for t in filtered_trades)
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
            
    logger.info(f"Optimization completed. Best confidence threshold: {best_threshold} with PnL: {best_pnl:.2f}")
    return best_threshold, best_pnl

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Example usage with dummy trades
    dummy_trades = [
        {"confidence": 70, "pnl": 100},
        {"confidence": 80, "pnl": 150},
        {"confidence": 60, "pnl": -50},
        {"confidence": 90, "pnl": 200},
        {"confidence": 75, "pnl": 80},
        {"confidence": 85, "pnl": -20},
    ]
    
    best_thresh, best_pnl = optimize_strategy(dummy_trades)
    if best_thresh is not None:
        print(f"Best confidence threshold: {best_thresh}% for a total PnL of {best_pnl:.2f}")
    else:
        print("No optimal strategy found.")
#!/usr/bin/env python3
"""
Update bot status to show it's running with live signals
"""
import json
import os
from datetime import datetime

def update_status():
    """Update bot status for frontend"""
    
    # Update bot status
    status_data = [{
        "status": "running",
        "reason": "Automated multi-instrument trading active",
        "timestamp": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }]
    
    os.makedirs('data', exist_ok=True)
    with open('data/bot_status.json', 'w') as f:
        json.dump(status_data, f, indent=2)
    
    # Update bot control
    control_data = [{
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "changed_by": "automated_system",
        "reason": "Multi-instrument trading active"
    }]
    
    with open('data/bot_control.json', 'w') as f:
        json.dump(control_data, f, indent=2)
    
    print("✅ Bot status updated to RUNNING")
    print("📊 Multi-instrument trading active")

if __name__ == "__main__":
    update_status()
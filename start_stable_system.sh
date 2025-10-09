#!/bin/bash
# Stable system startup script

echo "🚀 Starting STABLE Trading System..."

# Kill any existing processes
pkill -f stable_system.py
pkill -f system_monitor.py

# Wait a moment
sleep 2

# Start system monitor (which will start the main system)
nohup python3 system_monitor.py > monitor.log 2>&1 &

echo "✅ Stable system started with monitoring"
echo "📊 Check logs: tail -f monitor.log"
echo "🔍 Check status: ps aux | grep stable_system"
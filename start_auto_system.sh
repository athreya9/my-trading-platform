#!/bin/bash
# Auto start script for market system

echo "🤖 Starting AUTO MARKET SYSTEM..."

# Kill any existing processes
pkill -f auto_market_system.py
pkill -f stable_system.py

# Wait a moment
sleep 2

# Start auto market system
nohup python3 auto_market_system.py > auto_system.log 2>&1 &

echo "✅ AUTO MARKET SYSTEM STARTED"
echo "📊 System will auto-start/stop with Indian market hours"
echo "🕘 Market: 9:15 AM - 3:30 PM IST (Mon-Fri)"
echo "📱 Check Telegram for status updates"
echo "📋 Logs: tail -f auto_system.log"
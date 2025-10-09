#!/bin/bash
# Setup automated cron jobs for trading system

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create cron job for daily AI retraining at 4 PM IST
CRON_JOB="0 16 * * 1-5 cd $PROJECT_DIR && /usr/bin/python3 automation/cron_retrain.py >> logs/retrain.log 2>&1"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Automated cron job setup complete"
echo "📅 AI retraining will run daily at 4 PM IST (weekdays only)"
echo "📝 Logs: $PROJECT_DIR/logs/retrain.log"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Make scripts executable
chmod +x "$PROJECT_DIR/automation/cron_retrain.py"
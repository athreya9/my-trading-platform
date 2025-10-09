#!/bin/bash
# Complete automation setup - NO MANUAL COMMANDS NEEDED

echo "🤖 Setting up FULLY AUTOMATED trading system..."

# Create systemd service for auto-start
sudo tee /etc/systemd/system/ai-trading.service > /dev/null <<EOF
[Unit]
Description=AI Trading Platform
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/automated_accurate_system.py
Restart=always
RestartSec=10
Environment=PATH=/usr/bin:/usr/local/bin

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading.service
sudo systemctl start ai-trading.service

# Create cron job for system restart at 9:00 AM daily
(crontab -l 2>/dev/null; echo "0 9 * * 1-5 sudo systemctl restart ai-trading.service") | crontab -

# Create auto-recovery script
cat > auto_recovery.sh << 'EOF'
#!/bin/bash
# Auto-recovery script - runs every 5 minutes
if ! pgrep -f "automated_accurate_system.py" > /dev/null; then
    echo "$(date): System down, restarting..." >> recovery.log
    sudo systemctl restart ai-trading.service
fi
EOF

chmod +x auto_recovery.sh

# Add recovery cron job
(crontab -l 2>/dev/null; echo "*/5 * * * * $(pwd)/auto_recovery.sh") | crontab -

echo "✅ FULLY AUTOMATED SETUP COMPLETE!"
echo "🚀 System will:"
echo "   - Start automatically on boot"
echo "   - Restart daily at 9:00 AM"
echo "   - Auto-recover if crashed"
echo "   - Run without any manual intervention"
echo ""
echo "📊 Check status: sudo systemctl status ai-trading.service"
echo "📝 View logs: sudo journalctl -u ai-trading.service -f"
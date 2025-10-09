#!/usr/bin/env python3
"""
System Monitor - Ensures stable_system.py is always running
"""
import os
import time
import subprocess
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '8250334547:AAHFIXLgvwlJlUUasiXY-5wHJ85E2AeC39k')
        self.admin_id = "1375236879"
        self.process = None
        self.restart_count = 0
    
    def notify(self, message, target="admin"):
        """Send monitoring notification"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            if target == "admin":
                requests.post(url, json={'chat_id': self.admin_id, 'text': message}, timeout=10)
            elif target == "channel":
                requests.post(url, json={'chat_id': '@DATradingSignals', 'text': message}, timeout=10)
        except:
            pass
    
    def is_system_running(self):
        """Check if stable_system.py is running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'stable_system.py'], 
                                  capture_output=True, text=True)
            return bool(result.stdout.strip())
        except:
            return False
    
    def start_system(self):
        """Start the stable system"""
        try:
            self.process = subprocess.Popen([
                'python3', 'stable_system.py'
            ], cwd=os.getcwd())
            
            self.restart_count += 1
            logger.info(f"✅ System started (restart #{self.restart_count})")
            
            if self.restart_count > 1:
                self.notify(f"🔄 System restarted (#{self.restart_count})\n⏰ {datetime.now().strftime('%H:%M:%S')}", target="admin")
            
            return True
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def monitor(self):
        """Main monitoring loop"""
        logger.info("🔍 System monitor started")
        
        while True:
            try:
                if not self.is_system_running():
                    logger.warning("⚠️ System not running - restarting...")
                    self.start_system()
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Monitor stopped")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor()
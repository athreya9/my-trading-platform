#!/usr/bin/env python3
"""
Self-healing system with auto-fix capabilities
"""
import os
import json
import requests
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

class SelfHealingSystem:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.admin_chat_id = "1375236879"  # Your admin chat ID
        self.issues_found = []
        self.issues_fixed = []
        self.critical_issues = []
    
    def send_admin_alert(self, message):
        """Send alert to admin"""
        try:
            bot = Bot(token=self.bot_token)
            bot.send_message(chat_id=self.admin_chat_id, text=message, parse_mode='HTML')
        except Exception as e:
            print(f"Failed to send admin alert: {e}")
    
    def fix_telegram_bot(self):
        """Auto-fix Telegram bot issues"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "Telegram bot online"
            
            # Try restarting bot process
            subprocess.run(["pkill", "-f", "simple_bot.py"], capture_output=True)
            subprocess.Popen([sys.executable, "api/simple_bot.py"], cwd=os.getcwd())
            
            # Wait and test again
            import time
            time.sleep(5)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                self.issues_fixed.append("Restarted Telegram bot")
                return True, "Bot restarted successfully"
            
            self.critical_issues.append("Telegram bot unreachable")
            return False, "Bot restart failed"
            
        except Exception as e:
            self.critical_issues.append(f"Telegram bot error: {e}")
            return False, str(e)
    
    def fix_kite_connection(self):
        """Auto-fix Kite API issues"""
        try:
            from api.kite_live_engine import KiteLiveEngine
            engine = KiteLiveEngine()
            
            # Test connection
            if engine.kite:
                try:
                    engine.kite.profile()
                    return True, "Kite connection active"
                except:
                    pass
            
            # Try to refresh token using TOTP
            try:
                from api.kite_totp_login import get_kite_token_with_totp
                fresh_token = get_kite_token_with_totp()
                
                if fresh_token:
                    # Update .env file
                    env_content = []
                    with open('.env', 'r') as f:
                        for line in f:
                            if line.startswith('KITE_ACCESS_TOKEN='):
                                env_content.append(f'KITE_ACCESS_TOKEN={fresh_token}\n')
                            else:
                                env_content.append(line)
                    
                    with open('.env', 'w') as f:
                        f.writelines(env_content)
                    
                    self.issues_fixed.append("Refreshed Kite access token")
                    return True, "Kite token refreshed"
                
            except Exception as e:
                self.critical_issues.append(f"Kite token refresh failed: {e}")
                return False, f"Token refresh failed: {e}"
            
            self.critical_issues.append("Kite API connection failed")
            return False, "Connection failed"
            
        except Exception as e:
            self.critical_issues.append(f"Kite system error: {e}")
            return False, str(e)
    
    def fix_data_files(self):
        """Auto-fix data file issues"""
        try:
            files_fixed = []
            required_files = {
                'data/signals.json': [],
                'api-data/trading-signals.json': [],
                'data/subscribers.json': {}
            }
            
            for file_path, default_content in required_files.items():
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        json.dump(default_content, f, indent=2)
                    files_fixed.append(file_path)
                else:
                    # Validate JSON
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        with open(file_path, 'w') as f:
                            json.dump(default_content, f, indent=2)
                        files_fixed.append(f"{file_path} (corrupted)")
            
            if files_fixed:
                self.issues_fixed.append(f"Fixed data files: {', '.join(files_fixed)}")
            
            return True, f"Data files ready ({len(files_fixed)} fixed)"
            
        except Exception as e:
            self.critical_issues.append(f"Data files error: {e}")
            return False, str(e)
    
    def fix_dependencies(self):
        """Auto-fix missing dependencies"""
        try:
            missing_modules = []
            required_modules = ['telegram', 'kiteconnect', 'PIL', 'pytesseract', 'schedule']
            
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                # Auto-install missing modules
                for module in missing_modules:
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", module], 
                                     check=True, capture_output=True)
                        self.issues_fixed.append(f"Installed {module}")
                    except subprocess.CalledProcessError:
                        self.critical_issues.append(f"Failed to install {module}")
                        return False, f"Failed to install {module}"
            
            return True, "All dependencies available"
            
        except Exception as e:
            self.critical_issues.append(f"Dependencies error: {e}")
            return False, str(e)
    
    def fix_process_issues(self):
        """Auto-fix process and system issues"""
        try:
            fixes_applied = []
            
            # Check disk space
            import shutil
            free_space = shutil.disk_usage('.').free / (1024**3)  # GB
            if free_space < 1:
                self.critical_issues.append(f"Low disk space: {free_space:.1f}GB")
            
            # Check if automated system is running
            result = subprocess.run(["pgrep", "-f", "automated_accurate_system.py"], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                # System not running, try to start it
                subprocess.Popen([sys.executable, "automated_accurate_system.py"], 
                               cwd=os.getcwd())
                fixes_applied.append("Started automated system")
            
            # Clean up old log files
            log_files = ['bot.log', 'system.log']
            for log_file in log_files:
                if os.path.exists(log_file) and os.path.getsize(log_file) > 10*1024*1024:  # 10MB
                    with open(log_file, 'w') as f:
                        f.write(f"Log cleared at {datetime.now()}\n")
                    fixes_applied.append(f"Cleared {log_file}")
            
            if fixes_applied:
                self.issues_fixed.extend(fixes_applied)
            
            return True, f"Process checks complete ({len(fixes_applied)} fixes)"
            
        except Exception as e:
            self.critical_issues.append(f"Process error: {e}")
            return False, str(e)
    
    def run_comprehensive_health_check(self):
        """Run all health checks with auto-fix"""
        print("🔧 SELF-HEALING SYSTEM CHECK")
        print("=" * 50)
        
        checks = [
            ("Dependencies", self.fix_dependencies),
            ("Data Files", self.fix_data_files),
            ("Telegram Bot", self.fix_telegram_bot),
            ("Kite Connection", self.fix_kite_connection),
            ("Process Health", self.fix_process_issues)
        ]
        
        all_healthy = True
        
        for name, check_func in checks:
            try:
                success, message = check_func()
                status = "✅" if success else "❌"
                print(f"{status} {name}: {message}")
                
                if not success:
                    all_healthy = False
                    self.issues_found.append(f"{name}: {message}")
                    
            except Exception as e:
                print(f"❌ {name}: Critical error - {e}")
                self.critical_issues.append(f"{name}: {e}")
                all_healthy = False
        
        # Generate admin report
        self.send_admin_health_report(all_healthy)
        
        return all_healthy
    
    def send_admin_health_report(self, system_healthy):
        """Send comprehensive health report to admin"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if system_healthy and not self.issues_fixed:
            message = f"✅ <b>SYSTEM HEALTHY</b>\n⏰ {timestamp}\n\n🚀 All 5 instruments ready\n📱 Telegram alerts active\n💻 Frontend operational"
        else:
            message = f"🔧 <b>SYSTEM HEALTH REPORT</b>\n⏰ {timestamp}\n\n"
            
            if self.issues_fixed:
                message += f"✅ <b>AUTO-FIXED ({len(self.issues_fixed)}):</b>\n"
                for fix in self.issues_fixed[:5]:  # Show max 5
                    message += f"• {fix}\n"
                message += "\n"
            
            if self.critical_issues:
                message += f"❌ <b>NEEDS ADMIN ({len(self.critical_issues)}):</b>\n"
                for issue in self.critical_issues[:3]:  # Show max 3
                    message += f"• {issue}\n"
                message += "\n"
            
            if not self.critical_issues:
                message += "🚀 <b>SYSTEM OPERATIONAL</b>\n"
            else:
                message += "⚠️ <b>ADMIN INTERVENTION REQUIRED</b>\n"
        
        self.send_admin_alert(message)

def main():
    """Main health check function"""
    healer = SelfHealingSystem()
    healer.run_comprehensive_health_check()

if __name__ == "__main__":
    main()
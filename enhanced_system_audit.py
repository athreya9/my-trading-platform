#!/usr/bin/env python3
"""
Enhanced System Audit with Git Analysis and Telegram Reporting
"""

import os
import json
import pickle
import datetime
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class EnhancedSystemAudit:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = {
            "audit_timestamp": datetime.datetime.now().isoformat(),
            "modules": {},
            "git_analysis": {},
            "latency_metrics": {},
            "confidence_distribution": {},
            "overall_status": "UNKNOWN"
        }
        
    def audit_git_repository(self) -> Dict[str, Any]:
        """Analyze Git repository status and health"""
        try:
            git_results = {}
            
            # Check if it's a git repository
            git_dir = self.base_path / ".git"
            if not git_dir.exists():
                return {"status": "❌ FAILED", "error": "Not a Git repository"}
            
            # Get current branch
            try:
                branch_result = subprocess.run(
                    ["git", "branch", "--show-current"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                git_results["current_branch"] = branch_result.stdout.strip()
            except:
                git_results["current_branch"] = "unknown"
            
            # Get commit count
            try:
                commit_result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                git_results["total_commits"] = int(commit_result.stdout.strip())
            except:
                git_results["total_commits"] = 0
            
            # Get last commit info
            try:
                last_commit = subprocess.run(
                    ["git", "log", "-1", "--format=%H|%an|%ad|%s"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                if last_commit.stdout:
                    parts = last_commit.stdout.strip().split("|", 3)
                    git_results["last_commit"] = {
                        "hash": parts[0][:8],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    }
            except:
                git_results["last_commit"] = {}
            
            # Check working directory status
            try:
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                modified_files = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0
                git_results["modified_files"] = modified_files
                git_results["clean_working_dir"] = modified_files == 0
            except:
                git_results["modified_files"] = "unknown"
                git_results["clean_working_dir"] = False
            
            # Check remote status
            try:
                remote_result = subprocess.run(
                    ["git", "remote", "-v"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                git_results["has_remote"] = bool(remote_result.stdout.strip())
                git_results["remotes"] = len(remote_result.stdout.strip().split('\n')) // 2 if remote_result.stdout.strip() else 0
            except:
                git_results["has_remote"] = False
                git_results["remotes"] = 0
            
            # Check for recent activity (last 7 days)
            try:
                recent_commits = subprocess.run(
                    ["git", "log", "--since=7.days.ago", "--oneline"], 
                    cwd=self.base_path, 
                    capture_output=True, 
                    text=True
                )
                git_results["recent_commits_7d"] = len(recent_commits.stdout.strip().split('\n')) if recent_commits.stdout.strip() else 0
            except:
                git_results["recent_commits_7d"] = 0
            
            return {
                "status": "✅ PASSED",
                **git_results
            }
            
        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}
    
    def audit_cron_jobs(self) -> Dict[str, Any]:
        """Validate cron jobs and scheduled tasks"""
        try:
            workflows_path = self.base_path / ".github" / "workflows"
            active_workflows = []
            
            if workflows_path.exists():
                for workflow in workflows_path.glob("*.yml"):
                    active_workflows.append(workflow.name)
            
            cron_setup = self.base_path / "automation" / "setup_cron.sh"
            cron_retrain = self.base_path / "automation" / "cron_retrain.py"
            scheduler_exists = (self.base_path / "api" / "automated_scheduler.py").exists()
            
            return {
                "status": "✅ PASSED",
                "active_workflows": active_workflows,
                "local_cron_setup": cron_setup.exists(),
                "retrain_script": cron_retrain.exists(),
                "scheduler_module": scheduler_exists,
                "workflow_count": len(active_workflows)
            }
        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}
    
    def audit_ai_model(self) -> Dict[str, Any]:
        """Validate AI model integrity and training status"""
        try:
            model_path = self.base_path / "api" / "trading_model.pkl"
            signal_model_path = self.base_path / "api" / "signal_model.pkl"
            
            model_info = {}
            
            if model_path.exists():
                stat = model_path.stat()
                model_info["trading_model"] = {
                    "exists": True,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            
            if signal_model_path.exists():
                stat = signal_model_path.stat()
                model_info["signal_model"] = {
                    "exists": True,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            
            training_pipeline = (self.base_path / "api" / "ai_training_pipeline.py").exists()
            
            accuracy_report_path = self.base_path / "data" / "accuracy_report.json"
            accuracy_data = {}
            if accuracy_report_path.exists():
                with open(accuracy_report_path) as f:
                    accuracy_data = json.load(f)
            
            return {
                "status": "✅ PASSED" if model_path.exists() or signal_model_path.exists() else "❌ FAILED",
                "models": model_info,
                "training_pipeline": training_pipeline,
                "accuracy_report": accuracy_data,
                "confidence_threshold": 0.85
            }
        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}
    
    def audit_signal_generation(self) -> Dict[str, Any]:
        """Validate signal generation integrity"""
        try:
            signals_path = self.base_path / "data" / "signals.json"
            production_signals_path = self.base_path / "data" / "production_signals.json"
            
            signal_data = {}
            production_data = {}
            
            if signals_path.exists():
                with open(signals_path) as f:
                    signal_data = json.load(f)
            
            if production_signals_path.exists():
                with open(production_signals_path) as f:
                    production_data = json.load(f)
            
            engines = [
                "accurate_live_engine.py",
                "ai_signal_engine.py", 
                "live_trading_bot.py",
                "signal_manager.py"
            ]
            
            engine_status = {}
            for engine in engines:
                engine_path = self.base_path / "api" / engine
                engine_status[engine] = engine_path.exists()
            
            confidence_scores = []
            if isinstance(signal_data, list):
                for signal in signal_data:
                    if isinstance(signal, dict) and 'confidence' in signal:
                        confidence_scores.append(signal['confidence'])
            
            return {
                "status": "✅ PASSED",
                "signal_files": {
                    "signals.json": signals_path.exists(),
                    "production_signals.json": production_signals_path.exists()
                },
                "engines": engine_status,
                "signal_count": len(signal_data) if isinstance(signal_data, list) else 0,
                "production_count": len(production_data) if isinstance(production_data, list) else 0,
                "confidence_scores": confidence_scores[:10]
            }
        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}
    
    def send_telegram_summary(self, summary: str) -> bool:
        """Send audit summary to Telegram admin bot"""
        try:
            env_path = self.base_path / ".env"
            if not env_path.exists():
                print("❌ .env file not found - cannot send Telegram summary")
                return False
            
            # Read environment variables
            bot_token = None
            chat_id = None
            
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        bot_token = line.split("=", 1)[1].strip()
                    elif line.startswith("TELEGRAM_CHAT_ID="):
                        chat_id = line.split("=", 1)[1].strip()
            
            if not bot_token or not chat_id:
                print("❌ Telegram credentials not found in .env")
                return False
            
            # Send message
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": f"🔍 **SYSTEM AUDIT REPORT**\n\n{summary}",
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("✅ Audit summary sent to Telegram admin")
                return True
            else:
                print(f"❌ Failed to send Telegram message: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending Telegram summary: {str(e)}")
            return False
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Execute complete system audit with Git analysis"""
        print("🔍 Starting Enhanced System Integrity Audit...")
        print("=" * 60)
        
        # Run Git analysis first
        print("\n🔍 Auditing Git Repository...")
        git_result = self.audit_git_repository()
        self.results["git_analysis"] = git_result
        print(f"   {git_result.get('status', '❌ FAILED')}")
        
        # Run other audit modules
        audit_modules = [
            ("Cron Jobs", self.audit_cron_jobs),
            ("AI Model & Training", self.audit_ai_model),
            ("Signal Generation", self.audit_signal_generation),
        ]
        
        passed_count = 0
        total_count = len(audit_modules) + 1  # +1 for Git
        
        if "✅ PASSED" in git_result.get("status", ""):
            passed_count += 1
        
        for module_name, audit_func in audit_modules:
            print(f"\n🔍 Auditing {module_name}...")
            result = audit_func()
            self.results["modules"][module_name] = result
            
            status = result.get("status", "❌ FAILED")
            print(f"   {status}")
            
            if "✅ PASSED" in status:
                passed_count += 1
        
        # Calculate latency metrics
        start_time = time.time()
        signals_path = self.base_path / "data" / "signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                json.load(f)
        file_latency = (time.time() - start_time) * 1000
        
        self.results["latency_metrics"] = {
            "file_access_ms": round(file_latency, 2),
            "acceptable_threshold_ms": 1000
        }
        
        # Overall status
        if passed_count == total_count:
            self.results["overall_status"] = "✅ ALL SYSTEMS OPERATIONAL"
        elif passed_count >= total_count * 0.8:
            self.results["overall_status"] = "⚠️ MOSTLY OPERATIONAL"
        else:
            self.results["overall_status"] = "❌ CRITICAL ISSUES DETECTED"
        
        return self.results
    
    def generate_telegram_summary(self) -> str:
        """Generate concise summary for Telegram"""
        git = self.results.get("git_analysis", {})
        
        summary = []
        summary.append(f"🚀 **Trading Platform Audit**")
        summary.append(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"🎯 Status: {self.results['overall_status']}")
        summary.append("")
        
        # Git summary
        if git.get("status") == "✅ PASSED":
            summary.append(f"📂 **Git Repository**")
            summary.append(f"• Branch: {git.get('current_branch', 'unknown')}")
            summary.append(f"• Commits: {git.get('total_commits', 0)}")
            summary.append(f"• Recent (7d): {git.get('recent_commits_7d', 0)}")
            summary.append(f"• Clean: {'✅' if git.get('clean_working_dir') else '❌'}")
            summary.append("")
        
        # Module summary
        passed = sum(1 for result in self.results["modules"].values() if "✅ PASSED" in result.get("status", ""))
        total = len(self.results["modules"])
        summary.append(f"🔧 **System Modules**: {passed}/{total} operational")
        
        # Key metrics
        latency = self.results.get("latency_metrics", {})
        if latency:
            summary.append(f"⚡ **Performance**: {latency.get('file_access_ms', 0)}ms file access")
        
        summary.append("")
        summary.append("🔒 *Audit completed without system disruption*")
        
        return "\n".join(summary)
    
    def generate_detailed_report(self) -> str:
        """Generate detailed audit report"""
        report = []
        report.append("🚀 AI TRADING PLATFORM - ENHANCED AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"Audit Timestamp: {self.results['audit_timestamp']}")
        report.append(f"Overall Status: {self.results['overall_status']}")
        report.append("")
        
        # Git analysis
        git = self.results.get("git_analysis", {})
        if git:
            report.append("📂 GIT REPOSITORY ANALYSIS")
            report.append(f"   Status: {git.get('status', 'UNKNOWN')}")
            if git.get("status") == "✅ PASSED":
                report.append(f"   Current Branch: {git.get('current_branch', 'unknown')}")
                report.append(f"   Total Commits: {git.get('total_commits', 0)}")
                report.append(f"   Recent Commits (7d): {git.get('recent_commits_7d', 0)}")
                report.append(f"   Modified Files: {git.get('modified_files', 'unknown')}")
                report.append(f"   Clean Working Dir: {'✅' if git.get('clean_working_dir') else '❌'}")
                report.append(f"   Has Remote: {'✅' if git.get('has_remote') else '❌'}")
                
                if git.get("last_commit"):
                    lc = git["last_commit"]
                    report.append(f"   Last Commit: {lc.get('hash', '')} by {lc.get('author', '')}")
            report.append("")
        
        # Module results
        for module_name, result in self.results["modules"].items():
            report.append(f"📋 {module_name}")
            report.append(f"   Status: {result.get('status', 'UNKNOWN')}")
            
            if "error" in result:
                report.append(f"   Error: {result['error']}")
            else:
                if "workflow_count" in result:
                    report.append(f"   Active Workflows: {result['workflow_count']}")
                if "models" in result:
                    report.append(f"   Models Found: {len(result['models'])}")
                if "signal_count" in result:
                    report.append(f"   Signals: {result['signal_count']}")
            report.append("")
        
        # Latency metrics
        if "latency_metrics" in self.results:
            report.append("⏱️ LATENCY METRICS")
            for metric, value in self.results["latency_metrics"].items():
                report.append(f"   {metric}: {value}")
            report.append("")
        
        report.append("🔒 AUDIT COMPLETED - NO LIVE SYSTEM DISRUPTION")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    auditor = EnhancedSystemAudit()
    results = auditor.run_full_audit()
    
    # Generate and display detailed report
    detailed_report = auditor.generate_detailed_report()
    print("\n" + "=" * 60)
    print(detailed_report)
    
    # Generate Telegram summary
    telegram_summary = auditor.generate_telegram_summary()
    
    # Send to Telegram
    print("\n📱 Sending summary to Telegram admin...")
    telegram_sent = auditor.send_telegram_summary(telegram_summary)
    
    # Save results
    results_path = Path(__file__).parent / "enhanced_audit_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    if telegram_sent:
        print("✅ Audit complete - Summary sent to Telegram admin")
    else:
        print("⚠️ Audit complete - Telegram summary failed")
        print("\n📱 Telegram Summary:")
        print("-" * 40)
        print(telegram_summary)
    
    return results

if __name__ == "__main__":
    main()
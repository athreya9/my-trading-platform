#!/usr/bin/env python3
"""
Quick setup script to configure Telegram bot for trading alerts
"""
import os
from dotenv import load_dotenv, set_key

def setup_telegram():
    print("🤖 Telegram Bot Setup for Trading Alerts")
    print("=" * 50)
    
    # Load existing .env
    load_dotenv()
    
    print("\n1. Create a Telegram Bot:")
    print("   - Open Telegram and search for @BotFather")
    print("   - Send /newbot command")
    print("   - Choose a name and username for your bot")
    print("   - Copy the bot token")
    
    bot_token = input("\nEnter your Telegram Bot Token: ").strip()
    
    print("\n2. Get your Chat ID:")
    print("   - Start a chat with your new bot")
    print("   - Send any message to the bot")
    print("   - Visit: https://api.telegram.org/bot{}/getUpdates".format(bot_token))
    print("   - Look for 'chat':{'id': YOUR_CHAT_ID}")
    
    chat_id = input("\nEnter your Telegram Chat ID: ").strip()
    
    # Save to .env file
    env_file = '.env'
    set_key(env_file, 'TELEGRAM_BOT_TOKEN', bot_token)
    set_key(env_file, 'TELEGRAM_CHAT_ID', chat_id)
    
    print(f"\n✅ Telegram credentials saved to {env_file}")
    
    # Test the setup
    print("\n🧪 Testing Telegram connection...")
    
    try:
        from api.telegram_alerts import test_telegram_alerts
        test_telegram_alerts()
        print("\n🎉 Setup completed successfully!")
        print("\nYour trading bot will now send alerts to Telegram!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check your credentials and try again.")

if __name__ == "__main__":
    setup_telegram()
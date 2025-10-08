#!/usr/bin/env python3
"""
Complete Telegram Bot with OCR + Subscription Automation
"""
import json
import datetime
import io
import time
import schedule
from PIL import Image
import pytesseract
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import os
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
UPI_ID = "datrade@ybl"
SUBSCRIPTION_AMOUNT = "₹499"
SUBSCRIBERS_FILE = "data/subscribers.json"

# === COMMAND: /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to DA Trading Signals!\n\n"
        "🚀 Get high-confidence trade alerts with clear entries, targets, and stoploss.\n"
        f"💳 Subscription: {SUBSCRIPTION_AMOUNT}/month via UPI\n"
        f"📲 UPI ID: {UPI_ID}\n\n"
        "Type /subscribe to begin."
    )

# === COMMAND: /subscribe ===
async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📥 *Subscription Details*\n\n"
        f"To receive premium trade alerts, subscribe for just {SUBSCRIPTION_AMOUNT}/month.\n\n"
        f"💳 *Payment Method*: UPI\n"
        f"📲 *UPI ID*: {UPI_ID}\n"
        "🧾 After payment, send a screenshot here to activate your access.\n\n"
        "✅ Once confirmed, you'll start receiving premium trading alerts directly in this chat.",
        parse_mode='Markdown'
    )

# === OCR Screenshot Handler ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(photo_bytes))
        text = pytesseract.image_to_string(image)

        chat_id = str(update.message.chat_id)
        today = datetime.date.today()
        new_expiry = today + datetime.timedelta(days=30)

        # Check for payment verification keywords
        if (SUBSCRIPTION_AMOUNT in text or "499" in text or 
            "UPI" in text or "successful" in text.lower() or 
            "paid" in text.lower() or UPI_ID in text):
            
            # Load existing subscribers
            os.makedirs('data', exist_ok=True)
            try:
                with open(SUBSCRIBERS_FILE) as f:
                    users = json.load(f)
            except FileNotFoundError:
                users = {}

            # Add/update subscriber
            users[chat_id] = {
                "username": update.message.from_user.username or "unknown",
                "first_name": update.message.from_user.first_name or "unknown",
                "subscribed_on": str(today),
                "expires_on": str(new_expiry),
                "status": "active"
            }

            # Save subscribers
            with open(SUBSCRIBERS_FILE, "w") as f:
                json.dump(users, f, indent=2)

            await update.message.reply_text(
                f"✅ Payment verified!\n"
                f"📅 Subscription active until {new_expiry.strftime('%d %b %Y')}\n"
                f"🎯 You'll now receive premium trading alerts!\n\n"
                f"Welcome to DA Trading Signals! 🚀"
            )
            
            print(f"✅ New subscriber: {chat_id} ({users[chat_id]['first_name']})")
            
        else:
            await update.message.reply_text(
                "⚠️ Couldn't verify payment from screenshot.\n"
                f"Please ensure it clearly shows {SUBSCRIPTION_AMOUNT} payment to {UPI_ID}"
            )
    except Exception as e:
        print(f"Error processing photo: {e}")
        await update.message.reply_text(
            "❌ Error processing screenshot. Please try again or contact support."
        )

# === Daily Expiry Check ===
def check_expiry():
    try:
        with open(SUBSCRIBERS_FILE) as f:
            users = json.load(f)
    except FileNotFoundError:
        return

    today = datetime.date.today()
    bot = Bot(token=BOT_TOKEN)
    
    expired_count = 0
    warning_count = 0

    for uid, info in users.items():
        try:
            expiry = datetime.datetime.strptime(info["expires_on"], "%Y-%m-%d").date()
            days_left = (expiry - today).days
            
            if expiry < today and info["status"] == "active":
                info["status"] = "expired"
                bot.send_message(
                    chat_id=uid, 
                    text="⏳ Your DA Trading Signals subscription has expired.\n"
                         f"Renew for {SUBSCRIPTION_AMOUNT}/month to continue receiving alerts.\n"
                         "Type /subscribe to renew."
                )
                expired_count += 1
                
            elif days_left == 3 and info["status"] == "active":
                bot.send_message(
                    chat_id=uid, 
                    text=f"⚠️ Your subscription expires in 3 days ({expiry.strftime('%d %b %Y')}).\n"
                         "Renew now to avoid interruption in trading alerts.\n"
                         "Type /subscribe to renew."
                )
                warning_count += 1
                
        except Exception as e:
            print(f"Error checking expiry for {uid}: {e}")

    # Save updated status
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    
    if expired_count > 0 or warning_count > 0:
        print(f"📊 Expiry check: {expired_count} expired, {warning_count} warnings sent")

# === Get Active Subscribers ===
def get_active_subscribers():
    """Get list of active subscriber chat IDs"""
    try:
        with open(SUBSCRIBERS_FILE) as f:
            users = json.load(f)
        
        active_subscribers = []
        today = datetime.date.today()
        
        for uid, info in users.items():
            if info["status"] == "active":
                expiry = datetime.datetime.strptime(info["expires_on"], "%Y-%m-%d").date()
                if expiry >= today:
                    active_subscribers.append(uid)
        
        return active_subscribers
    except FileNotFoundError:
        return []

# === Schedule Daily Task ===
schedule.every().day.at("09:00").do(check_expiry)

# === Main Bot Setup ===
def run_subscription_bot():
    """Run the subscription bot"""
    print("🤖 Starting DA Trading Signals Subscription Bot...")
    
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("✅ Bot handlers registered")
    print(f"💳 UPI ID: {UPI_ID}")
    print(f"💰 Subscription: {SUBSCRIPTION_AMOUNT}/month")
    print("🚀 Bot is running...")
    
    app.run_polling()

if __name__ == "__main__":
    run_subscription_bot()
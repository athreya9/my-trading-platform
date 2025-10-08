#!/usr/bin/env python3
import asyncio
import json
import os
from datetime import datetime, timedelta
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import io

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
UPI_ID = "datrade@ybl"

async def start(update, context):
    await update.message.reply_text(
        "👋 Welcome to DA Trading Signals!\n\n"
        "🚀 Get high-confidence trade alerts\n"
        "💳 Subscription: ₹499/month\n"
        f"📲 UPI ID: {UPI_ID}\n\n"
        "Type /subscribe to begin."
    )

async def subscribe(update, context):
    await update.message.reply_text(
        "📥 *Subscription Details*\n\n"
        "To receive premium trade alerts, subscribe for just ₹499/month.\n\n"
        f"💳 UPI ID: {UPI_ID}\n"
        "🧾 After payment, send screenshot here for verification.\n\n"
        "⚠️ Screenshot must clearly show:\n"
        "• Amount: ₹499\n"
        "• UPI ID: datrade@ybl\n"
        "• Payment status: Successful\n\n"
        "✅ Once verified, you'll get alerts directly here!",
        parse_mode='Markdown'
    )

async def handle_photo(update, context):
    try:
        # Download and process image
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(photo_bytes))
        text = pytesseract.image_to_string(image)
        
        # Verify payment details
        if ("499" in text and "datrade@ybl" in text and 
            ("successful" in text.lower() or "paid" in text.lower() or "completed" in text.lower())):
            
            # Save subscriber
            chat_id = str(update.message.chat_id)
            os.makedirs('data', exist_ok=True)
            
            try:
                with open('data/subscribers.json') as f:
                    users = json.load(f)
            except FileNotFoundError:
                users = {}
            
            users[chat_id] = {
                "username": update.message.from_user.username or "unknown",
                "subscribed_on": str(datetime.now().date()),
                "expires_on": str((datetime.now() + timedelta(days=30)).date()),
                "status": "active"
            }
            
            with open('data/subscribers.json', 'w') as f:
                json.dump(users, f, indent=2)
            
            await update.message.reply_text(
                "✅ Payment verified successfully!\n"
                "📅 Subscription active for 30 days\n\n"
                "🎯 You'll receive premium trading alerts directly here!\n"
                "📢 Also join our channel: https://t.me/DATradingSignals\n\n"
                "Welcome to DA Trading Signals! 🚀"
            )
        else:
            await update.message.reply_text(
                "❌ Payment verification failed!\n\n"
                "Please ensure your screenshot clearly shows:\n"
                "• Amount: ₹499\n"
                "• UPI ID: datrade@ybl\n"
                "• Status: Successful/Paid/Completed\n\n"
                "Try uploading a clearer screenshot."
            )
    except Exception as e:
        await update.message.reply_text(
            "❌ Error processing screenshot.\n"
            "Please try again with a clear payment screenshot."
        )

def main():
    print("🤖 Starting bot...")
    
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("✅ Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
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
ADMIN_CHAT_IDS = ["1375236879"]  # Add admin chat IDs here

async def start(update, context):
    chat_id = str(update.message.chat_id)
    
    # Check if admin
    if chat_id in ADMIN_CHAT_IDS:
        await update.message.reply_text(
            "👋 Welcome Admin!\n\n"
            "🔧 Admin Commands:\n"
            "/adduser <chat_id> - Add user without payment\n"
            "/listusers - View all subscribers\n"
            "/removeuser <chat_id> - Remove user\n\n"
            "Regular commands: /subscribe"
        )
    else:
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
        
        # Check what was found in OCR
        found_amount = "499" in text
        found_upi = "datrade@ybl" in text or "datrade" in text
        found_status = any(word in text.lower() for word in ["successful", "paid", "completed", "success"])
        
        print(f"OCR Results - Amount: {found_amount}, UPI: {found_upi}, Status: {found_status}")
        print(f"OCR Text: {text[:200]}...")  # First 200 chars for debugging
        
        # Verify payment details
        if found_amount and found_upi and found_status:
            
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
            # Detailed error message
            missing = []
            if not found_amount: missing.append("Amount ₹499")
            if not found_upi: missing.append("UPI ID datrade@ybl")
            if not found_status: missing.append("Success status")
            
            await update.message.reply_text(
                f"❌ Payment verification failed!\n\n"
                f"Missing: {', '.join(missing)}\n\n"
                "Please ensure your screenshot clearly shows:\n"
                "• Amount: ₹499\n"
                "• UPI ID: datrade@ybl\n"
                "• Status: Successful/Paid/Completed\n\n"
                "Try uploading a clearer screenshot."
            )
    except Exception as e:
        print(f"Screenshot processing error: {e}")
        await update.message.reply_text(
            "❌ Error processing screenshot.\n"
            "Please try again with a clear payment screenshot.\n\n"
            "Make sure the image is clear and contains payment details."
        )

async def add_user_admin(update, context):
    chat_id = str(update.message.chat_id)
    if chat_id not in ADMIN_CHAT_IDS:
        await update.message.reply_text("❌ Admin only command")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /adduser <chat_id>")
        return
    
    new_user_id = context.args[0]
    
    os.makedirs('data', exist_ok=True)
    try:
        with open('data/subscribers.json') as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    
    users[new_user_id] = {
        "username": "admin_added",
        "subscribed_on": str(datetime.now().date()),
        "expires_on": str((datetime.now() + timedelta(days=365)).date()),  # 1 year
        "status": "active"
    }
    
    with open('data/subscribers.json', 'w') as f:
        json.dump(users, f, indent=2)
    
    await update.message.reply_text(f"✅ User {new_user_id} added as subscriber (1 year)")

async def list_users_admin(update, context):
    chat_id = str(update.message.chat_id)
    if chat_id not in ADMIN_CHAT_IDS:
        await update.message.reply_text("❌ Admin only command")
        return
    
    try:
        with open('data/subscribers.json') as f:
            users = json.load(f)
        
        if not users:
            await update.message.reply_text("No subscribers found")
            return
        
        message = "📄 Subscribers:\n\n"
        for uid, info in users.items():
            status = "✅" if info['status'] == 'active' else "❌"
            message += f"{status} {uid} - {info.get('username', 'unknown')} (expires: {info['expires_on']})\n"
        
        await update.message.reply_text(message)
        
    except FileNotFoundError:
        await update.message.reply_text("No subscribers file found")

async def remove_user_admin(update, context):
    chat_id = str(update.message.chat_id)
    if chat_id not in ADMIN_CHAT_IDS:
        await update.message.reply_text("❌ Admin only command")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /removeuser <chat_id>")
        return
    
    user_to_remove = context.args[0]
    
    try:
        with open('data/subscribers.json') as f:
            users = json.load(f)
        
        if user_to_remove in users:
            del users[user_to_remove]
            with open('data/subscribers.json', 'w') as f:
                json.dump(users, f, indent=2)
            await update.message.reply_text(f"✅ User {user_to_remove} removed")
        else:
            await update.message.reply_text(f"❌ User {user_to_remove} not found")
            
    except FileNotFoundError:
        await update.message.reply_text("No subscribers file found")

def main():
    print("🤖 Starting bot...")
    
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("adduser", add_user_admin))
    app.add_handler(CommandHandler("listusers", list_users_admin))
    app.add_handler(CommandHandler("removeuser", remove_user_admin))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("✅ Bot is running...")
    print(f"🔑 Admin IDs: {ADMIN_CHAT_IDS}")
    app.run_polling()

if __name__ == "__main__":
    main()
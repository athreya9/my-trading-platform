from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import os
from dotenv import load_dotenv

load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_first_name = update.effective_user.first_name
    welcome_text = (
        f" Hey {user_first_name}!\n\n"
        "Welcome to DA-Tradingbot \n"
        "I’ll be sending you high-confidence trading signals, market updates, and actionable insights.\n\n"
        "✅ To get started, just stay tuned for alerts.\n"
        " If you ever need help or want to customize your experience, type /help.\n\n"
        "Let’s make smart trades together!"
    )
    await update.message.reply_text(welcome_text)

if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()
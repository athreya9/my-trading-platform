# AI Trading Platform 🚀

A complete AI-powered trading platform with real-time alerts, live data analysis, and automated signal generation.

## 🎯 Features

- **AI-Powered Signals**: Machine learning algorithms analyze market data
- **Telegram Alerts**: Real-time trading alerts sent to your phone
- **Live Dashboard**: React-based frontend with real-time updates
- **Automated Trading**: GitHub Actions for continuous signal generation
- **Risk Management**: Built-in stop-loss and target calculations
- **Backtesting**: Historical strategy performance analysis

## 🚀 Quick Start

### 1. Setup Telegram Bot
```bash
python3 setup_telegram.py
```

### 2. Start the Platform
```bash
python3 start_trading_platform.py
```

### 3. Access Dashboard
- Frontend: http://localhost:3000
- API: http://localhost:8000

## 📋 Complete Setup Guide

### Prerequisites
- Python 3.8+
- Node.js 16+
- Telegram account

### Step 1: Clone and Install
```bash
git clone <your-repo>
cd "My trading platform"
pip3 install -r api/requirements.txt
```

### Step 2: Configure Telegram
1. Create bot with @BotFather on Telegram
2. Run setup script: `python3 setup_telegram.py`
3. Follow the prompts to configure credentials

### Step 3: Test the System
```bash
python3 test_complete_system.py
```

### Step 4: Generate Initial Signals
```bash
python3 generate_test_signals.py
```

### Step 5: Start Everything
```bash
python3 start_trading_platform.py
```

## 🔧 Configuration

### Environment Variables (.env)
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
KITE_API_KEY=your_kite_key (optional)
KITE_ACCESS_TOKEN=your_access_token (optional)
```

## 📱 Telegram Setup

1. **Create Bot**:
   - Message @BotFather on Telegram
   - Send `/newbot`
   - Choose name and username
   - Save the bot token

2. **Get Chat ID**:
   - Start chat with your bot
   - Send any message
   - Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Find your chat ID in the response

## 🤖 How It Works

### Signal Generation
1. **Data Collection**: Fetches live market data using yfinance
2. **AI Analysis**: Machine learning models analyze technical indicators
3. **Signal Generation**: Generates BUY/SELL/HOLD signals with confidence scores
4. **Alert System**: High-confidence signals trigger Telegram alerts

### Automation
- **GitHub Actions**: Runs every 15 minutes during market hours
- **Live Bot**: Continuously monitors and generates signals
- **Auto-deployment**: Updates are automatically deployed

## 🔍 Troubleshooting

### Common Issues

1. **No Telegram Alerts**
   - Check bot token and chat ID in .env
   - Verify bot is started with @BotFather
   - Test with: `python3 setup_telegram.py`

2. **Empty Signals**
   - Run: `python3 generate_test_signals.py`
   - Check market hours (9:15 AM - 3:30 PM IST)
   - Verify internet connection for data fetching

3. **Frontend Not Loading**
   - Check if backend is running on port 8000
   - Verify CORS settings in api/main.py
   - Check frontend URL in origins list

## ⚠️ Disclaimer

This is for educational purposes only. Trading involves risk. Always do your own research and never invest more than you can afford to lose.

---

**Status**: ✅ Fully Operational
**Last Updated**: 2024
**Version**: 2.0


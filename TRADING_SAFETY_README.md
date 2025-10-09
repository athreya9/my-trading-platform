# 🚨 AUTO-TRADING SAFETY GUIDE

## ⚠️ CRITICAL SAFETY FEATURES

### 🔒 PAPER TRADING MODE (DEFAULT)
- **REAL MONEY TRADING IS DISABLED BY DEFAULT**
- All trades are simulated (paper trading)
- No real money at risk
- Test the system thoroughly before enabling real trading

### 🛡️ MULTIPLE SAFETY LAYERS

#### 1. **Daily Limits**
- Max ₹5,000 per day
- Max ₹1,000 per trade
- Max 3 open positions

#### 2. **Signal Validation**
- Minimum 85% confidence required
- Validates all required fields
- Checks position limits

#### 3. **Risk Management**
- 15% stop loss on all trades
- 20% profit target
- 30-second position monitoring

#### 4. **Emergency Controls**
- `/panic` command exits all positions instantly
- `/autotrade off` disables trading
- `/status` shows current state

## 🎮 TELEGRAM COMMANDS

### Admin Commands (Your ID: 1375236879)
```
/autotrade on    - Enable auto-trading
/autotrade off   - Disable auto-trading
/status          - Show trading status
/panic           - Emergency exit all positions
```

## 📝 PAPER TRADING FIRST

### Test Phase (Current)
1. **PAPER_TRADING = True** (in auto_trading_engine.py)
2. All trades are simulated
3. No real money involved
4. Test all features thoroughly

### Going Live (When Ready)
1. Change **PAPER_TRADING = False**
2. Set **REAL_TRADING_ENABLED = True**
3. Start with small amounts
4. Monitor closely

## 🔧 CONFIGURATION

### Current Settings (Conservative)
```python
DAILY_LIMIT = 5000          # ₹5,000 max per day
MAX_POSITION_SIZE = 1000    # ₹1,000 max per trade
MIN_CONFIDENCE = 0.85       # 85% minimum confidence
MAX_OPEN_POSITIONS = 3      # Max 3 positions
```

### Risk Levels
- **Conservative**: Current settings
- **Moderate**: Increase limits by 50%
- **Aggressive**: Double limits (NOT RECOMMENDED)

## 📊 MONITORING

### Real-time Alerts
- Trade execution notifications
- P&L updates
- Exit notifications
- Error alerts

### Data Logging
- All trades logged to `data/auto_trades.json`
- Open positions in `data/open_positions.json`
- Full audit trail maintained

## ⚡ QUICK START

1. **Test Paper Trading**
   ```bash
   python3 auto_trading_engine.py
   ```

2. **Enable Auto-Trading**
   - Send `/autotrade on` to bot
   - System will process high-confidence signals

3. **Monitor Status**
   - Send `/status` for current state
   - Watch Telegram for notifications

4. **Emergency Stop**
   - Send `/panic` to exit all positions
   - Send `/autotrade off` to disable

## 🚨 IMPORTANT WARNINGS

### ⚠️ NEVER
- Enable real trading without thorough testing
- Trade with money you can't afford to lose
- Ignore risk management rules
- Leave system unmonitored

### ✅ ALWAYS
- Start with paper trading
- Test all commands
- Monitor positions closely
- Keep emergency controls ready

## 📞 SUPPORT

If you encounter issues:
1. Check logs in console
2. Use `/status` command
3. Use `/panic` if needed
4. Disable with `/autotrade off`

**Remember: This is for educational purposes. Trading involves risk.**
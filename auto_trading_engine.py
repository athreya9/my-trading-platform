#!/usr/bin/env python3
"""
SECURE AUTO-TRADING ENGINE - PAPER TRADING FIRST
⚠️ REAL MONEY TRADING DISABLED BY DEFAULT
"""
import json
import os
import time
import threading
from datetime import datetime, timedelta
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureAutoTrader:
    def __init__(self):
        # SAFETY: Paper trading mode by default
        self.PAPER_TRADING = True  # CHANGE TO False ONLY WHEN READY
        self.REAL_TRADING_ENABLED = False
        
        # Telegram config
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.admin_id = "1375236879"
        self.bot = Bot(token=self.bot_token)
        
        # Trading limits (STRICT)
        self.DAILY_LIMIT = 5000  # ₹5000 max per day
        self.MAX_POSITION_SIZE = 1000  # ₹1000 max per trade
        self.MIN_CONFIDENCE = 0.85  # 85% minimum confidence
        self.MAX_OPEN_POSITIONS = 3  # Max 3 positions at once
        
        # State tracking
        self.auto_trade_enabled = False
        self.used_today = 0
        self.open_positions = []
        self.trade_log = []
        self.last_reset = datetime.now().date()
        
        # Files
        self.trade_log_file = "data/auto_trades.json"
        self.positions_file = "data/open_positions.json"
        
        # Kite connection
        self.kite = None
        self._init_kite()
        
        # Load existing data
        self._load_data()
    
    def _init_kite(self):
        """Initialize Kite connection"""
        if self.PAPER_TRADING:
            logger.info("📝 PAPER TRADING MODE - No real orders")
            return
        
        try:
            from kiteconnect import KiteConnect
            api_key = os.getenv('KITE_API_KEY')
            access_token = os.getenv('KITE_ACCESS_TOKEN')
            
            if not api_key or not access_token:
                logger.error("❌ Kite credentials missing")
                return
            
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            
            # Test connection
            profile = self.kite.profile()
            logger.info(f"✅ Kite connected: {profile['user_name']}")
            
        except Exception as e:
            logger.error(f"❌ Kite connection failed: {e}")
            self.kite = None
    
    def _load_data(self):
        """Load existing trades and positions"""
        os.makedirs('data', exist_ok=True)
        
        # Load trade log
        try:
            with open(self.trade_log_file, 'r') as f:
                self.trade_log = json.load(f)
        except FileNotFoundError:
            self.trade_log = []
        
        # Load open positions
        try:
            with open(self.positions_file, 'r') as f:
                self.open_positions = json.load(f)
        except FileNotFoundError:
            self.open_positions = []
        
        # Calculate used amount for today
        today = datetime.now().date()
        self.used_today = sum(
            trade['amount'] for trade in self.trade_log 
            if datetime.fromisoformat(trade['timestamp']).date() == today
        )
        
        logger.info(f"📊 Loaded: {len(self.trade_log)} trades, {len(self.open_positions)} open positions")
        logger.info(f"💰 Used today: ₹{self.used_today}/{self.DAILY_LIMIT}")
    
    def _save_data(self):
        """Save trades and positions"""
        with open(self.trade_log_file, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
        
        with open(self.positions_file, 'w') as f:
            json.dump(self.open_positions, f, indent=2)
    
    def _notify_admin(self, message):
        """Send notification to admin"""
        try:
            self.bot.send_message(chat_id=self.admin_id, text=message, parse_mode='HTML')
            logger.info(f"📱 Notified: {message}")
        except Exception as e:
            logger.error(f"❌ Notification failed: {e}")
    
    def _reset_daily_limits(self):
        """Reset daily limits if new day"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.used_today = 0
            self.last_reset = today
            logger.info("🔄 Daily limits reset")
    
    def _validate_signal(self, signal):
        """Validate signal before trading"""
        errors = []
        
        # Check if auto-trading is enabled
        if not self.auto_trade_enabled:
            errors.append("Auto-trading disabled")
        
        # Check confidence
        if signal.get('confidence', 0) < self.MIN_CONFIDENCE:
            errors.append(f"Low confidence: {signal.get('confidence', 0):.1%}")
        
        # Check daily limit
        amount = signal.get('entry_price', 0) * signal.get('quantity', 1)
        if self.used_today + amount > self.DAILY_LIMIT:
            errors.append(f"Daily limit exceeded: ₹{self.used_today + amount} > ₹{self.DAILY_LIMIT}")
        
        # Check position size
        if amount > self.MAX_POSITION_SIZE:
            errors.append(f"Position too large: ₹{amount} > ₹{self.MAX_POSITION_SIZE}")
        
        # Check max positions
        if len(self.open_positions) >= self.MAX_OPEN_POSITIONS:
            errors.append(f"Too many positions: {len(self.open_positions)}/{self.MAX_OPEN_POSITIONS}")
        
        # Check required fields
        required = ['symbol', 'entry_price', 'quantity', 'strike', 'option_type']
        for field in required:
            if not signal.get(field):
                errors.append(f"Missing {field}")
        
        return errors
    
    def execute_trade(self, signal):
        """Execute a trade (paper or real)"""
        self._reset_daily_limits()
        
        # Validate signal
        errors = self._validate_signal(signal)
        if errors:
            logger.warning(f"❌ Signal rejected: {', '.join(errors)}")
            self._notify_admin(f"🚫 <b>Trade Rejected</b>\n{signal['symbol']} {signal['strike']} {signal['option_type']}\n\n❌ {', '.join(errors)}")
            return False
        
        # Calculate trade details
        symbol = f"{signal['symbol']}{signal['strike']}{signal['option_type']}"
        entry_price = signal['entry_price']
        quantity = signal['quantity']
        amount = entry_price * quantity
        
        # Execute trade
        if self.PAPER_TRADING:
            # Paper trading
            order_id = f"PAPER_{int(time.time())}"
            status = "PAPER_EXECUTED"
            logger.info(f"📝 PAPER TRADE: {symbol} @ ₹{entry_price}")
        else:
            # Real trading (when enabled)
            if not self.kite:
                logger.error("❌ Kite not connected")
                return False
            
            try:
                order = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NFO,
                    tradingsymbol=symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=quantity,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.kite.PRODUCT_MIS
                )
                order_id = order['order_id']
                status = "EXECUTED"
                logger.info(f"✅ REAL TRADE: {symbol} @ ₹{entry_price}")
            except Exception as e:
                logger.error(f"❌ Trade execution failed: {e}")
                self._notify_admin(f"❌ <b>Trade Failed</b>\n{symbol}\nError: {str(e)}")
                return False
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'amount': amount,
            'confidence': signal['confidence'],
            'reason': signal.get('reason', 'AI Signal'),
            'status': status,
            'stoploss': entry_price * 0.85,  # 15% SL
            'target': entry_price * 1.20     # 20% target
        }
        
        self.trade_log.append(trade_record)
        self.open_positions.append(trade_record)
        self.used_today += amount
        self._save_data()
        
        # Notify admin
        mode = "📝 PAPER" if self.PAPER_TRADING else "💰 REAL"
        self._notify_admin(
            f"{mode} <b>TRADE EXECUTED</b>\n\n"
            f"📊 {symbol}\n"
            f"💰 Entry: ₹{entry_price}\n"
            f"📦 Qty: {quantity}\n"
            f"💵 Amount: ₹{amount}\n"
            f"🎯 Confidence: {signal['confidence']:.1%}\n"
            f"📈 Target: ₹{trade_record['target']}\n"
            f"🛑 SL: ₹{trade_record['stoploss']}\n\n"
            f"💳 Used today: ₹{self.used_today}/{self.DAILY_LIMIT}"
        )
        
        return True
    
    def monitor_positions(self):
        """Monitor open positions for exit conditions"""
        while True:
            try:
                for position in self.open_positions[:]:
                    if self.PAPER_TRADING:
                        # Paper trading - simulate price movement
                        import random
                        current_price = position['entry_price'] * (1 + random.uniform(-0.1, 0.1))
                    else:
                        # Real trading - get live price
                        if not self.kite:
                            continue
                        try:
                            ltp_data = self.kite.ltp(f"NFO:{position['symbol']}")
                            current_price = ltp_data[f"NFO:{position['symbol']}"]['last_price']
                        except:
                            continue
                    
                    # Check exit conditions
                    if current_price <= position['stoploss']:
                        self._exit_position(position, current_price, "Stoploss Hit")
                    elif current_price >= position['target']:
                        self._exit_position(position, current_price, "Target Hit")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Position monitoring error: {e}")
                time.sleep(60)
    
    def _exit_position(self, position, exit_price, reason):
        """Exit a position"""
        try:
            if self.PAPER_TRADING:
                logger.info(f"📝 PAPER EXIT: {position['symbol']} @ ₹{exit_price}")
            else:
                if self.kite:
                    self.kite.place_order(
                        variety=self.kite.VARIETY_REGULAR,
                        exchange=self.kite.EXCHANGE_NFO,
                        tradingsymbol=position['symbol'],
                        transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                        quantity=position['quantity'],
                        order_type=self.kite.ORDER_TYPE_MARKET,
                        product=self.kite.PRODUCT_MIS
                    )
            
            # Calculate P&L
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
            
            # Update position
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now().isoformat()
            position['pnl'] = pnl
            position['exit_reason'] = reason
            position['status'] = 'CLOSED'
            
            # Remove from open positions
            self.open_positions.remove(position)
            self._save_data()
            
            # Notify
            mode = "📝 PAPER" if self.PAPER_TRADING else "💰 REAL"
            profit_emoji = "💚" if pnl > 0 else "❤️"
            self._notify_admin(
                f"{mode} <b>POSITION CLOSED</b>\n\n"
                f"📊 {position['symbol']}\n"
                f"💰 Entry: ₹{position['entry_price']}\n"
                f"🚪 Exit: ₹{exit_price}\n"
                f"{profit_emoji} P&L: ₹{pnl:.2f} ({pnl_pct:+.1f}%)\n"
                f"📝 Reason: {reason}"
            )
            
        except Exception as e:
            logger.error(f"❌ Exit failed: {e}")
    
    def panic_exit_all(self):
        """Emergency exit all positions"""
        logger.warning("🚨 PANIC EXIT INITIATED")
        
        for position in self.open_positions[:]:
            try:
                if self.PAPER_TRADING:
                    exit_price = position['entry_price'] * 0.95  # Assume 5% loss
                else:
                    if self.kite:
                        ltp_data = self.kite.ltp(f"NFO:{position['symbol']}")
                        exit_price = ltp_data[f"NFO:{position['symbol']}"]['last_price']
                    else:
                        exit_price = position['entry_price']
                
                self._exit_position(position, exit_price, "PANIC EXIT")
                
            except Exception as e:
                logger.error(f"❌ Panic exit failed for {position['symbol']}: {e}")
        
        self._notify_admin("🚨 <b>PANIC EXIT COMPLETED</b>\nAll positions closed!")
    
    def process_signal(self, signal):
        """Process incoming signal for auto-trading"""
        if not signal:
            return
        
        logger.info(f"📡 Signal received: {signal.get('symbol')} {signal.get('confidence', 0):.1%}")
        
        # Only trade high-confidence signals
        if signal.get('confidence', 0) >= self.MIN_CONFIDENCE:
            self.execute_trade(signal)
        else:
            logger.info(f"⚠️ Signal ignored: Low confidence {signal.get('confidence', 0):.1%}")

# Global instance
auto_trader = SecureAutoTrader()

# Telegram commands
async def toggle_auto_trade(update, context):
    """Toggle auto-trading on/off"""
    if str(update.message.chat_id) != auto_trader.admin_id:
        await update.message.reply_text("🚫 Unauthorized")
        return
    
    if not context.args:
        status = "ON" if auto_trader.auto_trade_enabled else "OFF"
        await update.message.reply_text(f"Auto-trading is {status}")
        return
    
    cmd = context.args[0].lower()
    if cmd == "on":
        auto_trader.auto_trade_enabled = True
        mode = "📝 PAPER" if auto_trader.PAPER_TRADING else "💰 REAL"
        await update.message.reply_text(f"✅ Auto-trading ENABLED\n{mode} MODE")
    elif cmd == "off":
        auto_trader.auto_trade_enabled = False
        await update.message.reply_text("🛑 Auto-trading DISABLED")
    else:
        await update.message.reply_text("Usage: /autotrade on|off")

async def panic_command(update, context):
    """Emergency panic exit"""
    if str(update.message.chat_id) != auto_trader.admin_id:
        await update.message.reply_text("🚫 Unauthorized")
        return
    
    auto_trader.panic_exit_all()
    await update.message.reply_text("🚨 PANIC EXIT EXECUTED!")

async def status_command(update, context):
    """Show trading status"""
    if str(update.message.chat_id) != auto_trader.admin_id:
        await update.message.reply_text("🚫 Unauthorized")
        return
    
    mode = "📝 PAPER" if auto_trader.PAPER_TRADING else "💰 REAL"
    status = "ON" if auto_trader.auto_trade_enabled else "OFF"
    
    message = (
        f"<b>AUTO-TRADING STATUS</b>\n\n"
        f"🤖 Mode: {mode}\n"
        f"⚡ Status: {status}\n"
        f"💰 Used today: ₹{auto_trader.used_today}/{auto_trader.DAILY_LIMIT}\n"
        f"📊 Open positions: {len(auto_trader.open_positions)}/{auto_trader.MAX_OPEN_POSITIONS}\n"
        f"📈 Total trades: {len(auto_trader.trade_log)}\n\n"
        f"⚠️ Min confidence: {auto_trader.MIN_CONFIDENCE:.0%}\n"
        f"💵 Max position: ₹{auto_trader.MAX_POSITION_SIZE}"
    )
    
    await update.message.reply_text(message, parse_mode='HTML')

def start_auto_trader():
    """Start the auto-trading system"""
    logger.info("🚀 Starting Auto-Trading Engine")
    
    # Start position monitoring
    monitor_thread = threading.Thread(target=auto_trader.monitor_positions, daemon=True)
    monitor_thread.start()
    
    # Setup Telegram bot
    app = Application.builder().token(auto_trader.bot_token).build()
    app.add_handler(CommandHandler("autotrade", toggle_auto_trade))
    app.add_handler(CommandHandler("panic", panic_command))
    app.add_handler(CommandHandler("status", status_command))
    
    # Start bot
    logger.info("✅ Auto-trader ready")
    app.run_polling()

if __name__ == "__main__":
    start_auto_trader()
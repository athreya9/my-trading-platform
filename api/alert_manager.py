# api/alert_manager.py
from datetime import datetime, time, timedelta

class AlertManager:
    def __init__(self):
        self.alert_cooldown = {
            'high_confidence': 30,  # minutes
            'medium_confidence': 60,
            'summary': 1440  # 24 hours
        }
        self.daily_alert_count = 0
        self.max_daily_alerts = 5
        self.last_alert_time = {}
        
    def manage_alert(self, signal):
        """Smart alert management system"""
        
        # 1. Daily Alert Limit
        if self.daily_alert_count >= self.max_daily_alerts:
            return False, "Daily alert limit reached"
            
        # 2. Market Hours Check
        if not self._during_market_hours():
            if signal.get('urgency') != 'critical':
                return False, "Outside market hours"
                
        # 3. Confidence-Based Filtering
        confidence = signal.get('confidence', 0)
        if confidence < self._get_confidence_threshold():
            return False, f"Confidence too low: {confidence}%"
            
        # 4. Cooldown Check
        symbol = signal.get('symbol')
        if self._in_cooldown(symbol, confidence):
            return False, f"Symbol {symbol} in cooldown"
            
        self.daily_alert_count += 1
        self._update_cooldown(symbol, confidence)
        
        return True, "Alert can be sent"

    def _get_confidence_threshold(self):
        """Dynamic confidence threshold based on market conditions"""
        current_hour = datetime.now().hour
        
        if current_hour in [9, 15]:  # Market open/close
            return 70  # Higher threshold during volatile periods
        else:
            return 75  # Standard threshold
            
    def _during_market_hours(self):
        """Check if market is open"""
        now = datetime.now().time()
        market_open = time(9, 15)
        market_close = time(15, 30)
        return market_open <= now <= market_close

    def _in_cooldown(self, symbol, confidence):
        """Check if a symbol is in cooldown"""
        last_alert = self.last_alert_time.get(symbol)
        if not last_alert:
            return False

        cooldown_minutes = self.alert_cooldown.get('high_confidence' if confidence >= 75 else 'medium_confidence', 60)
        
        if (datetime.now() - last_alert).seconds < cooldown_minutes * 60:
            return True
        
        return False

    def _update_cooldown(self, symbol, confidence):
        """Update the last alert time for a symbol"""
        self.last_alert_time[symbol] = datetime.now()
# api/test_alert_manager.py
import unittest
from .alert_manager import AlertManager
from datetime import datetime, timedelta

class TestAlertManager(unittest.TestCase):
    def test_alert_manager(self):
        alert_manager = AlertManager()

        # Test 1: High confidence signal, should pass
        signal1 = {'symbol': 'BANKNIFTY', 'confidence': 80, 'urgency': 'critical'}
        should_send, reason = alert_manager.manage_alert(signal1)
        self.assertTrue(should_send)

        # Test 2: Low confidence signal, should fail
        signal2 = {'symbol': 'NIFTY', 'confidence': 60, 'urgency': 'critical'}
        should_send, reason = alert_manager.manage_alert(signal2)
        self.assertFalse(should_send)

        # Test 3: High confidence signal for the same symbol within cooldown, should fail
        signal3 = {'symbol': 'BANKNIFTY', 'confidence': 85, 'urgency': 'critical'}
        should_send, reason = alert_manager.manage_alert(signal3)
        self.assertFalse(should_send)
        
        # Test 4: High confidence signal for a different symbol, should pass
        signal4 = {'symbol': 'RELIANCE', 'confidence': 90, 'urgency': 'critical'}
        should_send, reason = alert_manager.manage_alert(signal4)
        self.assertTrue(should_send)

        # Test 5: Daily alert limit
        alert_manager.daily_alert_count = 5
        signal5 = {'symbol': 'HDFCBANK', 'confidence': 80, 'urgency': 'critical'}
        should_send, reason = alert_manager.manage_alert(signal5)
        self.assertFalse(should_send)


if __name__ == '__main__':
    unittest.main()
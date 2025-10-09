#!/usr/bin/env python3
"""
Final Guardrails - Automated validation for all signals
"""
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

class TradingGuardrails:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.min_confidence = 85
        self.allowed_sources = ['KITE']
    
    def is_market_open(self):
        """Market hours validation"""
        now = datetime.now(self.ist)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def validate_signal(self, signal):
        """Final validation before dispatch"""
        errors = []
        
        # No demo/educational trades
        if any(word in str(signal).lower() for word in ['demo', 'educational', 'fake', 'test']):
            errors.append("Demo/educational content detected")
        
        # Market must be open
        if not self.is_market_open():
            errors.append("Market is closed")
        
        # Only KITE sources
        if signal.get('source') not in self.allowed_sources:
            errors.append(f"Invalid source: {signal.get('source')}")
        
        # Confidence gating
        if signal.get('confidence', 0) < self.min_confidence:
            errors.append(f"Low confidence: {signal.get('confidence')}%")
        
        # Log validation result
        if errors:
            logger.warning(f"❌ Signal rejected: {', '.join(errors)}")
            return False, errors
        
        logger.info(f"✅ Signal validated: {signal.get('symbol')} {signal.get('confidence')}%")
        return True, []

# Global guardrails instance
guardrails = TradingGuardrails()

def validate_before_dispatch(signal):
    """Validate signal before any public dispatch"""
    return guardrails.validate_signal(signal)
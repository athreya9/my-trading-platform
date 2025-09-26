# test_data_pipeline.py
from .data_collector import DataCollector

def test_immediately():
    """Test the new data collection system"""
    collector = DataCollector()
    
    # Test with one symbol
    data = collector.fetch_historical_data('RELIANCE.NS', period='1mo')
    if data is not None:
        collector.save_daily_data('RELIANCE.NS', data)
        print("✅ Data pipeline test successful!")
        
        # Send test alert
        collector.send_data_collection_alert('RELIANCE.NS', len(data))
    else:
        print("❌ Data pipeline test failed")

if __name__ == '__main__':
    test_immediately()
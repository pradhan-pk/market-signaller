
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
    NEWS_API_KEY = os.getenv('NEWS_API')

    # Snowflake Configuration
    SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
    SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
    SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
    SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
    SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE', 'FINANCIAL_DATA')
    SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')

    # Application Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    DEFAULT_REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', '30'))

    # Signal Generation Parameters
    DEFAULT_SIGNAL_PARAMS = {
        'rsi_oversold': int(os.getenv('RSI_OVERSOLD', '30')),
        'rsi_overbought': int(os.getenv('RSI_OVERBOUGHT', '70')),
        'sentiment_threshold': float(os.getenv('SENTIMENT_THRESHOLD', '0.5')),
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.3'))
    }

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_vars = [
            'ALPHA_VANTAGE_API_KEY',
            'NEWS_API_KEY',
            'SNOWFLAKE_ACCOUNT',
            'SNOWFLAKE_USER',
            'SNOWFLAKE_PASSWORD'
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)

        if missing_vars:
            print(f"Warning: Missing required environment variables: {missing_vars}")
            return False

        return True

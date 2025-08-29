
import logging
from src.config import DatabaseConfig
from src.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_database_connection():
    """Test database connection with current configuration."""
    try:
        # Create database config (you can modify this based on your setup)
        db_config = DatabaseConfig.from_env()
        
        print(f"Testing connection to: {db_config.server}/{db_config.database}")
        print(f"Using driver: {db_config.driver}")
        print(f"Trusted connection: {db_config.trusted_connection}")
        
        # Create database manager
        db_manager = DatabaseManager(db_config)
        
        # Test connection
        if db_manager.test_connection():
            print("✅ Database connection successful!")
            
            # Get database info
            info = db_manager.get_database_info()
            if info:
                print(f"📊 Database: {info['database_name']}")
                print(f"📊 Tables: {info['table_count']}")
                print(f"📊 Version: {info['version'][:100]}...")  # Truncate version string
            
        else:
            print("❌ Database connection failed!")
            
    except Exception as e:
        print(f"❌ Error during connection test: {str(e)}")

if __name__ == "__main__":
    test_database_connection()

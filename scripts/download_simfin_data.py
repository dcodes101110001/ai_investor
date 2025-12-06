#!/usr/bin/env python3
"""
SimFin Data Download Script
Downloads required SimFin data files into the stock_data directory.
Requires SIMFIN_API_KEY environment variable to be set.
"""

import os
import sys
import logging
from pathlib import Path

# Try importing simfin at module level
try:
    import simfin as sf
    SIMFIN_AVAILABLE = True
except ImportError:
    SIMFIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_data_directory():
    """Ensure the stock_data directory exists."""
    data_dir = Path('stock_data')
    
    if data_dir.exists():
        logger.info(f"Directory '{data_dir}' already exists")
    else:
        logger.info(f"Creating directory '{data_dir}'")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory '{data_dir}' created successfully")
    
    return data_dir


def download_simfin_data():
    """Download SimFin data files using the SimFin Python API."""
    global SIMFIN_AVAILABLE
    
    try:
        # Check if simfin is available
        if not SIMFIN_AVAILABLE:
            logger.error("SimFin library not installed. Please install it with: pip install simfin")
            return False
        
        # Get API key from environment variable
        api_key = os.environ.get('SIMFIN_API_KEY')
        if not api_key:
            logger.error("SIMFIN_API_KEY environment variable not set")
            logger.error("Please set your SimFin API key to download data")
            return False
        
        # Set API key and data directory
        logger.info("Setting SimFin API key...")
        sf.set_api_key(api_key)
        
        data_dir = setup_data_directory()
        sf.set_data_dir(str(data_dir))
        
        logger.info("Starting SimFin data download...")
        
        # Download required datasets for US market
        datasets = [
            ('Income Statements', lambda: sf.load_income(variant='annual', market='us')),
            ('Balance Sheets', lambda: sf.load_balance(variant='annual', market='us')),
            ('Cash Flow Statements', lambda: sf.load_cashflow(variant='annual', market='us')),
            ('Share Prices', lambda: sf.load_shareprices(variant='daily', market='us'))
        ]
        
        success_count = 0
        for dataset_name, download_func in datasets:
            try:
                logger.info(f"Downloading {dataset_name}...")
                df = download_func()
                logger.info(f"✓ {dataset_name} downloaded successfully (shape: {df.shape})")
                success_count += 1
            except Exception as e:
                logger.error(f"✗ Failed to download {dataset_name}: {str(e)}")
        
        # Check if all datasets were downloaded
        if success_count == len(datasets):
            logger.info("=" * 60)
            logger.info("All SimFin datasets downloaded successfully!")
            logger.info(f"Data saved to: {data_dir.absolute()}")
            logger.info("=" * 60)
            return True
        else:
            logger.warning(f"Downloaded {success_count}/{len(datasets)} datasets")
            logger.warning("Some datasets failed to download. Check logs above.")
            return False
            
    except Exception as e:
        logger.error(f"Error during download: {str(e)}", exc_info=True)
        return False


def main():
    """Main function to orchestrate the download process."""
    logger.info("=" * 60)
    logger.info("SimFin Data Download Script")
    logger.info("=" * 60)
    
    try:
        # Setup directory
        data_dir = setup_data_directory()
        
        # Download data
        success = download_simfin_data()
        
        if success:
            logger.info("Script completed successfully!")
            sys.exit(0)
        else:
            logger.error("Script completed with errors!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

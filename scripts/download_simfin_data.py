#!/usr/bin/env python3
"""
SimFin Data Downloader Script

This script downloads bulk data from SimFin (https://simfin.com/) 
and saves it to the stock_data directory.

Required files downloaded:
- us-income-annual-full-asreported.csv
- us-balance-annual-full-asreported.csv
- us-cashflow-annual-full-asreported.csv
- us-shareprices-daily.csv

Environment Variables:
- SIMFIN_API_KEY: Your SimFin API key (optional for free data)
"""

import os
import sys
import logging
import requests
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# SimFin API configuration
# SimFin bulk data download endpoint
# Reference: https://simfin.com/api/v2/documentation/
SIMFIN_BASE_URL = "https://www.simfin.com/api/v2/bulk/download"
SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY", "")

# Files to download from SimFin
FILES_TO_DOWNLOAD = [
    "us-income-annual-full-asreported.csv",
    "us-balance-annual-full-asreported.csv",
    "us-cashflow-annual-full-asreported.csv",
    "us-shareprices-daily.csv",
]


def ensure_directory_exists(directory_path):
    """
    Ensure the target directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
    
    Returns:
        Path object of the directory
    """
    path = Path(directory_path)
    if not path.exists():
        logger.info(f"Directory {directory_path} does not exist. Creating it...")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory {directory_path} created successfully.")
    else:
        logger.info(f"Directory {directory_path} already exists.")
    return path


def download_file(url, destination_path, timeout=300):
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        destination_path: Local path to save the file
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting download from: {url}")
        logger.info(f"Saving to: {destination_path}")
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination_path, 'wb') as f:
            if total_size > 0:
                logger.info(f"File size: {total_size / (1024*1024):.2f} MB")
                downloaded = 0
                last_logged = 0
                log_interval = 10 * 1024 * 1024  # Log every 10MB
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress at intervals
                        if downloaded - last_logged >= log_interval:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
                            last_logged = downloaded
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        file_size = os.path.getsize(destination_path)
        logger.info(f"Successfully downloaded {destination_path.name} ({file_size / (1024*1024):.2f} MB)")
        return True
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while downloading {url}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while downloading {url}")
        return False
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while downloading {url}: {e}")
        if e.response is not None:
            logger.error(f"Status code: {e.response.status_code}")
            if e.response.status_code == 401:
                logger.error("Authentication failed. Please check your SIMFIN_API_KEY.")
            elif e.response.status_code == 404:
                logger.error("File not found. The URL may be incorrect.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while downloading {url}: {e}")
        return False


def download_simfin_data(output_directory="stock_data"):
    """
    Download all required SimFin data files.
    
    Args:
        output_directory: Directory to save downloaded files
    
    Returns:
        True if all downloads successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("SimFin Data Download Started")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Ensure output directory exists
    output_path = ensure_directory_exists(output_directory)
    
    # Check for API key
    if SIMFIN_API_KEY:
        logger.info("SimFin API key found.")
    else:
        logger.warning("No SimFin API key found. Attempting to download free data.")
        logger.warning("If downloads fail, please set SIMFIN_API_KEY environment variable.")
    
    success_count = 0
    failure_count = 0
    failed_files = []
    
    # Define dataset mappings for SimFin API v2
    # Format: filename -> (dataset, variant, market)
    dataset_mappings = {
        "us-income-annual-full-asreported.csv": ("income", "annual-full-asreported", "us"),
        "us-balance-annual-full-asreported.csv": ("balance", "annual-full-asreported", "us"),
        "us-cashflow-annual-full-asreported.csv": ("cashflow", "annual-full-asreported", "us"),
        "us-shareprices-daily.csv": ("shareprices", "daily", "us"),
    }
    
    for filename in FILES_TO_DOWNLOAD:
        logger.info("-" * 80)
        logger.info(f"Processing file: {filename}")
        
        # Get dataset parameters
        if filename in dataset_mappings:
            dataset, variant, market = dataset_mappings[filename]
            
            # Construct download URL using SimFin API v2 format
            # https://www.simfin.com/api/v2/bulk/download?dataset=income&variant=annual-full-asreported&market=us&api-key=YOUR_KEY
            params = [
                f"dataset={dataset}",
                f"variant={variant}",
                f"market={market}",
            ]
            
            if SIMFIN_API_KEY:
                params.append(f"api-key={SIMFIN_API_KEY}")
            
            url = f"{SIMFIN_BASE_URL}?{'&'.join(params)}"
        else:
            logger.error(f"Unknown file mapping for {filename}")
            failure_count += 1
            failed_files.append(filename)
            continue
        
        destination = output_path / filename
        
        if download_file(url, destination):
            success_count += 1
        else:
            failure_count += 1
            failed_files.append(filename)
    
    # Summary
    logger.info("=" * 80)
    logger.info("Download Summary")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(FILES_TO_DOWNLOAD)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed downloads: {failure_count}")
    
    if failed_files:
        logger.error(f"Failed files: {', '.join(failed_files)}")
        logger.error("\nPlease check:")
        logger.error("1. Your internet connection")
        logger.error("2. SimFin API key is valid (if using premium data)")
        logger.error("3. SimFin service is available")
        logger.error("4. File URLs are correct")
    
    logger.info("=" * 80)
    
    if failure_count == 0:
        logger.info("All files downloaded successfully!")
        return True
    else:
        logger.error("Some files failed to download.")
        return False


def main():
    """Main entry point for the script."""
    try:
        # Get output directory from environment or use default
        output_dir = os.getenv("STOCK_DATA_DIR", "stock_data")
        
        # Run the download
        success = download_simfin_data(output_dir)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

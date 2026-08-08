"""
Script to download TSL dataset
"""

import os
import subprocess
from pathlib import Path
import requests
import zipfile


# Dataset URLs
TSL_DATASET_URL = "https://github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset/archive/refs/heads/master.zip"
ARASL_DATASET_URL = "https://huggingface.co/datasets/pain/ArASL_Database_Grayscale"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file from URL
    
    Args:
        url: Download URL
        dest_path: Destination path
        
    Returns:
        True if successful
    """
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {dest_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """
    Extract zip file
    
    Args:
        zip_path: Path to zip file
        dest_dir: Destination directory
        
    Returns:
        True if successful
    """
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        
        print(f"Extracted to {dest_dir}")
        return True
        
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def download_tsl_dataset():
    """Download TSL dataset from GitHub"""
    print("\n" + "="*60)
    print("Downloading TSL Dataset")
    print("="*60)
    
    # Create directory
    tsl_dir = RAW_DATA_DIR / "tsl_dataset"
    tsl_dir.mkdir(parents=True, exist_ok=True)
    
    # Download zip
    zip_path = tsl_dir / "tsl_dataset.zip"
    
    if not download_file(TSL_DATASET_URL, zip_path):
        return False
    
    # Extract
    if not extract_zip(zip_path, tsl_dir):
        return False
    
    # Rename extracted folder
    extracted_folder = tsl_dir / "First-ever-Tunisian-Sign-Language-Dataset-master"
    if extracted_folder.exists():
        final_folder = tsl_dir / "data"
        extracted_folder.rename(final_folder)
    
    print("\nTSL Dataset downloaded successfully!")
    print(f"Location: {tsl_dir}")
    
    return True


def download_arasl_dataset():
    """Download ArASL dataset (instructions)"""
    print("\n" + "="*60)
    print("ArASL Dataset Download Instructions")
    print("="*60)
    
    print("""
The ArASL dataset needs to be downloaded manually from HuggingFace:

1. Visit: https://huggingface.co/datasets/pain/ArASL_Database_Grayscale
2. Click on "Files and versions" tab
3. Download the dataset files
4. Extract to: data/raw/arasl/

Or use HuggingFace CLI:
    pip install huggingface_hub
    huggingface-cli download pain/ArASL_Database_Grayscale --repo-type dataset --local-dir data/raw/arasl/
""")
    
    return True


def main():
    """Main download function"""
    print("="*60)
    print("TSL Translator - Dataset Download")
    print("="*60)
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download TSL dataset
    download_tsl_dataset()
    
    # ArASL instructions
    download_arasl_dataset()
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDatasets location: {RAW_DATA_DIR}")
    print("\nNext step: Run 'python scripts/prepare_data.py' to process the data")


if __name__ == "__main__":
    main()

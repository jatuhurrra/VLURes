import os
import requests
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

# =======================================================================
# CONFIGURATION
# =======================================================================

# The Hugging Face Hub repository ID for your dataset.
DATASET_REPO_ID = "atamiles/VLURes"

# The local directory where the dataset will be saved.
# This will create subdirectories for each language (e.g., data/en, data/jp).
LOCAL_DATA_DIR = "data"

# Number of parallel workers for downloading images.
# Adjust based on your network and CPU capabilities.
NUM_WORKERS = 16

# =======================================================================

def download_image(args):
    """
    Downloads and verifies a single image from a URL.
    This function is designed to be used with a ThreadPoolExecutor.

    Args:
        args (tuple): A tuple containing (image_url, save_path).
    
    Returns:
        None. Errors are printed to the console.
    """
    image_url, save_path = args

    # Skip download if the file already exists to make the script restartable.
    if os.path.exists(save_path):
        return

    try:
        # Make the HTTP request to get the image.
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Write the image content to the specified file.
        with open(save_path, 'wb') as f:
            f.write(response.content)

        # Crucial Step: Verify that the downloaded file is a valid image.
        # This prevents corrupted or empty files from polluting the dataset.
        try:
            with Image.open(save_path) as img:
                img.verify()
        except (IOError, SyntaxError, Image.UnidentifiedImageError):
            # If verification fails, delete the corrupted file.
            print(f"\nWarning: Corrupted image downloaded from {image_url}. Deleting file.")
            os.remove(save_path)

    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., connection error, timeout, 404).
        print(f"\nWarning: Could not download image from {image_url}. Error: {e}")
    except Exception as e:
        # Handle other unexpected errors.
        print(f"\nWarning: An unexpected error occurred for URL {image_url}: {e}")

def main():
    """
    Main function to download the VLURes dataset, including all associated images.
    """
    print("=====================================================")
    print(" VLURes Benchmark Data Download Script")
    print("=====================================================")
    print(f"Dataset Repository: {DATASET_REPO_ID}")
    print(f"Local Save Directory: {LOCAL_DATA_DIR}\n")

    # Load the dataset metadata from the Hugging Face Hub.
    # This will download and cache the dataset information (JSON/Parquet files).
    try:
        print("Loading dataset metadata from Hugging Face Hub...")
        dataset_dict = load_dataset(DATASET_REPO_ID)
        print("Metadata loaded successfully.\n")
    except Exception as e:
        print(f"Error: Could not load dataset from Hugging Face Hub. Please check your internet connection and the repository name.")
        print(f"Details: {e}")
        return

    # Iterate through each split (language) in the dataset.
    for lang in dataset_dict.keys():
        print(f"--- Processing language: {lang.upper()} ---")
        
        # Create a specific directory for the current language's images.
        lang_image_dir = os.path.join(LOCAL_DATA_DIR, lang, "images")
        os.makedirs(lang_image_dir, exist_ok=True)
        print(f"Image files will be saved to: {lang_image_dir}")

        # Prepare a list of download tasks (URL, save_path) for the thread pool.
        download_tasks = []
        for example in dataset_dict[lang]:
            # Assumes the dataset columns are named 'image_url' and 'id'.
            # Adjust these keys if your dataset uses different column names.
            image_url = example.get("image_url")
            unique_id = example.get("id")

            if not image_url or not unique_id:
                print(f"Warning: Skipping an entry with missing 'image_url' or 'id' in the '{lang}' split.")
                continue

            # Construct the save path for the image.
            # Using a simple ID-based naming convention.
            filename = f"{unique_id}.jpg" # Assuming most images are jpeg
            save_path = os.path.join(lang_image_dir, filename)
            download_tasks.append((image_url, save_path))

        if not download_tasks:
            print("No images to download for this language.")
            continue
            
        # Use a ThreadPoolExecutor to download images in parallel with a progress bar.
        print(f"Starting download of {len(download_tasks)} images for {lang.upper()} using {NUM_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Use tqdm to create a progress bar for the downloads.
            list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks), desc=f"Downloading {lang.upper()} images"))

        print(f"Finished processing language: {lang.upper()}\n")

    print("=====================================================")
    print("All downloads complete. The VLURes dataset is ready.")
    print("=====================================================")

if __name__ == "__main__":
    main()

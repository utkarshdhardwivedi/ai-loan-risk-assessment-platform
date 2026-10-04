import os
import sys

DATASET_SLUG = "laotse/credit-risk-dataset"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def download():
    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

    try:
        kaggle.api.authenticate()
    except Exception as exc:
        print(f"ERROR: Kaggle authentication failed: {exc}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(DATASET_SLUG, path=OUTPUT_DIR, unzip=True)
    print("Download complete.")


if __name__ == "__main__":
    download()

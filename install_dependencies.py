import subprocess
import sys
import os
import requests
from tqdm import tqdm

def download_llm_model():
    """Downloads the Phi-2 GGUF model if not already present."""
    model_dir = "models"
    model_name = "phi-2.Q4_K_M.gguf"
    model_path = os.path.join(model_dir, model_name)

    if os.path.exists(model_path):
        print(f"LLM model '{model_name}' already exists. Skipping download.")
        return True

    print("LLM model not found. Starting download...")
    os.makedirs(model_dir, exist_ok=True)

    url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(model_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        print(f"\nLLM model '{model_name}' downloaded successfully to '{model_dir}/'.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading LLM model: {e}")
        # Clean up partially downloaded file
        if os.path.exists(model_path):
            os.remove(model_path)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during model download: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

def install_requirements():
    """Installs the packages listed in requirements.txt."""
    print("Installing Python packages from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages from requirements.txt installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print("Please check your internet connection and try again.")
        return False
    except FileNotFoundError:
        print("Error: requirements.txt not found.")
        return False

def download_nltk_data():
    """Downloads the required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        packages = ["punkt", "punkt_tab", "vader_lexicon", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]
        all_successful = True
        for pkg in packages:
            print(f"Downloading {pkg}...")
            try:
                nltk.download(pkg)
                print(f"'{pkg}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading '{pkg}': {e}")
                all_successful = False

        if all_successful:
            print("All NLTK data downloaded successfully.")
        else:
            print("Some NLTK data failed to download. Please check your internet connection.")

        return all_successful
    except ImportError:
        print("NLTK not found. Please ensure it is installed.")
        return False

if __name__ == "__main__":
    if install_requirements():
        # Only attempt to download NLTK data if NLTK was installed successfully
        if download_nltk_data():
            if download_llm_model():
                # Create a file to indicate that the installation is complete.
                with open(".install_complete", "w") as f:
                    f.write("Installation complete.")
                print("\nDependency installation complete.")
            else:
                print("\nLLM model download failed.")
        else:
            print("\nNLTK data download failed.")
    else:
        print("\nDependency installation failed.")

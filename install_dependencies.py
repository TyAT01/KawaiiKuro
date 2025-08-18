import subprocess
import sys

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
            # Create a file to indicate that the installation is complete.
            with open(".install_complete", "w") as f:
                f.write("Installation complete.")
            print("\nDependency installation complete.")
        else:
            print("\nNLTK data download failed.")
    else:
        print("\nDependency installation failed.")

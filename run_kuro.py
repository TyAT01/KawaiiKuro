import os
import subprocess
import sys

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def download_nltk_data():
    """Downloads necessary NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        packages = ["punkt", "punkt_tab", "vader_lexicon", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]
        for pkg in packages:
            nltk.download(pkg)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        sys.exit(1)

def main():
    """Main function to run the setup and start the application."""
    install_flag = ".install_complete"

    if not os.path.exists(install_flag):
        print("First-time setup: Installing dependencies and downloading data...")
        install_dependencies()
        download_nltk_data()
        with open(install_flag, "w") as f:
            f.write("Installation complete.")
        print("Setup complete.")
    else:
        print("Dependencies and data are already installed.")

    print("Starting Kuro...")
    try:
        subprocess.run([sys.executable, "kuro_app.py"] + sys.argv[1:])
    except Exception as e:
        print(f"Error starting Kuro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

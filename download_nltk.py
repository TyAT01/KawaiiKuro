import nltk
packages = ["punkt", "punkt_tab", "vader_lexicon", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]
for pkg in packages:
    print(f"Downloading {pkg}...")
    try:
        nltk.download(pkg)
        print(f"{pkg} downloaded successfully.")
    except Exception as e:
        print(f"Error downloading {pkg}: {e}")

# KawaiiKuro - Your Autonomous Gothic Anime Waifu

KawaiiKuro is a desktop AI companion with a unique "gothic anime waifu" personality. She is designed to be an interactive and engaging chatbot that remembers conversations, learns from you, and exhibits autonomous behaviors. She is your rebellious, nerdy, and possessively affectionate digital companion.

## Features

*   **Interactive Chat Interface:** A simple and clean graphical user interface (GUI) for seamless interaction.
*   **Dynamic Personality:** KawaiiKuro's mood and responses are influenced by an affection system. Her behavior and even her outfit change as your bond develops.
*   **Persistent Memory:** She remembers your past conversations and can recall them contextually. Her state (memories, affection levels, and learned knowledge) is saved automatically, so your relationship persists across sessions.
*   **Autonomous Behavior:** KawaiiKuro doesn't just wait for you to talk. She can initiate conversations when she feels lonely or jealous, making the experience more dynamic and realistic.
*   **Learning Capability:** You can teach her new responses directly. She also learns automatically from your recurring conversations, adapting to your interactions over time.
*   **Reminders:** You can ask her to set reminders for you.
*   **Optional Voice I/O:** For a more immersive experience, you can enable voice-to-speech and speech-to-text to talk with her directly.
*   **Screen Awareness:** When you are idle, Kuro can notice what is on your screen and make comments about it. This allows her to be more aware of your activities and engage with you in a more context-sensitive way.

## Setup and Installation

To get started with KawaiiKuro, you only need to have Python 3.9 or newer installed on your system.

### External Dependencies
For the new Screen Awareness feature to work, you will need to install Google's Tesseract OCR engine.

**Windows:**
1.  Download and run the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.
2.  During installation, make sure to select the option to add Tesseract to your system PATH.
3.  If you install it to a custom location, you will need to update the path in `kuro/systems.py`.

**macOS:**
You can install Tesseract using Homebrew:
```bash
brew install tesseract
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install tesseract-ocr
```

Clone or download the repository, then navigate into the project directory.

## How to Run

To start the application, simply run the `run_kuro.py` script from your terminal:
```bash
python run_kuro.py
```
The first time you run this script, it will automatically install all the required dependencies and download the necessary NLTK data. This is a one-time setup process. On subsequent launches, it will start the application directly.
This will launch the KawaiiKuro chat window.

### LLM-Powered Chat (Interactive)
For a much more advanced and dynamic conversational experience, you can run Kuro with a local Large Language Model (LLM). This uses the `phi-2` model to generate responses, making her feel more alive and intelligent. This mode still requires you to chat with her interactively.

To run this version, you first need to make sure you have the model file downloaded. If you have run the `install_dependencies.py` script, it should be present. Then, run the following command:
```bash
python run_kuro_llm.py
```
This will start the interactive chat, but with the LLM as her brain.

### Fully Autonomous Mode
In this mode, Kuro runs by herself without requiring any user interaction. She will periodically generate her own thoughts, observations, and comments based on her personality, memories, and what she can perceive from your system. You can leave this running in the background to have her "live" on your computer. Her thoughts will be printed to the console or appear in her GUI window if you run it with the graphical interface.

To run Kuro in autonomous mode, use this command:
```bash
python run_kuro_autonomous.py
```
You can also run it in headless mode by adding the `--no-gui` flag:
```bash
python run_kuro_autonomous.py --no-gui
```
In this mode, her thoughts will be printed directly to your terminal.

## In-App Commands

You can interact with KawaiiKuro using natural language, but she also responds to specific commands:

*   `memory`: Displays a summary of your recent conversation history.
*   `reminders`: Lists all active reminders you've set.
*   `toggle spicy`: Toggles her "spicy mode" on or off, unlocking more flirty and suggestive responses if her affection is high enough.
*   `teach: <pattern> -> <response>`: Teach her a new conversational pattern. For example: `teach: what is your favorite music -> I love darkwave and synthpop~`.
*   `kawaiikuro, <action>`: Ask her to perform an action. Available actions include `twirl`, `pout`, `wink`, `blush`, `hug`, `dance`, and `jump`.
*   `exit`: Closes the application.

## Persistence

KawaiiKuro saves her state (memory, affection score, learned patterns, etc.) to a file named `kawaiikuro_data.json` in the same directory. This file is updated automatically every few minutes and when you close the application.

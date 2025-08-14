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

## Setup and Installation

To get started with KawaiiKuro, follow these steps:

1.  **Python:** Ensure you have Python 3.9 or newer installed.

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    (Or simply download the `kawaii_kuro.py` and `requirements.txt` files into the same directory).

3.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **NLTK Data:** The first time you run the application, it will automatically download necessary data models from the NLTK library. This is a one-time setup process.

## How to Run

To start the application, simply run the main script from your terminal:
```bash
python kawaii_kuro.py
```
This will launch the KawaiiKuro chat window.

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

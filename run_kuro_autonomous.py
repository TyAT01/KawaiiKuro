import argparse
import time
import traceback
from tkinter import messagebox
import tkinter as tk
from collections import Counter
import faulthandler
import signal
from datetime import datetime

# Register a signal handler to dump a traceback on SIGUSR1
faulthandler.register(signal.SIGUSR1)

from kuro.personality import PersonalityEngine
from kuro.knowledge import KnowledgeGraph, GoalManager
from kuro.memory import MemoryManager
from kuro.systems import ReminderManager, SystemAwareness, VoiceIO, BehaviorScheduler
from kuro.utils import MathEvaluator
from kuro.llm_dialogue import LLMDialogueManager
from kuro.persistence import Persistence
from kuro.gui import KawaiiKuroGUI


def handle_crash(e: Exception):
    """Logs the exception and shows a user-friendly error message."""
    print(f"FATAL ERROR: {e}")
    log_message = f"--- CRASH LOG: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
    log_message += traceback.format_exc()
    log_message += "\n--- END OF LOG ---\n"

    with open("crash.log", "a", encoding="utf-8") as f:
        f.write(log_message)

    try:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror(
            "Fatal Error",
            "Oh no, something went terribly wrong... I've saved a crash report to 'crash.log'.\n"
            "I've saved our memories, so please restart me~"
        )
        error_root.destroy()
    except Exception as tk_e:
        print(f"Could not display Tkinter error message box: {tk_e}")


def main_autonomous():
    """Main function to run Kuro in autonomous mode."""
    print("DEBUG: Autonomous Main function started.")
    parser = argparse.ArgumentParser(description="KawaiiKuro (Autonomous) - Your Autonomous Gothic Anime Waifu")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless (command-line) mode.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice I/O.")
    args = parser.parse_args()
    print("DEBUG: Parsed arguments.")

    # Initialize all components
    print("DEBUG: Initializing PersonalityEngine...")
    personality = PersonalityEngine()
    print("DEBUG: Initializing KnowledgeGraph...")
    kg = KnowledgeGraph()
    print("DEBUG: Initializing MemoryManager...")
    memory = MemoryManager()
    print("DEBUG: Initializing ReminderManager...")
    reminders = ReminderManager()
    print("DEBUG: Initializing MathEvaluator...")
    math_eval = MathEvaluator()
    print("DEBUG: Initializing SystemAwareness...")
    system_awareness = SystemAwareness()
    print("DEBUG: Initializing GoalManager...")
    gm = GoalManager(kg, memory, personality)
    print("DEBUG: Initializing LLMDialogueManager...")
    dialogue = LLMDialogueManager(personality, memory, kg)
    print("DEBUG: Initializing VoiceIO...")
    voice = VoiceIO(rate=140, enabled=not args.no_voice)
    print("DEBUG: Initializing Persistence...")
    persistence = Persistence(personality, dialogue, memory, reminders, kg, gm)
    print("DEBUG: Initialized Persistence.")

    # Load persistence
    state = persistence.load()

    # Restore state
    personality.affection_score = int(state.get('affection_score', 0))
    personality.spicy_mode = bool(state.get('spicy_mode', False))
    # ... (omitting the rest of state loading for brevity, it's identical)
    kg.from_dict(state.get('knowledge_graph', {}))
    memory.from_list(state.get('memory', []))
    gm.from_dict(state.get('goal_manager'))


    if not args.no_gui:
        gui = KawaiiKuroGUI(dialogue, personality, voice, gm)
        scheduler = BehaviorScheduler(
            voice=voice, dialogue=dialogue, personality=personality, reminders=reminders,
            system=system_awareness, gui_ref=lambda text: gui.thread_safe_post(text),
            kg=kg, goal_manager=gm, persistence=persistence, math_eval=math_eval,
        )
        scheduler.start()

        def on_key(event):
            scheduler.mark_interaction()
        gui.root.bind_all('<Key>', on_key)

        try:
            print("Kuro is running autonomously with GUI. Close the window to exit.")
            gui.root.mainloop()
        except Exception as e:
            handle_crash(e)
        finally:
            print("Shutting down...")
            persistence.save()
            scheduler.stop()
    else: # Headless mode
        # The gui_ref will just print to console in headless mode.
        def headless_post(text):
            print(f"KawaiiKuro: {text}")

        scheduler = BehaviorScheduler(
            voice=voice, dialogue=dialogue, personality=personality, reminders=reminders,
            system=system_awareness, gui_ref=headless_post,
            kg=kg, goal_manager=gm, persistence=persistence, math_eval=math_eval,
        )
        scheduler.start()

        print("Kuro is running autonomously in headless mode. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown signal received.")
        except Exception as e:
            print(f"\nFATAL ERROR in headless mode: {e}")
            log_message = f"--- CRASH LOG (Headless): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            log_message += traceback.format_exc()
            log_message += "\n--- END OF LOG ---\n"
            with open("crash.log", "a", encoding="utf-8") as f:
                f.write(log_message)
            print("A crash report has been saved to 'crash.log'.")
        finally:
            print("Saving state and shutting down...")
            persistence.save()
            scheduler.stop()


if __name__ == "__main__":
    main_autonomous()

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
# Import the new LLMDialogueManager
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
        # Use a new, clean Tk root for the error message to avoid issues with a corrupted mainloop
        error_root = tk.Tk()
        error_root.withdraw() # Hide the empty root window
        messagebox.showerror(
            "Fatal Error",
            "Oh no, something went terribly wrong... I've saved a crash report to 'crash.log'.\n"
            "I've saved our memories, so please restart me~"
        )
        error_root.destroy()
    except Exception as tk_e:
        print(f"Could not display Tkinter error message box: {tk_e}")


def main_llm():
    print("DEBUG: LLM Main function started.")
    parser = argparse.ArgumentParser(description="KawaiiKuro (LLM) - Your Autonomous Gothic Anime Waifu")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless (command-line) mode.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice I/O to prevent slow startup.")
    parser.add_argument("--input-file", type=str, help="Path to a file containing user inputs for headless mode.")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with shorter delays.")
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

    # --- Use the LLMDialogueManager ---
    print("DEBUG: Initializing LLMDialogueManager...")
    dialogue = LLMDialogueManager(personality, memory, kg)
    print("DEBUG: Initialized LLMDialogueManager.")

    print("DEBUG: Initializing VoiceIO...")
    voice = VoiceIO(rate=140, enabled=not args.no_voice)
    print("DEBUG: Initializing Persistence...")
    # Note: The original persistence for DialogueManager (learned_patterns) won't apply here.
    persistence = Persistence(personality, dialogue, memory, reminders, kg, gm)
    print("DEBUG: Initialized Persistence.")

    # Load persistence
    state = persistence.load()

    # Restore state (same as before, but dialogue patterns are ignored)
    personality.affection_score = int(state.get('affection_score', 0))
    personality.spicy_mode = bool(state.get('spicy_mode', False))
    personality.relationship_status = state.get('relationship_status', 'Strangers')
    personality.rival_mention_count = int(state.get('rival_mention_count', 0))
    personality.rival_names = set(state.get('rival_names', []))
    personality.user_preferences = Counter(state.get('user_preferences', {}))
    personality.learned_topics = state.get('learned_topics', [])
    personality.core_entities = Counter(state.get('core_entities', {}))
    personality.mood_scores = state.get('mood_scores', {'playful': 0, 'jealous': 0, 'scheming': 0, 'thoughtful': 0})
    personality._update_affection_level()
    kg.from_dict(state.get('knowledge_graph', {}))
    memory.from_list(state.get('memory', []))
    memory.summaries = state.get('memory_summaries', [])
    for r in state.get('reminders', []):
        if 'text' in r and 'time' in r:
            reminders.reminders.append(r)
    gm.from_dict(state.get('goal_manager'))

    if not args.no_gui:
        # The GUI needs a slight adaptation if we want it to handle LLM-specific things,
        # but for now, it should work as is.
        gui = KawaiiKuroGUI(dialogue, personality, voice, gm)

        scheduler = BehaviorScheduler(
            voice=voice,
            dialogue=dialogue,
            personality=personality,
            reminders=reminders,
            system=system_awareness,
            gui_ref=lambda text: gui.thread_safe_post(text),
            kg=kg,
            goal_manager=gm,
            persistence=persistence,
            math_eval=math_eval, # Math evaluator is part of the old system, LLM can do it directly.
            test_mode=args.test_mode
        )
        scheduler.start()

        def on_key(event):
            scheduler.mark_interaction()
        gui.root.bind_all('<Key>', on_key)

        try:
            gui.root.mainloop()
        except Exception as e:
            handle_crash(e)
        finally:
            persistence.save()
            scheduler.stop()
    else: # Headless mode
        # The LLM doesn't benefit from the scheduler as much, but we run it for consistency.
        scheduler = BehaviorScheduler(
            voice=voice,
            dialogue=dialogue,
            personality=personality,
            reminders=reminders,
            system=system_awareness,
            gui_ref=lambda text: print(f"\nKawaiiKuro (autonomous): {text}\nYou: ", end=''),
            kg=kg,
            goal_manager=gm,
            persistence=persistence,
            math_eval=math_eval,
            test_mode=bool(args.input_file) or args.test_mode
        )
        scheduler.start()

        print("KawaiiKuro (LLM) is running in headless mode.")

        def process_input(user_input):
            if user_input.lower().strip() == "exit":
                return False
            scheduler.mark_interaction()
            # The rest of the system still revolves around the 'respond' method.
            response = dialogue.respond(user_input)
            print(f"You: {user_input.strip()}")
            print(f"KawaiiKuro: {response}\n")
            # We add to memory here since the LLM manager doesn't do it automatically
            # after a response.
            dialogue.add_memory(user_input.strip(), response)
            return True

        try:
            if args.input_file:
                print(f"Reading inputs from {args.input_file}...")
                with open(args.input_file, 'r') as f:
                    for line in f:
                        if not process_input(line):
                            break
                return
            else:
                print("Type 'exit' to quit.")
                while True:
                    user_input = input("You: ")
                    if not process_input(user_input):
                        break
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
    main_llm()

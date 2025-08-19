import argparse
import time
import traceback
from tkinter import messagebox
import tkinter as tk
from collections import Counter

from kuro.personality import PersonalityEngine
from kuro.knowledge import KnowledgeGraph, GoalManager
from kuro.memory import MemoryManager
from kuro.systems import ReminderManager, SystemAwareness, VoiceIO, BehaviorScheduler
from kuro.utils import MathEvaluator
from kuro.dialogue import DialogueManager
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


def main():
    print("DEBUG: Main function started.")
    parser = argparse.ArgumentParser(description="KawaiiKuro - Your Autonomous Gothic Anime Waifu")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless (command-line) mode.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice I/O to prevent slow startup.")
    parser.add_argument("--input-file", type=str, help="Path to a file containing user inputs for headless mode.")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with shorter delays.")
    parser.add_argument("--autonomous-test", action="store_true", help="Run autonomous schedulers for a short period and exit.")
    args = parser.parse_args()
    print("DEBUG: Parsed arguments.")

    # Initialize all components
    print("DEBUG: Initializing PersonalityEngine...")
    personality = PersonalityEngine()
    print("DEBUG: Initialized PersonalityEngine.")
    print("DEBUG: Initializing KnowledgeGraph...")
    kg = KnowledgeGraph()
    print("DEBUG: Initialized KnowledgeGraph.")
    print("DEBUG: Initializing MemoryManager...")
    memory = MemoryManager()
    print("DEBUG: Initialized MemoryManager.")
    print("DEBUG: Initializing ReminderManager...")
    reminders = ReminderManager()
    print("DEBUG: Initialized ReminderManager.")
    print("DEBUG: Initializing MathEvaluator...")
    math_eval = MathEvaluator()
    print("DEBUG: Initialized MathEvaluator.")
    print("DEBUG: Initializing SystemAwareness...")
    system_awareness = SystemAwareness()
    print("DEBUG: Initialized SystemAwareness.")
    print("DEBUG: Initializing GoalManager...")
    gm = GoalManager(kg, memory, personality)
    print("DEBUG: Initialized GoalManager.")
    print("DEBUG: Initializing DialogueManager...")
    dialogue = DialogueManager(personality, memory, reminders, math_eval, kg)
    print("DEBUG: Initialized DialogueManager.")
    print("DEBUG: Initializing VoiceIO...")
    voice = VoiceIO(rate=140, enabled=not args.no_voice)
    print("DEBUG: Initialized VoiceIO.")
    print("DEBUG: Initializing Persistence...")
    persistence = Persistence(personality, dialogue, memory, reminders, kg, gm)
    print("DEBUG: Initialized Persistence.")

    # Load persistence
    state = persistence.load()

    # Restore state
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
    for k, v in state.get('learned_patterns', {}).items():
        if isinstance(v, list):
            dialogue.learned_patterns[k] = v

    if args.autonomous_test:
        print("Running in autonomous test mode for 30 seconds...")
        scheduler = BehaviorScheduler(
            voice=voice, dialogue=dialogue, personality=personality, reminders=reminders,
            system=system_awareness, gui_ref=lambda text: print(f"\n{text}\n"), kg=kg,
            goal_manager=gm, persistence=persistence, math_eval=math_eval, test_mode=True
        )
        scheduler.start()
        time.sleep(30)
        print("Autonomous test finished. Shutting down.")
        persistence.save()
        scheduler.stop()
        return

    if not args.no_gui:
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
            math_eval=math_eval,
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
    else:
        # In headless mode, post autonomous messages to console
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

        print("KawaiiKuro is running in headless mode.")

        def process_input(user_input):
            if user_input.lower().strip() == "exit":
                return False
            scheduler.mark_interaction()
            response = dialogue.respond(user_input)
            print(f"You: {user_input.strip()}")
            print(f"KawaiiKuro: {response}\n")
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
    main()

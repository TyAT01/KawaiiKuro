import os
import json
from typing import Dict, Any, Optional

from kuro.config import DATA_FILE

# Forward declarations
class PersonalityEngine:
    pass

class DialogueManager:
    pass

class MemoryManager:
    pass

class ReminderManager:
    pass

class KnowledgeGraph:
    pass

class GoalManager:
    pass

class Persistence:
    def __init__(
        self,
        personality: PersonalityEngine,
        dialogue: DialogueManager,
        memory: MemoryManager,
        reminders: ReminderManager,
        knowledge_graph: KnowledgeGraph,
        goal_manager: GoalManager,
    ):
        self.p = personality
        self.dm = dialogue
        self.m = memory
        self.r = reminders
        self.kg = knowledge_graph
        self.gm = goal_manager

    def load(self) -> Dict[str, Any]:
        bak_file = f"{DATA_FILE}.bak"

        def _load_from(file_path: str) -> Optional[Dict[str, Any]]:
            if not os.path.exists(file_path):
                return None
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load data from {file_path}: {e}")
                return None

        # Try loading the main file first
        data = _load_from(DATA_FILE)
        if data is not None:
            return data

        # If main file fails, try loading the backup file
        print("Main data file failed to load, attempting to load from backup...")
        data = _load_from(bak_file)
        if data is not None:
            print("Successfully loaded from backup file for this session. The next successful save will repair the main data file.")
            return data

        # If both fail, return empty
        print("Could not load main data file or backup. Starting with a fresh state.")
        return {}

    def save(self):
        # Enforce a global lock order to prevent deadlocks.
        # This is the single place where all data is gathered for saving.
        with self.p.lock, self.m.lock, self.kg.lock, self.gm.lock, self.dm.lock, self.r.lock:
            data = {
                'affection_score': self.p.affection_score,
                'spicy_mode': self.p.spicy_mode,
                'relationship_status': self.p.relationship_status,
                'rival_mention_count': self.p.rival_mention_count,
                'rival_names': list(self.p.rival_names),
                'user_preferences': dict(self.p.user_preferences),
                'learned_topics': self.p.learned_topics,
                'core_entities': dict(self.p.core_entities),
                'mood_scores': self.p.mood_scores,
                'knowledge_graph': {
                    'entities': self.kg.entities,
                    'relations': self.kg.relations
                },
                'goal_manager': {
                    'active_goal': self.gm.active_goal.__dict__ if self.gm.active_goal else None,
                    'completed_goals': self.gm.completed_goals
                },
                'learned_patterns': self.dm.learned_patterns,
                'memory': [e.__dict__ for e in self.m.entries],
                'memory_summaries': self.m.summaries,
                'reminders': self.r.reminders,
            }

        import tempfile
        bak_file = f"{DATA_FILE}.bak"

        # Create a temporary file in the same directory to ensure atomic rename works
        temp_dir = os.path.dirname(os.path.abspath(DATA_FILE))

        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=temp_dir, delete=False) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file_path = tmp_file.name

            # If the main file exists, atomically move it to the backup location.
            if os.path.exists(DATA_FILE):
                os.replace(DATA_FILE, bak_file)

            # Atomically move the new temporary file to become the main data file.
            os.replace(tmp_file_path, DATA_FILE)

        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"Error during save: {e}. Attempting to restore from backup.")
            try:
                if os.path.exists(bak_file):
                    os.replace(bak_file, DATA_FILE)
            except OSError as e_restore:
                print(f"FATAL: Could not restore backup file: {e_restore}")
        finally:
            # Clean up the temp file if it still exists after a failed operation.
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

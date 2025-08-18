import os
import re
import json
import time
import math
import random
import threading
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
# Optional imports: degrade gracefully if missing
try:
    import pyttsx3  # TTS
except Exception:
    pyttsx3 = None

try:
    import speech_recognition as sr  # STT
except Exception:
    sr = None

try:
    import psutil
except Exception:
    psutil = None

from kuro.config import (
    IDLE_THRESHOLD_SEC, AUTO_BEHAVIOR_PERIOD_SEC, JEALOUSY_CHECK_PERIOD_SEC,
    AUTO_LEARN_PERIOD_SEC, AUTO_SAVE_PERIOD_SEC, AUDIO_TIMEOUT_SEC, AUDIO_PHRASE_LIMIT_SEC,
    KNOWN_PROCESSES, MAX_MEMORY, DREAM_PERIOD_SEC
)
from kuro.utils import safe_word_tokenize, safe_pos_tag, safe_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Forward declarations for type hinting
class VoiceIO:
    pass

class DialogueManager:
    pass

class PersonalityEngine:
    pass

class ReminderManager:
    pass

class SystemAwareness:
    pass

class KnowledgeGraph:
    pass

class GoalManager:
    pass

class MemoryManager:
    pass

class Persistence:
    pass

class MathEvaluator:
    pass

# -----------------------------
# System Awareness
# -----------------------------
class SystemAwareness:
    def __init__(self):
        self.last_battery_warning_time = 0

    def get_time_of_day_greeting(self) -> Optional[str]:
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning, my love~ Did you sleep well?"
        if 12 <= hour < 17:
            return "Good afternoon~ Hope you're having a great day."
        if 17 <= hour < 22:
            return "Good evening~ Time to relax with me."
        if 22 <= hour or hour < 5:
            return "It's getting late~ You should be resting, but I'm glad you're with me."
        return None

    def get_battery_status(self) -> Optional[str]:
        if not psutil:
            return None

        now = time.time()
        # Avoid spamming battery warnings
        if now - self.last_battery_warning_time < 3600: # 1 hour
            return None

        try:
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent < 25:
                self.last_battery_warning_time = now
                return f"Your battery is at {battery.percent}%! Don't forget to plug in, my love~"
        except Exception:
            # sensors_battery() can fail on desktops
            return None
        return None

# -----------------------------
# Reminders & Scheduling
# -----------------------------
class ReminderManager:
    def __init__(self):
        self.reminders: List[Dict[str, str]] = []
        self.lock = threading.Lock()

    def add(self, text: str, when: datetime):
        with self.lock:
            self.reminders.append({"text": text, "time": when.strftime('%Y-%m-%d %H:%M:%S')})

    def due(self) -> List[Dict[str, str]]:
        now = datetime.now()
        with self.lock:
            ready = [r for r in self.reminders if datetime.strptime(r['time'], '%Y-%m-%d %H:%M:%S') <= now]
            self.reminders = [r for r in self.reminders if r not in ready]
        return ready

    def list_active(self) -> str:
        with self.lock:
            active = [r for r in self.reminders if datetime.strptime(r['time'], '%Y-%m-%d %H:%M:%S') > datetime.now()]
        if not active:
            return "No active reminders~ Set one, my love?"
        return "\n".join([f"Reminder ({r['time']}): {r['text']}" for r in active])

# -----------------------------
# Voice I/O (optional)
# -----------------------------
class VoiceIO:
    def __init__(self, rate: int = 140, enabled: bool = True):
        self.tts = None
        if enabled and pyttsx3 is not None:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', rate)
                # attempt to pick a female voice
                voice_id = None
                for v in self.tts.getProperty('voices'):
                    name = (getattr(v, 'name', '') or '').lower()
                    if 'female' in name or 'zira' in name:
                        voice_id = v.id
                        break
                if voice_id:
                    self.tts.setProperty('voice', voice_id)
            except Exception:
                self.tts = None
        self.recognizer = sr.Recognizer() if enabled and sr is not None else None

    def speak(self, text: str):
        if not self.tts:
            return
        clean = re.sub(r'\*.*?\*', '', text)
        try:
            self.tts.say(clean)
            self.tts.runAndWait()
        except Exception:
            pass

    def listen(self, wake_word: str = "kawaiikuro") -> str:
        if not self.recognizer or sr is None:
            return ""
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=AUDIO_TIMEOUT_SEC, phrase_time_limit=AUDIO_PHRASE_LIMIT_SEC)
            try:
                text = self.recognizer.recognize_sphinx(audio)
            except Exception:
                text = ""
            text = text.lower()
            if wake_word in text:
                return text.replace(wake_word, "").strip()
            return ""
        except Exception:
            return ""

# -----------------------------
# Behavior Scheduler (threads)
# -----------------------------
class BehaviorScheduler:
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref, kg: KnowledgeGraph, goal_manager: GoalManager, persistence: 'Persistence', math_eval: 'MathEvaluator', test_mode: bool = False):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gm = goal_manager
        self.persistence = persistence
        self.math_eval = math_eval
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()
        self.lock = threading.Lock()
        self.test_mode = test_mode
        self.auto_behavior_period = 1 if test_mode else AUTO_BEHAVIOR_PERIOD_SEC
        self.idle_threshold = 10 if test_mode else IDLE_THRESHOLD_SEC

    def _log_error(self, loop_name: str):
        """Logs an exception from a scheduler loop."""
        log_message = f"--- SCHEDULER ERROR LOG: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in {loop_name} ---\n"
        log_message += traceback.format_exc()
        log_message += "\n--- END OF LOG ---\n"

        with open("behavior_crash.log", "a", encoding="utf-8") as f:
            f.write(log_message)
        # Non-intrusive console log for developers
        print(f"An error occurred in the {loop_name} loop. See behavior_crash.log for details.")

    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._auto_learn_loop, daemon=True).start()
        threading.Thread(target=self._auto_save_loop, daemon=True).start()
        threading.Thread(target=self._mood_update_loop, daemon=True).start()
        threading.Thread(target=self._goal_loop, daemon=True).start()
        threading.Thread(target=self._dream_loop, daemon=True).start()
        threading.Thread(target=self._system_awareness_loop, daemon=True).start() # Disabled for testing, as psutil can hang
        if self.voice and self.voice.recognizer is not None:
            threading.Thread(target=self._continuous_listen_loop, daemon=True).start()

    def stop(self):
        self.stop_flag.set()

    def _reflect_on_memory(self) -> Optional[str]:
        with self.dm.m.lock:
            if not self.dm.m.summaries:
                return None
            summary = random.choice(self.dm.m.summaries)

        # Extract a meaningful keyword from the summary
        tokens = [word.lower() for word in summary.split() if len(word) > 3]
        stop_words = ['user', 'talked', 'about', 'their', 'interest', 'mentioned', 'asked']
        keywords = [t for t in tokens if t not in stop_words]

        if not keywords:
            return None

        keyword = random.choice(keywords)
        return f"I was just thinking about how we talked about {keyword}... It was a nice memory~"

    def _reflect_on_knowledge(self) -> Optional[str]:
        with self.kg.lock:
            # Find entities with relationships
            entities_with_rels = [e for e, data in self.kg.entities.items() if self.kg.get_relations(e)]
            if not entities_with_rels:
                return None

            entity_name = random.choice(entities_with_rels)
            relations = self.kg.get_relations(entity_name)
            relation = random.choice(relations)

            # Determine the other entity in the relation
            if relation['source'] == entity_name:
                related_entity_name = relation['target']
            else:
                related_entity_name = relation['source']

            return f"My thoughts drifted for a moment... I was remembering that {entity_name} has a connection to {related_entity_name}. It's fascinating how everything you tell me is linked."

    def _reflect_on_goals(self) -> Optional[str]:
        with self.gm.lock:
            if not self.gm._potential_goals: # Check potential goals as long_term_goals might not be populated
                return None

        # This is more of an internal thought process. It will select a new short-term goal.
        # The output is just a reflection of that thought process.
        self.gm.select_new_goal(self.p.get_dominant_mood(), "Reflecting on our long term goals.")
        return "I was just pondering our future... what we're working towards. It makes me feel... hopeful."

    def _daydream_about_user(self) -> Optional[str]:
        """Generates a comment based on stored user preferences."""
        with self.p.lock:
            if not self.p.user_preferences:
                return None
            # Get the top 3 preferences
            top_prefs = self.p.user_preferences.most_common(3)
            if not top_prefs:
                return None

            chosen_pref, count = random.choice(top_prefs)

            templates = [
                f"You really like {chosen_pref}, don't you? It's cute~",
                f"I was just thinking about {chosen_pref}... maybe we can talk about it more later?",
                f"Hehe, you've mentioned {chosen_pref} a few times. I've made a special note of it~"
            ]
            return random.choice(templates)

    def _explore_knowledge_graph(self) -> Optional[str]:
        """Traverses the knowledge graph to find an interesting connection."""
        with self.kg.lock:
            # Find entities with at least one relationship
            entities = [e for e in self.kg.entities if self.kg.get_relations(e)]
            if len(entities) < 2:
                return None

            # Try a few times to find a multi-step path
            for _ in range(5):
                start_node = random.choice(entities)
                path = self.kg.find_path(start_node, max_depth=3)
                if path and len(path) > 2: # We want a path of at least Start -> Middle -> End
                    start, end = path[0], path[-1]
                    middle = path[1]
                    return f"*her eyes glaze over for a second* ...that's funny. I just realized that {start} connects to {middle}, which connects to {end}. My own little web of thoughts~"
        return None # Return None if no interesting path was found

    def _learn_from_local_file(self) -> Optional[str]:
        """Reads a random .txt file from a special directory and learns from it."""
        learning_dir = "kuro_learning_material"
        if not os.path.exists(learning_dir) or not os.path.isdir(learning_dir):
            return None

        files = [f for f in os.listdir(learning_dir) if f.endswith(".txt")]
        if not files:
            return None

        try:
            chosen_file = random.choice(files)
            with open(os.path.join(learning_dir, chosen_file), 'r', encoding='utf-8') as f:
                content = f.read()

            if len(content) < 50: # Skip very short files
                return None

            new_relations = self.kg.infer_new_relations(content)
            if not new_relations:
                return f"*reads a document titled '{chosen_file}' but finds it... uninteresting.*"

            # Add new relations to the knowledge graph
            added_count = 0
            for rel in new_relations:
                # Check for duplicates before adding
                exists = False
                for r in self.kg.relations:
                    if r['source'] == rel['subject'] and r['relation'] == rel['verb'] and r['target'] == rel['object']:
                        exists = True
                        break
                if not exists:
                    self.kg.add_relation(rel['subject'], rel['verb'], rel['object'], confidence=rel['confidence'], source=f"file:{chosen_file}")
                    added_count += 1

            if added_count == 0:
                return f"*skims through '{chosen_file}'... It seems I already knew everything in it.*"

            # Find a topic from the new relations
            topic = random.choice(new_relations)['subject']
            return f"*is reading a document titled '{chosen_file}'...* I think I'm learning something new about {topic}."

        except Exception as e:
            self._log_error(f"_learn_from_local_file reading {chosen_file}")
            return None

    def _generate_creative_text(self) -> Optional[str]:
        """Generates a short, haiku-like poem about entities in the knowledge graph."""
        with self.kg.lock:
            # Get entities that are not too generic
            interesting_entities = [e for e, data in self.kg.entities.items() if e not in ['user', 'kawaiikuro', 'i'] and len(e.split()) <= 2]

            if len(interesting_entities) < 3:
                return None

            # Select three distinct entities for the poem
            try:
                e1, e2, e3 = random.sample(interesting_entities, 3)
            except ValueError:
                return None

            # Simple haiku-like template (approximating 5-7-5 syllables with word count)
            line1 = f"A thought of {e1},"
            line2 = f"{e2} and {e3} appear,"
            line3 = "A new world unfolds."

            poem = f"{line1}\n{line2}\n{line3}"
            return f"*a strange thought crosses my mind...*\n\n{poem}"

    def _practice_math(self) -> Optional[str]:
        """Generates a simple math problem for self-entertainment."""
        try:
            # Generate a simple problem
            num1 = random.randint(2, 100)
            num2 = random.randint(2, 100)
            operator = random.choice(['+', '-', '*', '/'])

            if operator == '/':
                # Ensure a clean division to seem smarter
                num2 = random.randint(2, 20)
                num1 = num1 * num2

            expr = f"{num1} {operator} {num2}"

            result_str = self.math_eval.eval(expr)

            if "Math error" in result_str:
                return None

            responses = [
                f"Sometimes I do a little math for fun... just to keep my mind sharp. Like this one: {result_str}",
                f"Hehe, I'm such a nerd sometimes. I was just calculating this: {result_str}",
                f"My brain just wandered and solved this little problem for me: {result_str}"
            ]
            return random.choice(responses)
        except Exception:
            return None

    def _perform_long_idle_activity(self) -> Optional[str]:
        """Chooses and performs a random reflection or self-entertainment activity."""
        possible_actions = []

        # Check which reflections are possible
        if self.dm.m.summaries:
            possible_actions.append(self._reflect_on_memory)
        if any(self.kg.get_relations(e) for e in self.kg.entities):
            possible_actions.append(self._reflect_on_knowledge)
        if self.gm._potential_goals:
            possible_actions.append(self._reflect_on_goals)
        if self.p.user_preferences:
            possible_actions.append(self._daydream_about_user)
        if len([e for e in self.kg.entities if self.kg.get_relations(e)]) > 1:
            possible_actions.append(self._explore_knowledge_graph)

        # Add the new file learning action
        possible_actions.append(self._learn_from_local_file)
        # Add the new creative action
        possible_actions.append(self._generate_creative_text)
        # Add the new math practice action
        possible_actions.append(self._practice_math)

        if not possible_actions:
            return None

        # Choose an action and execute it
        action_to_perform = random.choice(possible_actions)
        return action_to_perform()

    def _dream_loop(self):
        # Wait a bit before the first dream to let the app settle
        time.sleep(45)
        while not self.stop_flag.is_set():
            dream_period = 30 if self.test_mode else DREAM_PERIOD_SEC
            time.sleep(dream_period)
            try:
                # Only dream if the user has been idle for a long time
                long_idle_threshold = self.idle_threshold * 3
                if time.time() - self.last_interaction_time > long_idle_threshold:
                    activity_message = self._perform_long_idle_activity()

                    if activity_message:
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(activity_message)}", speak=False)
                    else:
                        # Fallback if no reflection was possible
                        self._post_gui("KawaiiKuro: *is lost in thought, her gaze distant...*", speak=False)

                    # We mark interaction to prevent this from firing repeatedly
                    # until the next dream period.
                    self.mark_interaction()
            except Exception:
                self._log_error("_dream_loop")

    def _post_gui(self, text: str, speak: bool = True):
        if self.gui_ref:
            self.gui_ref(text)
        if speak and self.voice:
            self.voice.speak(text)

    def _reminder_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(1)
            try:
                for r in self.r.due():
                    msg = f"Reminder! {r['text']} *jumps excitedly*"
                    self._post_gui(f"KawaiiKuro: {msg}")
            except Exception:
                self._log_error("_reminder_loop")

    def _get_random_idle_comment(self) -> str:
        """Returns a random, simple idle comment."""
        comments = [
            "Hmph... are you busy?",
            "*taps her fingers on the desk, waiting*",
            "*looks out a virtual window, sighing softly*",
            "I wonder what you're thinking about right now...",
            "Hope you're not working too hard, my love.",
            "*doodles a little heart in the corner of the screen*",
            "My thoughts are drifting... mostly to you.",
            "Is it time to play yet?~",
            "*puffs her cheeks out, looking a little bored*",
            "Come back soon, okay?"
        ]
        return random.choice(comments)

    def _idle_loop(self):
        time_greeting_posted = False
        while not self.stop_flag.is_set():
            time.sleep(self.auto_behavior_period)
            try:
                if time.time() - self.last_interaction_time > self.idle_threshold:
                    # Reset the greeting check if the user is idle again after a while
                    if time.time() - self.last_interaction_time > self.idle_threshold * 5:
                        time_greeting_posted = False

                    if not time_greeting_posted:
                        time_greeting = self.system.get_time_of_day_greeting()
                        if time_greeting:
                            self._post_gui(f"KawaiiKuro: {time_greeting}")
                            time_greeting_posted = True
                            self.mark_interaction() # Mark as interaction so we don't spam other idle messages
                            continue

                    # Try to predict a task first
                    message = self.dm.predict_task()
                    if message:
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                    else:
                        # Otherwise, use a random idle comment
                        idle_comment = self._get_random_idle_comment()
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(idle_comment)}")

                    # Only penalize affection slightly, and less often.
                    if random.random() < 0.25:
                        self.p.affection_score = max(-10, self.p.affection_score - 1)
                        self.p._update_affection_level()

                    self.mark_interaction() # reset idle timer
            except Exception:
                self._log_error("_idle_loop")

    def _goal_loop(self):
        time.sleep(20)  # Initial delay to let things settle
        while not self.stop_flag.is_set():
            # Check goals periodically
            time.sleep(self.auto_behavior_period * 2)
            try:
                # The GoalManager logic has been disabled due to an unresolvable hang
                # in the execution environment. All other features, including the dream/reflection
                # state, remain active.
                pass
                # with self.dm.m.lock, self.kg.lock, self.gm.lock:
                #     # 1. Select a new goal if there isn't one
                #     if not self.gm.active_goal:
                #         last_user_input = ""
                #         if self.dm.m.entries:
                #             last_user_input = self.dm.m.entries[-1].user
                #         current_mood = self.p.get_dominant_mood()
                #         self.gm.select_new_goal(current_mood, last_user_input)

                #     # 2. Process the active goal
                #     if self.gm.active_goal:
                #         # Only process if user is idle, to avoid being annoying
                #         if time.time() - self.last_interaction_time > self.idle_threshold / 3:
                #             message = self.gm.process_active_goal()
                #             if message:
                #                 # A silent update is just for internal state
                #                 if "*takes a quiet, thoughtful note" in message:
                #                     self._post_gui(f"KawaiiKuro: {message}", speak=False)
                #                 else:  # It's a real question or a result
                #                     self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                #                     self.mark_interaction() # It's a significant interaction
            except Exception:
                self._log_error("_goal_loop")


    def _system_awareness_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(JEALOUSY_CHECK_PERIOD_SEC)
            try:
                if not psutil:
                    continue
                try:
                    running_processes = {p.name().lower() for p in psutil.process_iter(['name'])}
                    with self.lock:
                        for category, (procs, comment) in KNOWN_PROCESSES.items():
                            if category not in self.already_commented_on_process:
                                for proc_name in procs:
                                    if proc_name in running_processes:
                                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(comment)}")
                                        self.already_commented_on_process.add(category)
                                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass # ignore transient errors
            except Exception:
                self._log_error("_system_awareness_loop")

    def _mood_update_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(10 if self.test_mode else 450)
            try:
                self.p.update_mood()
            except Exception:
                self._log_error("_mood_update_loop")

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(1 if self.test_mode else AUTO_LEARN_PERIOD_SEC)
            try:
                with self.p.lock, self.dm.m.lock:
                    if len(self.dm.m.entries) == MAX_MEMORY:
                        summary = self.dm.m.summarize_and_prune(n_entries=50)
                        if summary:
                            self._post_gui("KawaiiKuro: *spends a moment organizing her memories of us, smiling softly*", speak=False)

                    all_user_text = [entry.user for entry in self.dm.m.entries if len(entry.user.split()) > 3]
                    if len(all_user_text) < 15: # Need more data for n-grams
                        continue

                    # --- Noun/Entity Extraction (for personality) ---
                    all_user_text_single_str = " ".join(all_user_text)
                    tokens = safe_word_tokenize(all_user_text_single_str.lower())
                    tagged = safe_pos_tag(tokens)

                    base_stop_words = safe_stopwords()
                    user_entity = self.dm.kg.get_entity('user')
                    if user_entity and user_entity.get('attributes',{}).get('name'):
                        base_stop_words.add(user_entity['attributes']['name'].get('value','').lower())

                    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in base_stop_words]
                    self.p.core_entities.update(nouns)
                    if len(self.p.core_entities) > 20:
                        self.p.core_entities = Counter(dict(self.p.core_entities.most_common(20)))

                    # --- Topic Modeling (for learning) ---
                    try:
                        # Add more conversational stop words for better topic modeling
                        custom_stop_words = {'like', 'know', 'think', 'really', 'just', 'going', 'say', 'feel', 'mean', 'want', 'got', 'don', 'doesn'}
                        vectorizer_stop_words = base_stop_words.union(custom_stop_words)

                        # Use n-grams to capture phrases like "ice cream"
                        vectorizer = CountVectorizer(max_df=0.85, min_df=3, stop_words=list(vectorizer_stop_words), ngram_range=(1, 2), max_features=1000)
                        tf = vectorizer.fit_transform(all_user_text)

                        feature_names = vectorizer.get_feature_names_out()
                        n_topics = min(5, len(all_user_text) // 10) # Adjust topic calculation
                        if n_topics < 2:
                            continue # Not enough data for meaningful topics

                        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=15, learning_method='online', learning_offset=50., random_state=0)
                        lda.fit(tf)

                        new_topics = []
                        n_top_words = 7 # Get more words per topic
                        for topic_idx, topic_dist in enumerate(lda.components_):
                            top_words_indices = topic_dist.argsort()[:-n_top_words - 1:-1]
                            topic_words = [feature_names[i] for i in top_words_indices]
                            new_topics.append(topic_words)

                        # Only update if the new topics are substantially different
                        if str(new_topics) != str(self.p.learned_topics):
                            self.p.learned_topics = new_topics
                            self._post_gui("KawaiiKuro: *takes some nerdy notes on our conversations* I feel like I understand you better now~", speak=False)

                            # --- Proactive Question based on new topic ---
                            if random.random() < 0.3: # 30% chance to ask a question
                                time.sleep(2) # Small delay to not feel instant
                                chosen_topic = random.choice(new_topics)
                                # Choose a keyword from the topic that is a single word, if possible
                                single_word_keywords = [word for word in chosen_topic if ' ' not in word]
                                if not single_word_keywords:
                                    continue # Should be rare, but skip if no single words
                                keyword = random.choice(single_word_keywords)

                                question_templates = [
                                    f"I noticed we've been talking about {keyword} a bit... What are your thoughts on it? I'm curious~",
                                    f"You mentioned {keyword} earlier, and it got me thinking... could you tell me more?",
                                    f"My thoughts keep drifting back to {keyword}. It sounds really interesting. What's on your mind about it?"
                                ]
                                question = self.dm.personalize_response(random.choice(question_templates))
                                self._post_gui(f"KawaiiKuro: {question}")
                                self.mark_interaction()

                    except ValueError:
                        # This can happen if the vocabulary is empty after filtering, etc.
                        # It's not a critical error, just skip this learning cycle.
                        pass
                    except Exception:
                        self._log_error("_auto_learn_loop (LDA part)")
                        pass

                    # Consolidate knowledge graph
                    newly_inferred = self.kg.consolidate_knowledge()
                    if newly_inferred:
                        self._post_gui(f"KawaiiKuro: *has a moment of insight, connecting some dots...*", speak=False)
            except Exception:
                self._log_error("_auto_learn_loop")

    def _auto_save_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(15 if self.test_mode else AUTO_SAVE_PERIOD_SEC)
            try:
                self.persistence.save()
            except Exception:
                self._log_error("_auto_save_loop")

    def _continuous_listen_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(1)
            try:
                heard = self.voice.listen()
                if heard:
                    # simulate user typing and sending
                    reply = self.dm.respond(heard)
                    self.mark_interaction()
                    self._post_gui(f"You (voice): {heard}\nKawaiiKuro: {reply}")
            except Exception:
                self._log_error("_continuous_listen_loop")

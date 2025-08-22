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
    import psutil
except Exception:
    psutil = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import mss
except ImportError:
    mss = None

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    pytesseract = None


from kuro.config import (
    IDLE_THRESHOLD_SEC, AUTO_BEHAVIOR_PERIOD_SEC, JEALOUSY_CHECK_PERIOD_SEC,
    AUTO_LEARN_PERIOD_SEC, AUTO_SAVE_PERIOD_SEC, AUDIO_TIMEOUT_SEC, AUDIO_PHRASE_LIMIT_SEC,
    KNOWN_PROCESSES, MAX_MEMORY, DREAM_PERIOD_SEC, AUTONOMOUS_THOUGHT_PERIOD_SEC
)
from kuro.utils import safe_word_tokenize, safe_pos_tag, safe_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from kuro.planner import Planner

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

class Planner:
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

    def get_screen_content(self) -> Optional[str]:
        if not all([Image, mss, pytesseract]):
            return None

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            text = pytesseract.image_to_string(img)
            # Basic filtering to find "interesting" text
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
            if not lines:
                return None

            # Very simple analysis: just pick a random "interesting" line
            interesting_line = random.choice(lines)
            return f"I was just looking at your screen... I saw something about '{interesting_line[:50]}...'. What are you working on?~"

        except FileNotFoundError:
            # Tesseract is not installed or not in PATH
            # This should be logged or handled more gracefully
            return "I tried to see what you're doing, but my vision is blurry... Is Tesseract installed?"
        except Exception as e:
            # Other errors (e.g., no screen, permission issues)
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
        self.recognizer = None
        self.pyttsx3 = None
        self.sr = None

        if enabled:
            try:
                self.pyttsx3 = __import__('pyttsx3')
                self.tts = self.pyttsx3.init()
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
                print("VoiceIO: pyttsx3 initialization failed, TTS will be disabled.")
                self.tts = None

            try:
                self.sr = __import__('speech_recognition')
                self.recognizer = self.sr.Recognizer()
            except Exception:
                print("VoiceIO: SpeechRecognition not found, STT will be disabled.")
                self.recognizer = None

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
        if not self.recognizer or not self.sr:
            return ""
        try:
            with self.sr.Microphone() as source:
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
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref, kg: KnowledgeGraph, goal_manager: GoalManager, persistence: 'Persistence', math_eval: 'MathEvaluator', planner: 'Planner', test_mode: bool = False):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gm = goal_manager
        self.persistence = persistence
        self.math_eval = math_eval
        self.planner = planner
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.last_autonomous_action_time = 0 # NEW: Tracks Kuro's own actions
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

    def mark_autonomous_action(self):
        """NEW: Updates the timestamp for Kuro's own actions."""
        self.last_autonomous_action_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._auto_learn_loop, daemon=True).start()
        threading.Thread(target=self._auto_save_loop, daemon=True).start()
        threading.Thread(target=self._mood_update_loop, daemon=True).start()
        threading.Thread(target=self._goal_loop, daemon=True).start()
        threading.Thread(target=self._dream_loop, daemon=True).start()
        threading.Thread(target=self._system_awareness_loop, daemon=True).start()
        threading.Thread(target=self._screen_awareness_loop, daemon=True).start()
        if self.voice and self.voice.recognizer is not None:
            threading.Thread(target=self._continuous_listen_loop, daemon=True).start()

    def stop(self):
        print("Scheduler: Stop flag set.")
        self.stop_flag.set()

    def _dream_loop(self):
        """
        The 'dream loop' is now the core strategic planning loop for Kuro.
        It runs periodically when the user is idle, allowing Kuro to reflect,
        set long-term goals, and create plans to achieve them.
        """
        # Wait a bit before the first planning session to let the app settle
        time.sleep(45)
        while not self.stop_flag.is_set():
            dream_period = 30 if self.test_mode else DREAM_PERIOD_SEC
            time.sleep(dream_period)
            try:
                # Only plan if the user has been idle for a long time
                long_idle_threshold = self.idle_threshold * 3
                if time.time() - self.last_interaction_time > long_idle_threshold:

                    # This is Kuro's strategic thinking time.
                    # We acquire the planner's lock to ensure thread safety.
                    with self.planner.lock:
                        if not self.planner.has_active_goal():
                            # If there's no goal, Kuro's #1 priority is to create one.
                            self._post_gui("KawaiiKuro: *is lost in thought, a faint smirk on her lips as she schemes...*", speak=False)
                            new_goal = self.planner.generate_new_goal()
                            if new_goal:
                                self.planner.active_goal = new_goal
                                # A new goal is a significant event, trigger a thought.
                                self.trigger_autonomous_thought()
                                self.mark_autonomous_action()

                        elif not self.planner.has_plan():
                            # If there's a goal but no plan, time to make a plan.
                            self._post_gui(f"KawaiiKuro: *taps her chin thoughtfully, figuring out how to achieve her latest desire...*", speak=False)
                            plan = self.planner.generate_plan_for_goal(self.planner.active_goal)
                            if plan:
                                self.planner.active_goal.plan = plan
                                # Making a plan is also a significant event.
                                self.trigger_autonomous_thought()
                                self.mark_autonomous_action()

            except Exception:
                self._log_error("_dream_loop")
        print("Scheduler: _planning_loop stopped.")

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
        print("Scheduler: _reminder_loop stopped.")

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
                            self.mark_autonomous_action() # CHANGED
                            continue

                    # Post a random idle comment. The LLM thought is triggered separately.
                    idle_comment = self._get_random_idle_comment()
                    self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(idle_comment)}")

                    # Only penalize affection slightly, and less often.
                    if random.random() < 0.25:
                        with self.p.lock:
                            self.p.affection_score = max(-10, self.p.affection_score - 1)
                            self.p._update_affection_level()

                    # This call ensures that even in long periods of inactivity, Kuro will
                    # periodically attempt to have an autonomous thought, replacing the old loop.
                    self.trigger_autonomous_thought()

                    self.mark_autonomous_action() # CHANGED
            except Exception:
                self._log_error("_idle_loop")
        print("Scheduler: _idle_loop stopped.")

    def _goal_loop(self):
        """
        This loop is responsible for executing the current step of a plan.
        It interfaces between the high-level Planner and the tactical GoalManager.
        """
        time.sleep(20)  # Initial delay to let things settle
        while not self.stop_flag.is_set():
            time.sleep(self.auto_behavior_period * 2)
            try:
                # This loop should not run if the user is actively interacting.
                if time.time() - self.last_interaction_time < self.idle_threshold:
                    continue

                # Use a combined lock to ensure thread safety across all managers
                with self.planner.lock, self.gm.lock:
                    if not self.planner.has_active_goal() or not self.planner.has_plan():
                        continue

                    # Find the current step to execute
                    active_step = next((step for step in self.planner.active_goal.plan if step.status == 'active'), None)

                    if not active_step:
                        # If no active step, find the next pending one and activate it.
                        pending_step = next((step for step in self.planner.active_goal.plan if step.status == 'pending'), None)
                        if pending_step:
                            active_step = pending_step
                            active_step.status = 'active'
                        else:
                            # No pending steps left, the goal is complete!
                            # This logic will be expanded later.
                            self.planner.active_goal.status = 'complete'
                            continue

                    # Now, process the active step with the GoalManager
                    question = self.gm.process_plan_step(active_step.description)

                    if question:
                        # The GoalManager needs more information. Ask the user.
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(question)}")
                        self.mark_autonomous_action()
                    else:
                        # The prerequisites for the step are met. We can mark it as complete.
                        print(f"DEBUG: Plan step '{active_step.description}' completed.")
                        active_step.status = 'complete'
                        # A small, silent acknowledgement of progress.
                        self._post_gui("KawaiiKuro: *nods to herself, her plan proceeding perfectly...*", speak=False)
                        self.mark_autonomous_action()

            except Exception:
                self._log_error("_goal_loop")
        print("Scheduler: _goal_loop stopped.")


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
                                        # Noticing a new process is a good trigger for a thought.
                                        self.trigger_autonomous_thought()
                                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass # ignore transient errors
            except Exception:
                self._log_error("_system_awareness_loop")
        print("Scheduler: _system_awareness_loop stopped.")

    def _screen_awareness_loop(self):
        time.sleep(20) # Initial delay
        while not self.stop_flag.is_set():
            time.sleep(30) # Check every 30 seconds
            try:
                # Only run if the user is idle
                if time.time() - self.last_interaction_time > self.idle_threshold * 2:
                    comment = self.system.get_screen_content()
                    if comment:
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(comment)}")
            except Exception:
                self._log_error("_screen_awareness_loop")
        print("Scheduler: _screen_awareness_loop stopped.")

    def _mood_update_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(10 if self.test_mode else 450)
            try:
                mood_changed = self.p.update_mood()
                if mood_changed:
                    # A change in mood is a perfect time for an autonomous thought.
                    self.trigger_autonomous_thought()
            except Exception:
                self._log_error("_mood_update_loop")
        print("Scheduler: _mood_update_loop stopped.")

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(1 if self.test_mode else AUTO_LEARN_PERIOD_SEC)
            try:
                with self.p.lock, self.dm.m.lock, self.kg.lock:
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
                            # Learning something new is a great trigger for a thought
                            self.trigger_autonomous_thought()

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
                        # A moment of insight should trigger a new thought
                        self.trigger_autonomous_thought()
            except Exception:
                self._log_error("_auto_learn_loop")
        print("Scheduler: _auto_learn_loop stopped.")

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

    def trigger_autonomous_thought(self, force: bool = False):
        """
        Generates a proactive, autonomous thought, with a cooldown to prevent spamming.
        This is now event-driven instead of being in its own loop.
        """
        now = time.time()
        # Cooldown period to prevent thoughts from happening too close together.
        cooldown = 30 if self.test_mode else AUTONOMOUS_THOUGHT_PERIOD_SEC
        # CHANGED: Check against the last *autonomous action*, not just the last thought.
        if not force and (now - self.last_autonomous_action_time < cooldown):
            return

        # Only think if the user is idle and we're using an LLM
        if (now - self.last_interaction_time > self.idle_threshold) and hasattr(self.dm, 'generate_autonomous_thought'):
            try:
                thought = self.dm.generate_autonomous_thought()
                if thought:
                    self._post_gui(f"KawaiiKuro: {thought}")
                    self.mark_autonomous_action() # CHANGED: Mark as an autonomous action, not a user interaction.
            except Exception:
                self._log_error("trigger_autonomous_thought")

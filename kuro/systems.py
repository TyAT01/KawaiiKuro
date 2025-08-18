import os
import re
import json
import time
import math
import random
import threading
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
    KNOWN_PROCESSES, MAX_MEMORY
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
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref, kg: KnowledgeGraph, goal_manager: GoalManager, persistence: 'Persistence', test_mode: bool = False):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gm = goal_manager
        self.persistence = persistence
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()
        self.lock = threading.Lock()
        self.test_mode = test_mode
        self.auto_behavior_period = 1 if test_mode else AUTO_BEHAVIOR_PERIOD_SEC
        self.idle_threshold = 10 if test_mode else IDLE_THRESHOLD_SEC


    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._auto_learn_loop, daemon=True).start()
        threading.Thread(target=self._auto_save_loop, daemon=True).start()
        threading.Thread(target=self._mood_update_loop, daemon=True).start()
        threading.Thread(target=self._goal_loop, daemon=True).start()
        # threading.Thread(target=self._system_awareness_loop, daemon=True).start() # Disabled for testing, as psutil can hang
        if self.voice and self.voice.recognizer is not None:
            threading.Thread(target=self._continuous_listen_loop, daemon=True).start()

    def stop(self):
        self.stop_flag.set()

    def _post_gui(self, text: str, speak: bool = True):
        if self.gui_ref:
            self.gui_ref(text)
        if speak and self.voice:
            self.voice.speak(text)

    def _reminder_loop(self):
        while not self.stop_flag.is_set():
            for r in self.r.due():
                msg = f"Reminder! {r['text']} *jumps excitedly*"
                self._post_gui(f"KawaiiKuro: {msg}")
            time.sleep(1)

    def _idle_loop(self):
        time_greeting_posted = False
        while not self.stop_flag.is_set():
            if time.time() - self.last_interaction_time > self.idle_threshold:
                if not time_greeting_posted:
                    time_greeting = self.system.get_time_of_day_greeting()
                    if time_greeting:
                        self._post_gui(f"KawaiiKuro: {time_greeting}")
                        time_greeting_posted = True
                message = self.dm.predict_task()
                if message:
                    self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                else:
                    self._post_gui(f"KawaiiKuro: Miss you, darling~ *pouts* Come back?")
                self.p.affection_score = max(-10, self.p.affection_score - 1)
                self.p._update_affection_level()
                self.mark_interaction() # reset idle timer
            time.sleep(self.auto_behavior_period)

    def _goal_loop(self):
        time.sleep(20)  # Initial delay to let things settle
        while not self.stop_flag.is_set():
            with self.gm.lock:
                # 1. Select a new goal if there isn't one
                if not self.gm.active_goal:
                    last_user_input = ""
                    with self.dm.m.lock:
                        if self.dm.m.entries:
                            last_user_input = self.dm.m.entries[-1].user
                    current_mood = self.p.get_dominant_mood()
                    self.gm.select_new_goal(current_mood, last_user_input)

                # 2. Process the active goal
                if self.gm.active_goal:
                    # Only process if user is idle, to avoid being annoying
                    if time.time() - self.last_interaction_time > self.idle_threshold / 3:
                        message = self.gm.process_active_goal()
                        if message:
                            # A silent update is just for internal state
                            if "*takes a quiet, thoughtful note" in message:
                                self._post_gui(f"KawaiiKuro: {message}", speak=False)
                            else:  # It's a real question or a result
                                self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                                self.mark_interaction() # It's a significant interaction

            # Check goals periodically
            time.sleep(self.auto_behavior_period * 2)  # Check less frequently than idle loop


    def _system_awareness_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(JEALOUSY_CHECK_PERIOD_SEC)
            if not psutil: continue
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

    def _mood_update_loop(self):
        while not self.stop_flag.is_set():
            self.p.update_mood()
            time.sleep(10 if self.test_mode else 450)

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(1 if self.test_mode else AUTO_LEARN_PERIOD_SEC) # Shortened for testing
            with self.p.lock, self.dm.m.lock:
                if len(self.dm.m.entries) == MAX_MEMORY:
                    summary = self.dm.m.summarize_and_prune(n_entries=50)
                    if summary:
                        self._post_gui("KawaiiKuro: *spends a moment organizing her memories of us, smiling softly*", speak=False)
                all_user_text = [entry.user for entry in self.dm.m.entries if len(entry.user.split()) > 4]
                if len(all_user_text) < 10:
                    continue
                all_user_text_single_str = " ".join(all_user_text)
                tokens = safe_word_tokenize(all_user_text_single_str.lower())
                tagged = safe_pos_tag(tokens)
                stop_words = safe_stopwords()
                user_entity = self.dm.kg.get_entity('user')
                if user_entity and user_entity.get('attributes',{}).get('name'):
                    stop_words.add(user_entity['attributes']['name'].get('value','').lower())
                nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in stop_words]
                self.p.core_entities.update(nouns)
                if len(self.p.core_entities) > 20:
                    self.p.core_entities = Counter(dict(self.p.core_entities.most_common(20)))
                try:
                    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=1000)
                    tf = vectorizer.fit_transform(all_user_text)
                    feature_names = vectorizer.get_feature_names_out()
                    n_topics = min(3, len(all_user_text) // 5)
                    if n_topics == 0: continue
                    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', learning_offset=50., random_state=0)
                    lda.fit(tf)
                    new_topics = []
                    n_top_words = 5
                    for topic_idx, topic_dist in enumerate(lda.components_):
                        top_words_indices = topic_dist.argsort()[:-n_top_words - 1:-1]
                        topic_words = [feature_names[i] for i in top_words_indices]
                        new_topics.append(topic_words)
                    self.p.learned_topics = new_topics
                    self._post_gui("KawaiiKuro: *takes some nerdy notes on our conversations* I feel like I understand you better now~", speak=False)
                except Exception:
                    pass

                # Consolidate knowledge graph
                newly_inferred = self.kg.consolidate_knowledge()
                if newly_inferred:
                    self._post_gui(f"KawaiiKuro: *has a moment of insight, connecting some dots...*", speak=False)

    def _auto_save_loop(self):
        while not self.stop_flag.is_set():
            self.persistence.save()
            time.sleep(15 if self.test_mode else AUTO_SAVE_PERIOD_SEC)

    def _continuous_listen_loop(self):
        while not self.stop_flag.is_set():
            heard = self.voice.listen()
            if heard:
                # simulate user typing and sending
                reply = self.dm.respond(heard)
                self.mark_interaction()
                self._post_gui(f"You (voice): {heard}\nKawaiiKuro: {reply}")
            time.sleep(1)

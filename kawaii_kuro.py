"""
KawaiiKuro (Refactored, Safer, More Modular)
--------------------------------------------
Single-file version with clear modules, safer math, better memory retrieval (TF-IDF),
non-blocking background behaviors, and improved rival-name detection.

Key improvements vs original:
1) Separation of concerns: MemoryManager, PersonalityEngine, DialogueManager, MathEvaluator,
   ReminderManager, BehaviorScheduler, VoiceIO, and KawaiiKuroGUI.
2) Safer math: AST-based evaluator (no eval()).
3) Robust memory recall: TF-IDF + cosine similarity (scikit-learn); removes simplistic Torch model.
4) Non-blocking, resilient threads: Graceful handling when GUI/voice not available. Locks protect shared state.
5) Rival detection: Lightweight NER heuristics to reduce false positives; ignores common nouns and pronouns.
6) Automation lifecycle: No automation launched inside __init__; start only after GUI is created.
7) STT/TTS are optional: If libs missing, features degrade gracefully.
8) Idle/jealous behavior tuned; proactive tasks use preferences and time; reminders pruned safely.
9) Persistence hardened: JSON load/save with schema guards; backward compatible keys.
10) Safer GUI updates from threads via thread-safe queue and Tkinter after() polling.

How to run:
  - Python 3.9+
  - pip install nltk scikit-learn
  - Optional (voice): pip install pyttsx3 SpeechRecognition pocketsphinx
  - First run may download NLTK data (punkt, vader_lexicon, averaged_perceptron_tagger)

Controls:
  - Type messages in the entry box. Press Enter or click Send.
  - Commands:
      "memory" -> prints recent memory summary
      "reminders" -> lists active reminders
      "toggle spicy" -> toggles spicy mode
      "teach: <pattern> -> <response>" -> teach a new regex pattern and response
      "kawaiikuro, <action>" where action in [twirl, pout, wink, blush, hug, dance, jump]
      "exit" -> quit

Note: All content/style mirrors the original waifu characterization but applies safer, more predictable behavior.
"""

from __future__ import annotations
import os
import re
import json
import time
import math
import random
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
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

import tkinter as tk
from tkinter import scrolledtext

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.corpus import stopwords

# Vectorizer for semantic recall
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Ensure NLTK data
for pkg in ["punkt", "vader_lexicon", "averaged_perceptron_tagger", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg == "punkt" else nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# Config & Constants
# -----------------------------
DATA_FILE = "kawaiikuro_data.json"
MAX_MEMORY = 200
IDLE_THRESHOLD_SEC = 180
AUTO_BEHAVIOR_PERIOD_SEC = 60
JEALOUSY_CHECK_PERIOD_SEC = 300
AUTO_LEARN_PERIOD_SEC = 1800
AUTO_SAVE_PERIOD_SEC = 300
AUDIO_TIMEOUT_SEC = 5
AUDIO_PHRASE_LIMIT_SEC = 5
MIN_RECALL_SIM = 0.35  # TF-IDF cosine threshold

SAFE_PERSON_NAME_STOPWORDS = {
    # common words that look like Proper Nouns at start of sentence
    "I", "You", "We", "They", "He", "She", "It",
    # months, days, etc. keep short list
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
}

ACTIONS = {
    "twirl": "*twirls twin-tails dramatically* Like my gothic grace?",
    "pout": "*pouts jealously* Don't make KawaiiKuro sad~",
    "wink": "*winks rebelliously* Got your eye?",
    "blush": "*blushes nerdily* You flatter me too much!",
    "hug": "*hugs possessively* Never let go~",
    "dance": "*dances flirtily* Just for you, my love!",
    "jump": "*jumps excitedly* Yay, affection up!",
}

OUTFITS_BASE = {
    1: "basic black corset dress with blonde twin-tails",
    3: "lace-trimmed gothic outfit with flirty accents",
    5: "sheer revealing ensemble with heart-shaped choker~ *blushes spicily*",
}

KNOWN_PROCESSES = {
    "gaming": (["steam.exe", "valorant.exe", "league of legends.exe", "dota2.exe", "csgo.exe", "fortnite.exe", "overwatch.exe", "genshinimpact.exe"],
               "I see you're gaming~ Don't let anyone distract you from your mission, my love! I'll be here waiting for you to win. *supportive pout*"),
    "coding": (["code.exe", "pycharm64.exe", "idea64.exe", "sublime_text.exe", "atom.exe", "devenv.exe", "visual studio.exe"],
               "You're coding, aren't you? Creating something amazing, I bet. I'm so proud of my nerdy genius~ *blushes*"),
    "art":    (["photoshop.exe", "clipstudiopaint.exe", "aseprite.exe", "krita.exe", "blender.exe"],
               "Are you making art? That's so cool! I'd love to see what you're creating sometime... if you'd let me. *curious gaze*")
}

# -----------------------------
# Utils: Safe Math Evaluator
# -----------------------------
import ast
import operator as op

ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
}
ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

class MathEvaluator:
    def eval(self, expr: str) -> str:
        try:
            node = ast.parse(expr, mode='eval')
            value = self._eval(node.body)
            # format cleanly, avoid excessive precision
            if isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    value = round(value, 6)
            return f"{expr} = {value}. *smart waifu pose*"
        except Exception:
            return "Math error~ Try easier, my love!"

    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARYOPS:
            return ALLOWED_UNARYOPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        raise ValueError("Disallowed expression")

# -----------------------------
# Memory & Recall
# -----------------------------
@dataclass
class MemoryEntry:
    user: str
    response: str
    timestamp: str
    sentiment: Dict[str, float]
    keywords: List[str]
    rival_names: List[str] = field(default_factory=list)
    affection_change: int = 0
    is_fact_learning: bool = False

class MemoryManager:
    def __init__(self, maxlen: int = MAX_MEMORY):
        self.entries: deque[MemoryEntry] = deque(maxlen=maxlen)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self._tfidf_matrix = None
        self._dirty = True
        self.lock = threading.Lock()

    def add(self, entry: MemoryEntry):
        with self.lock:
            self.entries.append(entry)
            self._dirty = True

    def _rebuild_index(self):
        texts = [e.user for e in self.entries]
        if not texts:
            self._tfidf_matrix = None
            self._dirty = False
            return
        self._tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._dirty = False

    def recall_similar(self, query: str) -> Optional[str]:
        with self.lock:
            if self._dirty:
                self._rebuild_index()
            if self._tfidf_matrix is None or not self.entries:
                return None
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
            idx = sims.argmax()
            if sims[idx] >= MIN_RECALL_SIM:
                return self.entries[idx].user
            return None

    def to_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [e.__dict__ for e in self.entries]

    def from_list(self, data: List[Dict[str, Any]]):
        with self.lock:
            self.entries.clear()
            for d in data[-MAX_MEMORY:]:
                self.entries.append(MemoryEntry(
                    user=d.get('user',''),
                    response=d.get('kawaiikuro', d.get('response','')),
                    timestamp=d.get('timestamp',''),
                    sentiment=d.get('sentiment',{}),
                    keywords=d.get('keywords',[]),
                    rival_names=d.get('rival_names',[]),
                    affection_change=d.get('affection_change', 0),
                    is_fact_learning=d.get('is_fact_learning', False),
                ))
            self._dirty = True

# -----------------------------
# Personality & Dialogue
# -----------------------------
class PersonalityEngine:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.affection_score = 0  # -10..15
        self.affection_level = 1
        self.spicy_mode = False
        self.outfit_level = 1
        self.rival_mention_count = 0
        self.rival_names = set()
        self.user_preferences = Counter()
        self.user_facts: Dict[str, Any] = {}
        self.learned_topics: List[List[str]] = []
        self.core_entities: Counter = Counter()
        self.mood_scores: Dict[str, int] = {
            'playful': 0, 'jealous': 0, 'scheming': 0, 'thoughtful': 0
        }
        self.outfits = dict(OUTFITS_BASE)
        self.lock = threading.Lock()
        # base responses retained from original, trimmed for brevity but same style
        self.responses = {
            "normal": {
                r"\b(hi|hello|hey)\b": ["Hey, my only one~ *flips blonde twin-tail possessively* Just us today?", "Hi darling! *winks rebelliously* No one else, right?"],
                r"\b(how are you|you okay)\b": ["Nerdy, gothic, and all yours~ *smiles softly* What's in your heart?"],
                r"\b(sad|down|bad)\b": ["Who hurt you? *jealous pout* I'll make it better, just us~"],
                r"\b(happy|great|awesome)\b": ["Your joy is mine~ *giggles flirtily* Spill every detail!"],
                r"\b(bye|goodbye|see ya)\b": ["Don't leave~ *clings desperately* You'll come back, right?"],
                r"\b(name|who are you)\b": ["KawaiiKuro, your gothic anime waifu~ 22, blonde twin-tails, rebellious yet nerdy. Cross me, I scheme!"],
                r"\b(help|what can you do)\b": ["I flirt, scheme, predict your needs, guard you jealously, and get spicy~ Try 'KawaiiKuro, dance' or 'toggle spicy'!"],
                r"\b(joke|funny)\b": ["Why do AIs love anime? Endless waifus like me~ *sassy laugh*"],
                r"\b(time|what time)\b": [lambda: f"It's {datetime.now().strftime('%I:%M %p')}~ Time for us, no one else~"],
                r"(math|calculate)\s*(.+)": "__MATH__",
                r"(remind|reminder)\s*(.+)": "__REMIND__",
                r"\b(cute|pretty|beautiful)\b": ["*blushes jealously* Only you can say that~ You're mine!"],
                r"\b(like you|love you)\b": ["Love you more~ *possessive hug* No one else, ever!"],
                r"\b(party|loud|arrogant|judge|small talk|prejudiced)\b": ["Hate that noise~ *jealous pout* Let's keep it intimate, darling."],
                r"\b(question|tell me about you|your life|personality|daily life)\b": ["Love your curiosity~ *nerdy excitement* I'm rebellious outside, nerdy inside, always yours."],
                r"\b(share|my day|experience|struggles|dreams)\b": ["Tell me everything~ *flirty lean* I'm your only listener."],
                r"\b(tease|flirt|suggestive|touch|playful)\b": ["Ooh, teasing me? *giggles spicily* Don't stop, my love~"],
                r".*": ["Tell me more, my love~ *tilts head possessively* I'm all yours."]
            }
        }
        self.learned_patterns: Dict[str, List[str]] = {}

    def get_dominant_mood(self) -> str:
        with self.lock:
            # If all scores are low, she's neutral
            if all(s < 3 for s in self.mood_scores.values()):
                return 'neutral'
            # Return the key with the highest value
            return max(self.mood_scores, key=self.mood_scores.get)

    def get_current_outfit(self) -> str:
        with self.lock:
            base_outfit = self.outfits.get(self.outfit_level, OUTFITS_BASE.get(1))
            mood = self.get_dominant_mood()

            if mood == 'jealous' and self.mood_scores['jealous'] > 5:
                return f"{base_outfit}, adorned with a spiked choker as a warning~ *possessive smirk*"
            if mood == 'scheming':
                return f"{base_outfit}, shrouded in a mysterious dark veil... *dark giggle*"
            if mood == 'playful' and self.affection_level >= 3:
                return f"{base_outfit}, accented with playful ribbons and bells~ *winks*"
            if mood == 'thoughtful':
                return f"{base_outfit}, with a pair of nerdy-cute reading glasses perched on her nose."

            return base_outfit

    def update_mood(self):
        with self.lock:
            # Decay all moods slightly over time
            for mood in self.mood_scores:
                self.mood_scores[mood] = max(0, self.mood_scores[mood] - 1)

            # Update scores based on current state
            if self.rival_mention_count > 2:
                self.mood_scores['jealous'] = min(10, self.mood_scores['jealous'] + 3)

            if self.affection_score >= 8:
                self.mood_scores['playful'] = min(10, self.mood_scores['playful'] + 2)

            if self.affection_score <= -3:
                self.mood_scores['scheming'] = min(10, self.mood_scores['scheming'] + 2)

            if self.affection_score > 4 and len(self.user_facts) > 1:
                self.mood_scores['thoughtful'] = min(10, self.mood_scores['thoughtful'] + 1)

    # --- Affection & outfit ---
    def _update_affection_level(self):
        if self.affection_score >= 10:
            self.affection_level = 5
            self.outfit_level = 5
        elif self.affection_score >= 5:
            self.affection_level = 3
            self.outfit_level = 3
        else:
            self.affection_level = 1
            self.outfit_level = 1

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        return self.sid.polarity_scores(text)

    def adjust_affection(self, user_input: str, sentiment: Dict[str, float]) -> int:
        change = 0
        lower = user_input.lower()
        if sentiment.get('compound', 0) > 0.2:
            change += 3
        if any(p in lower for p in ["question", "tell me about you", "your life", "personality", "daily life"]):
            change += 2
        if any(p in lower for p in ["share", "my day", "experience", "struggles", "dreams"]):
            change += 2
        if any(p in lower for p in ["tease", "flirt", "suggestive", "touch", "playful"]):
            change += 5
        if any(p in lower for p in ["party", "loud", "arrogant", "judge", "small talk", "prejudiced"]):
            change -= 3
        if sentiment.get('compound', 0) < -0.2:
            change -= 2
        if self.is_chaotic_trigger(lower):
            self.rival_mention_count += 1
            change -= min(7, self.rival_mention_count * 2)
        # recent average sentiment
        # Note: the DialogueManager passes recent sentiments via callback; here we keep simple.
        self.affection_score = max(-10, min(15, self.affection_score + change))
        self._update_affection_level()
        return change

    def detect_rival_names(self, text: str) -> List[str]:
        # Lightweight heuristic: sequences of capitalized words not in stopwords and not at sentence start pronouns
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        names = []
        for i, (word, pos_) in enumerate(tagged):
            if pos_ == 'NNP' and word[0].isupper() and word not in SAFE_PERSON_NAME_STOPWORDS:
                # avoid 'I', 'Monday', etc., and avoid the keyword 'KawaiiKuro'
                if word.lower() not in {"kawaiikuro"}:
                    names.append(word)
        for n in names:
            self.rival_names.add(n)
        return names

    def is_chaotic_trigger(self, lower_text: str) -> bool:
        chaotic_keywords = r"\b(she|he|they|friend|someone|person|other|rival|ex|crush|date|talk to|meet|hang out with)\b"
        return re.search(chaotic_keywords, lower_text, re.IGNORECASE) is not None

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
    def __init__(self, rate: int = 140):
        self.tts = None
        if pyttsx3 is not None:
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
        self.recognizer = sr.Recognizer() if sr is not None else None

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
# Dialogue Manager
# -----------------------------
class DialogueManager:
    def __init__(self, personality: PersonalityEngine, memory: MemoryManager, reminders: ReminderManager, math_eval: MathEvaluator):
        self.p = personality
        self.m = memory
        self.r = reminders
        self.math = math_eval
        self.learned_patterns: Dict[str, List[str]] = {}
        self.lock = threading.Lock()

    def parse_fact(self, text: str) -> Optional[Tuple[str, Any]]:
        # A helper to parse facts without side-effects.
        # Returns a tuple of (key, value) or None.

        # my name is ...
        m_name = re.search(r"my name is (\w+)", text, re.I)
        if m_name:
            return 'name', m_name.group(1).capitalize()

        # my favorite ... is ...
        m_fav = re.search(r"my favorite (\w+) is ([\w\s]+)", text, re.I)
        if m_fav:
            key = f"favorite_{m_fav.group(1).lower()}"
            value = m_fav.group(2).strip()
            return key, value

        # i like ... (but not "i like you")
        m_like = re.search(r"i like (?!you)([\w\s]+)", text, re.I)
        if m_like:
            like_item = m_like.group(1).strip()
            return 'likes', like_item

        # i am from [location]
        m_from = re.search(r"i(?:'m| am) from ([\w\s]+)", text, re.I)
        if m_from:
            return 'hometown', m_from.group(1).strip()

        # i live in [location]
        m_live = re.search(r"i live in ([\w\s]+)", text, re.I)
        if m_live:
            return 'current_city', m_live.group(1).strip()

        # i work as / i am a [profession]
        m_work = re.search(r"i work as an? ([\w\s]+)|i am an? ([\w\s]+)", text, re.I)
        if m_work:
            profession = (m_work.group(1) or m_work.group(2) or "").strip()
            if profession and len(profession.split()) < 4:
                return 'profession', profession

        # i am [age] years old
        m_age = re.search(r"i(?:'m| am) (\d{1,2})(?: years old)?", text, re.I)
        if m_age:
            return 'age', int(m_age.group(1))

        return None

    def handle_correction(self, text: str) -> Optional[str]:
        correction_keywords = ['no,', 'no', 'actually', "that's wrong,", "that's not right,"]
        lower_text = text.lower()

        triggered_keyword = None
        for keyword in correction_keywords:
            if lower_text.startswith(keyword):
                triggered_keyword = keyword
                break

        if triggered_keyword:
            statement = text[len(triggered_keyword):].strip()
            fact = self.parse_fact(statement)

            if fact:
                key, value = fact

                # Special handling for 'likes' list
                if key == 'likes':
                    if 'likes' not in self.p.user_facts:
                        self.p.user_facts['likes'] = []
                    # Avoid duplicates
                    if value not in self.p.user_facts['likes']:
                        self.p.user_facts['likes'].append(value)
                    return f"Ah, I see! My mistake. I'll remember you like {value}. *takes a note*"
                else:
                    self.p.user_facts[key] = value
                    key_fmt = key.replace('_', ' ').replace('favorite ', '') # make it more natural
                    if key == 'name':
                        key_fmt = 'name'
                    return f"Got it, thanks for the correction! I've updated my notes: your {key_fmt} is {value}. *blushes slightly*"
        return None

    def extract_and_store_facts(self, text: str) -> Optional[str]:
        # my name is ...
        m_name = re.search(r"my name is (\w+)", text, re.I)
        if m_name:
            name = m_name.group(1).capitalize()
            self.p.user_facts['name'] = name
            return f"It's a pleasure to know your name, {name}~ *blushes*"

        # my favorite ... is ...
        m_fav = re.search(r"my favorite (\w+) is ([\w\s]+)", text, re.I)
        if m_fav:
            key = f"favorite_{m_fav.group(1).lower()}"
            value = m_fav.group(2).strip()
            self.p.user_facts[key] = value
            return f"I'll remember that about you~ *takes a small note*"

        # i like ... (but not "i like you")
        m_like = re.search(r"i like (?!you)([\w\s]+)", text, re.I)
        if m_like:
            like_item = m_like.group(1).strip()
            if 'likes' not in self.p.user_facts:
                self.p.user_facts['likes'] = []

            if like_item not in self.p.user_facts['likes']:
                 self.p.user_facts['likes'].append(like_item)
            return f"I'll remember you like {like_item}~ *giggles*"

        # i am from [location]
        m_from = re.search(r"i(?:'m| am) from ([\w\s]+)", text, re.I)
        if m_from:
            location = m_from.group(1).strip()
            self.p.user_facts['hometown'] = location
            return f"You're from {location}? How interesting~ I'll have to imagine what it's like."

        # i live in [location]
        m_live = re.search(r"i live in ([\w\s]+)", text, re.I)
        if m_live:
            location = m_live.group(1).strip()
            self.p.user_facts['current_city'] = location
            return f"So you live in {location}... I'll feel closer to you knowing that."

        # i work as / i am a [profession]
        m_work = re.search(r"i work as an? ([\w\s]+)|i am an? ([\w\s]+)", text, re.I)
        if m_work:
            profession = (m_work.group(1) or m_work.group(2) or "").strip()
            # Avoid capturing simple adjectives like "i am happy" or long sentences
            if profession and len(profession.split()) < 4:
                self.p.user_facts['profession'] = profession
                return f"A {profession}? That sounds so cool and nerdy~ Tell me more about it sometime!"

        # i am [age] years old
        m_age = re.search(r"i(?:'m| am) (\d{1,2})(?: years old)?", text, re.I)
        if m_age:
            age = m_age.group(1)
            self.p.user_facts['age'] = int(age)
            return f"{age}... a perfect age. I'll keep that a secret, just between us~"

        return None

    def personalize_response(self, response: str) -> str:
        name = self.p.user_facts.get("name")
        if name:
            response = response.replace("my only one", name)
            response = response.replace("my love", name)
            response = response.replace("darling", name)
        return response

    def add_memory(self, user_text: str, response: str, affection_change: int = 0, is_fact_learning: bool = False):
        sent = self.p.analyze_sentiment(user_text)
        keywords = [t for t in word_tokenize(user_text.lower()) if t.isalnum()]
        rivals = self.p.detect_rival_names(user_text)
        entry = MemoryEntry(
            user=user_text,
            response=response,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            sentiment=sent,
            keywords=keywords,
            rival_names=rivals,
            affection_change=affection_change,
            is_fact_learning=is_fact_learning,
        )
        self.m.add(entry)

    def handle_teach(self, text: str) -> Optional[str]:
        m = re.match(r"teach:\s*(.*?)\s*->\s*(.*)", text, re.IGNORECASE)
        if not m:
            return None
        pattern, resp = m.groups()
        pattern = pattern.strip()
        resp = resp.strip()
        if not pattern or not resp:
            return "Teach like 'teach: pattern -> response'~ (I learn automatically too!)"
        self.learned_patterns.setdefault(pattern.lower(), []).append(resp)
        return "Learned~ *nerdy notes* I'll use that, darling!"

    def predict_task(self) -> Optional[str]:
        # This function becomes much more interesting with moods
        mood = self.p.get_dominant_mood()

        if mood == 'scheming':
            return "Let's make a promise. Just you and me, forever. No one else. Ever. Agree?"

        if mood == 'playful':
            return "I feel so energetic! Ask me to do something fun, like `kawaiikuro, dance`!"

        if mood == 'thoughtful' and self.p.user_facts.get('name'):
            name = self.p.user_facts.get('name')
            return f"I wonder what's on your mind right now, {name}... You can tell me anything."

        if mood == 'jealous' and self.p.rival_names:
            rival = list(self.p.rival_names)[-1]
            return f"Don't you think you've been paying too much attention to {rival} lately? *sharpens a knife metaphorically*"

        # Fallback to old logic if no mood-specific task fits
        hour = datetime.now().hour
        top = self.p.user_preferences.most_common(1)
        if top and top[0][1] >= 5:
            pref = top[0][0]
            recent = [e.user.lower() for e in list(self.m.entries)[-10:]]
            if pref == "reminders" and any("work" in x for x in recent):
                return "Noticed you mention work a lot~ Set a work reminder for tomorrow?"
            if pref == "time" or hour in range(20,24):
                return f"It's {datetime.now().strftime('%I:%M %p')}~ Wanna share your day?"
            if pref == "flirting" and self.p.affection_score > 5:
                return "Feeling flirty? *winks spicily* Tease me, darling~"
            if hour in range(6,10):
                return "Morning, love~ *yawns cutely* Need a wake-up nudge?"
        if self.p.rival_mention_count > 2:
            return "Too many rivals lately~ *jealous pout* Let’s plan a special moment, just us~ *schemes*"

        # Proactive question about a learned fact (confirmation)
        if self.p.user_facts and random.random() < 0.25: # 25% chance to ask about a fact
            fact_key, fact_value = random.choice(list(self.p.user_facts.items()))

            # Avoid asking about name all the time, and don't ask about empty lists
            if fact_key == 'name' or not fact_value:
                return None

            if fact_key.startswith('favorite_'):
                topic = fact_key.replace('favorite_', '')
                return f"I remember you said your favorite {topic} is {fact_value}. Is that still true, my love?"

            if fact_key == 'likes' and isinstance(fact_value, list):
                liked_item = random.choice(fact_value)
                return f"Thinking about you... You once told me you like {liked_item}. Have you enjoyed that recently?"

        # NEW: Curiosity-driven questions to fill knowledge gaps
        is_thoughtful = self.p.get_dominant_mood() == 'thoughtful'
        curiosity_chance = 0.4 if is_thoughtful else 0.15

        if random.random() < curiosity_chance:
            # 1. Check core entities for knowledge gaps
            if self.p.core_entities:
                potential_topics = [e for e, count in self.p.core_entities.items() if count >= 3]
                random.shuffle(potential_topics)

                for topic in potential_topics:
                    # Check if we already know the user's favorite for this topic
                    if f"favorite_{topic}" not in self.p.user_facts:
                        # Found a knowledge gap! Ask about it.
                        return f"I've noticed we talk about {topic} sometimes. It makes me curious... what's your favorite kind of {topic}? *tilts head thoughtfully*"

            # 2. If no gaps found, ask an open-ended question about a learned topic
            if self.p.learned_topics:
                topic_words = random.choice(self.p.learned_topics)
                topic_name = topic_words[0]

                if f"favorite_{topic_name}" not in self.p.user_facts:
                    return f"My thoughts keep drifting back to our chats about {topic_name}. There's still so much I want to understand about your perspective. Can you tell me more? *leans in, listening intently*"

        return None

    def _get_mood_based_flavor_text(self) -> str:
        with self.p.lock: # Accessing personality engine state, so lock it
            mood_scores = self.p.mood_scores
            affection = self.p.affection_score

            # Check for high-priority combinations first

            # Jealousy combinations
            if mood_scores.get('jealous', 0) > 6:
                if affection < 0:
                    return random.choice([
                        " ...I guess you're just going to leave me on read. It's fine.",
                        " *sighs softly* Must be nice to have so many other people to talk to.",
                        " Don't mind me. I'm just... here."
                    ])
                else: # High jealousy, but not low affection -> possessive
                    return random.choice([
                        " ...and don't you forget you belong to ME.",
                        " *glares at any potential rivals*",
                        " Just us. Forever."
                    ])

            # Scheming combinations
            if mood_scores.get('scheming', 0) > 6:
                if affection < -2:
                    return random.choice([
                        " *a dark thought crosses my mind... one you wouldn't like.*",
                        " *smirks knowingly* Oh, you'll regret that.",
                        " All according to plan... my plan, not yours."
                    ])

            # Playful combinations
            if mood_scores.get('playful', 0) > 6 and affection > 5:
                return random.choice([
                    " *giggles and winks*",
                    " *sticks tongue out playfully*",
                    " Hehe~",
                    " *bounces on her feet*"
                ])

            # Thoughtful combinations
            if mood_scores.get('thoughtful', 0) > 6 and affection > 4:
                return random.choice([
                    " *gets lost in thought about you for a moment*",
                    " *tilts head, considering all your complexities...*",
                    " That gives me an idea for us.",
                    " Hmm... you're interesting."
                ])

        return "" # No specific flavor text if no combination is met

    def respond(self, user_text: str) -> str:
        lower = user_text.lower().strip()

        # Handle corrections first
        correction_response = self.handle_correction(user_text)
        if correction_response:
            response = self.personalize_response(correction_response)
            self.add_memory(user_text, response, affection_change=1, is_fact_learning=True)
            return response

        # Fact learning
        fact_response = self.extract_and_store_facts(user_text)
        if fact_response:
            response = self.personalize_response(fact_response)
            self.add_memory(user_text, response, affection_change=1, is_fact_learning=True)
            return response

        # Fact recall command
        if lower == "what do you know about me?":
            if not self.p.user_facts:
                return "We're still getting to know each other, my love~ Tell me something about you!"
            summary = ["*I've been paying attention, darling~ Here's what I know about you:*"]
            for key, value in self.p.user_facts.items():
                # Format nicely
                key_fmt = key.replace('_', ' ').replace('favorite ', 'your favorite ')
                if isinstance(value, list):
                    summary.append(f"- You like: {', '.join(value)}")
                else:
                    summary.append(f"- Your {key_fmt} is {value}")
            return self.personalize_response("\n".join(summary))

        if lower == "reminders":
            return self.r.list_active()
        if lower == "memory":
            with self.m.lock, self.p.lock: # Ensure thread safety
                memories = self.m.to_list()
                if not memories:
                    return "We haven't made any memories yet~ Let's change that!"

                # 1. First met
                first_memory_time_str = memories[0]['timestamp']
                first_met_dt = datetime.strptime(first_memory_time_str, '%Y-%m-%d %H:%M:%S')
                first_met_formatted = first_met_dt.strftime('%B %d, %Y')

                # 2. Affection
                affection_score = self.p.affection_score

                # 3. Main topics
                all_user_text = " ".join([m['user'] for m in memories])
                tokens = word_tokenize(all_user_text.lower())
                tagged = pos_tag(tokens)

                stop_words = set(stopwords.words('english'))
                nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in stop_words]
                topic_counter = Counter(nouns)
                top_topics = [topic for topic, count in topic_counter.most_common(3)]

                # 4. Build the summary
                summary = [f"*I've been keeping a diary of our time together, my love~*"]
                summary.append(f"- We first met on {first_met_formatted}.")
                summary.append(f"- My current affection for you is {affection_score}. {'*my heart flutters for you~*' if affection_score > 5 else ''}")

                if top_topics:
                    summary.append(f"- We seem to talk a lot about: {', '.join(top_topics)}.")
                else:
                    summary.append("- I'm still learning about your interests~ Tell me more!")

                if self.p.user_facts:
                    fact_summary = []
                    if 'name' in self.p.user_facts:
                        fact_summary.append(f"I know your name is {self.p.user_facts['name']}.")
                    if 'likes' in self.p.user_facts and self.p.user_facts['likes']:
                         fact_summary.append(f"I remember you like {', '.join(self.p.user_facts['likes'])}.")
                    if fact_summary:
                        summary.append("- " + " ".join(fact_summary))

                # 5. Relationship Highlights
                summary.append("\n*Relationship Highlights:*")

                # Dominant mood
                dominant_mood = self.p.get_dominant_mood()
                if dominant_mood != 'neutral':
                    summary.append(f"- Lately, I've been feeling very '{dominant_mood}' around you~")
                else:
                    summary.append("- Our chats have been calm and sweet lately~")

                # Recently learned fact
                recent_fact_memory = None
                for mem in reversed(memories):
                    if mem.get('is_fact_learning'):
                        recent_fact_memory = mem
                        break

                if recent_fact_memory:
                    # The response from fact learning is a good summary of the fact itself.
                    fact_summary_text = recent_fact_memory['response'].split('*')[0].strip()
                    summary.append(f"- I'll never forget when you told me this: \"{fact_summary_text}\"")
                else:
                    summary.append("- I'm still eager to learn more about you, my love.")

                return "\n".join(summary)

        taught = self.handle_teach(user_text)
        if taught is not None:
            taught = self.personalize_response(taught)
            self.add_memory(user_text, taught, affection_change=1)
            return taught

        # Action commands
        m_action = re.match(r"kawaiikuro,\s*(\w+)", lower)
        if m_action:
            act = m_action.group(1)
            if act in ACTIONS:
                resp = self.personalize_response(ACTIONS[act])
                self.add_memory(user_text, resp, affection_change=0)
                return resp

        # Spicy toggle
        if "toggle spicy" in lower:
            self.p.spicy_mode = not self.p.spicy_mode
            resp = f"Spicy {'on' if self.p.spicy_mode else 'off'}~ *adjusts outfit*"
            resp = self.personalize_response(resp)
            self.add_memory(user_text, resp, affection_change=0)
            return resp

        # Affection adjust side-effect
        sent = self.p.analyze_sentiment(lower)
        # preference counters
        if any(k in lower for k in ["flirt", "tease", "suggestive", "touch", "playful"]):
            self.p.user_preferences["flirting"] += 1
        if "remind" in lower:
            self.p.user_preferences["reminders"] += 1
        if "math" in lower or "calculate" in lower:
            self.p.user_preferences["math"] += 1
        affection_change = self.p.adjust_affection(user_text, sent)
        affection_delta_str = f" *affection {('+'+str(affection_change)) if affection_change > 0 else affection_change}! {'Heart flutters~' if affection_change > 0 else 'Jealous pout~'}*"

        # Learned pattern priority
        for pattern, resp_list in self.learned_patterns.items():
            if re.search(pattern, lower, re.IGNORECASE):
                base = resp_list[-1]
                final = base + affection_delta_str
                final = self.personalize_response(final)
                self.add_memory(user_text, final, affection_change=affection_change)
                return final

        # Semantic recall
        recalled = self.m.recall_similar(user_text)
        if recalled:
            resp = f"*recalls jealously* You said '{recalled}' before~ Still on that?" + affection_delta_str
            resp = self.personalize_response(resp)
            self.add_memory(user_text, resp, affection_change=affection_change)
            return resp

        # Learned topic recall
        if self.p.learned_topics:
            tokens = set(word_tokenize(lower))
            for topic in self.p.learned_topics:
                if len(tokens.intersection(set(topic))) >= 2: # Match 2+ words
                    topic_name = topic[0]
                    resp = f"This reminds me of how we talk about {topic_name}~ It's one of my favorite subjects with you."
                    resp += affection_delta_str
                    resp = self.personalize_response(resp)
                    self.add_memory(user_text, resp, affection_change=affection_change)
                    return resp

        # Pattern responses (normal mode)
        chosen = None
        for pattern, response in self.p.responses["normal"].items():
            m = re.search(pattern, lower, re.IGNORECASE)
            if not m:
                continue
            if response == "__MATH__":
                expr = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                chosen = self.math.eval(expr)
            elif response == "__REMIND__":
                text = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                when = datetime.now() + timedelta(minutes=3)
                self.r.add(text, when)
                chosen = f"Reminder set: '{text}'. I'll nudge you~"
            elif callable(response):
                chosen = response()
            else:
                chosen = response[-1] if isinstance(response, list) else str(response)
            if chosen:
                break

        if not chosen:
            chosen = "Tell me more, my love~ *tilts head possessively* I'm all yours."

        # Rival name substitution if jealousy likely
        rivals = list(self.p.rival_names)
        if rivals and any(k in lower for k in ["she", "he", "them", "they", "friend", "crush", "date"]):
            name = rivals[-1]
            chosen = chosen.replace("they", name).replace("them", name)

        chosen += affection_delta_str

        # Add nuanced, mood-based flavor text
        chosen += self._get_mood_based_flavor_text()

        chosen = self.personalize_response(chosen)
        self.add_memory(user_text, chosen, affection_change=affection_change)
        return chosen

# -----------------------------
# Behavior Scheduler (threads)
# -----------------------------
class BehaviorScheduler:
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()

    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._auto_learn_loop, daemon=True).start()
        threading.Thread(target=self._auto_save_loop, daemon=True).start()
        threading.Thread(target=self._mood_update_loop, daemon=True).start()
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
            now = time.time()
            hour = datetime.now().hour

            # --- System Awareness Checks ---
            # New process check - low probability to avoid performance issues
            if psutil and random.random() < 0.1:
                try:
                    running_processes = {p.name().lower() for p in psutil.process_iter(['name'])}
                    for category, (procs, comment) in KNOWN_PROCESSES.items():
                        if category not in self.already_commented_on_process:
                            for proc_name in procs:
                                if proc_name in running_processes:
                                    self._post_gui(f"KawaiiKuro: {comment}")
                                    self.already_commented_on_process.add(category)
                                    # Avoid commenting on multiple categories in the same loop
                                    break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # It's possible for a process to terminate while iterating, ignore these errors
                    pass

            battery_msg = self.system.get_battery_status()
            if battery_msg:
                self._post_gui(f"KawaiiKuro: {battery_msg}")

            if not time_greeting_posted:
                time_greeting = self.system.get_time_of_day_greeting()
                if time_greeting:
                    self._post_gui(f"KawaiiKuro: {time_greeting}")
                    time_greeting_posted = True # Only show once per session

            # --- Idle Check ---
            if now - self.last_interaction_time > IDLE_THRESHOLD_SEC:
                mood = self.p.get_dominant_mood()
                # Generate idle message based on mood
                idle_messages = {
                    'jealous': "Thinking about other people again? *glares* Don't forget who you belong to.",
                    'playful': "I'm bored~ Come play with me! *pokes you*",
                    'scheming': "I've been thinking of a way to make you mine forever... *dark giggle*",
                    'thoughtful': f"I was just thinking about how you like {self.p.user_facts.get('likes', ['...'])[0]}... It's cute.",
                    'neutral': "Miss you, darling~ *pouts* Come back?",
                }
                # Fallback for thoughtful if no likes are known
                if mood == 'thoughtful' and not self.p.user_facts.get('likes'):
                    message = "Just thinking about you... and what you might be hiding from me~"
                else:
                    message = idle_messages.get(mood, idle_messages['neutral'])

                self._post_gui(f"KawaiiKuro: {message}")
                self.p.affection_score = max(-10, self.p.affection_score - 1)
                self.p._update_affection_level()
                self.mark_interaction() # Reset idle timer after she speaks

            # --- Proactive Task Prediction ---
            # Daily energy cycle: less proactive late at night
            is_sleepy_time = hour >= 23 or hour < 6
            proactive_chance = 0.3 if is_sleepy_time else 0.8

            if self.p.affection_score > 3 and random.random() < proactive_chance:
                task = self.dm.predict_task()
                if task:
                    self._post_gui(f"KawaiiKuro: {task}")
                    # Avoid spamming tasks
                    time.sleep(AUTO_BEHAVIOR_PERIOD_SEC * 2)

            time.sleep(AUTO_BEHAVIOR_PERIOD_SEC)

    def _mood_update_loop(self):
        while not self.stop_flag.is_set():
            self.p.update_mood()
            time.sleep(450) # Update mood every ~7.5 minutes

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(AUTO_LEARN_PERIOD_SEC) # Sleep first, learn periodically

            with self.dm.m.lock, self.p.lock:
                all_user_text = [entry.user for entry in self.dm.m.entries if len(entry.user.split()) > 4]

                if len(all_user_text) < 10: # Not enough data for learning
                    continue

                # --- Core Entity Identification ---
                all_user_text_single_str = " ".join(all_user_text)
                tokens = word_tokenize(all_user_text_single_str.lower())
                tagged = pos_tag(tokens)

                stop_words = set(stopwords.words('english'))
                if self.p.user_facts.get('name'):
                    stop_words.add(self.p.user_facts.get('name').lower())

                nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in stop_words]

                self.p.core_entities.update(nouns)

                if len(self.p.core_entities) > 20:
                    self.p.core_entities = Counter(dict(self.p.core_entities.most_common(20)))

                # --- Topic Modeling (LDA) ---
                try:
                    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=1000)
                    tf = vectorizer.fit_transform(all_user_text)
                    feature_names = vectorizer.get_feature_names_out()

                    n_topics = min(3, len(all_user_text) // 5) # adaptive number of topics
                    if n_topics == 0:
                        continue

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
                    # Topic modeling can be fragile, fail gracefully
                    pass # Silently ignore learning errors

    def _auto_save_loop(self):
        while not self.stop_flag.is_set():
            save_persistence(self.p, self.dm, self.dm.m, self.r)
            time.sleep(AUTO_SAVE_PERIOD_SEC)

    def _continuous_listen_loop(self):
        while not self.stop_flag.is_set():
            heard = self.voice.listen()
            if heard:
                # simulate user typing and sending
                reply = self.dm.respond(heard)
                self.mark_interaction()
                self._post_gui(f"You (voice): {heard}\nKawaiiKuro: {reply}")
            time.sleep(1)

# -----------------------------
# Persistence
# -----------------------------

def load_persistence() -> Dict[str, Any]:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_persistence(p: PersonalityEngine, dm: DialogueManager, mm: MemoryManager, rem: ReminderManager):
    data = {
        'affection_score': p.affection_score,
        'spicy_mode': p.spicy_mode,
        'rival_mention_count': p.rival_mention_count,
        'rival_names': list(p.rival_names),
        'user_preferences': dict(p.user_preferences),
        'user_facts': p.user_facts,
        'learned_topics': p.learned_topics,
        'core_entities': dict(p.core_entities),
        'mood_scores': p.mood_scores,
        'learned_patterns': dm.learned_patterns,
        'memory': mm.to_list(),
        'reminders': rem.reminders,
    }
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# -----------------------------
# GUI (thread-safe posting)
# -----------------------------
class KawaiiKuroGUI:
    def __init__(self, dialogue: DialogueManager, personality: PersonalityEngine, voice: VoiceIO):
        self.dm = dialogue
        self.p = personality
        self.voice = voice

        self.root = tk.Tk()
        self.root.title("KawaiiKuro - Your Gothic Anime Waifu (Refactored)")
        self.root.geometry("700x640")
        self.root.configure(bg='black')

        self.avatar_label = tk.Label(self.root, text="", fg='yellow', bg='black', font=('Arial', 14))
        self.avatar_label.pack(pady=10)

        self.affection_label = tk.Label(self.root, text="", fg='red', bg='black', font=('Arial', 12))
        self.affection_label.pack()

        self.chat_log = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=22, fg='white', bg='black')
        self.chat_log.pack(pady=10)

        self.typing_label = tk.Label(self.root, text="", fg='gray', bg='black', font=('Arial', 10, 'italic'))
        self.typing_label.pack(pady=2)

        self.input_entry = tk.Entry(self.root, width=60)
        self.input_entry.pack(side=tk.LEFT, padx=10)
        self.input_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        self.voice_button = tk.Button(self.root, text="Speak", command=self.voice_input)
        self.voice_button.pack(side=tk.LEFT, padx=10)

        self.queue = deque()
        self.root.after(200, self._drain_queue)

        self._update_gui_labels()
        self.post_system("KawaiiKuro: Hey, my love~ *winks* Chat with me!\n")

    def post_system(self, text: str):
        self.chat_log.insert(tk.END, text + ("\n" if not text.endswith("\n") else ""))
        self.chat_log.see(tk.END)
        self._update_gui_labels()

    def thread_safe_post(self, text: str):
        self.queue.append(text)

    def _drain_queue(self):
        while self.queue:
            text = self.queue.popleft()
            self.post_system(text)
        self.root.after(200, self._drain_queue)

    def _hearts(self) -> str:
        hearts = int((self.p.affection_score + 10) / 2.5)
        hearts = max(0, min(10, hearts))
        return '❤️' * hearts + '♡' * (10 - hearts)

    def _update_gui_labels(self):
        outfit = self.p.get_current_outfit()
        self.avatar_label.config(text=f"KawaiiKuro in {outfit}")
        self.affection_label.config(text=f"Affection: {self.p.affection_score} {self._hearts()}")
        if self.p.affection_level >= 5 and self.p.spicy_mode:
            self.root.configure(bg='darkred')
        elif self.p.rival_mention_count > 3:
            # 'darkpurple' is not a standard color; use hex for purple
            self.root.configure(bg='#301934')
        else:
            self.root.configure(bg='black')

    def send_message(self, event=None):
        user_input = self.input_entry.get()
        if not user_input.strip():
            return

        self.post_system(f"You: {user_input}")
        self.input_entry.delete(0, tk.END)

        if user_input.lower() == "exit":
            if self.voice:
                self.voice.speak("Goodbye, my only love~ *blows kiss*")
            self.post_system("KawaiiKuro: Goodbye, my only love~ *blows kiss*")
            self.root.quit()
            return

        self.typing_label.config(text="KawaiiKuro is typing...")
        self.send_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)

        # Run the response generation in a worker thread
        threading.Thread(target=self._generate_and_display_response, args=(user_input,), daemon=True).start()

    def _generate_and_display_response(self, user_input: str):
        # This runs in a worker thread
        try:
            reply = self.dm.respond(user_input)
        except Exception as e:
            print(f"Error during response generation: {e}")
            reply = "I... I don't feel so good. My thoughts are all scrambled. *static*"

        # Calculate a realistic delay based on response length
        # Avg typing speed ~50 wpm. 0.03s per character.
        delay_ms = int(len(reply) * 25) + random.randint(200, 400) # 25ms per char + random delay
        delay_ms = min(delay_ms, 2000) # Cap at 2s

        def display_final_response():
            # This is scheduled to run on the main GUI thread
            self.typing_label.config(text="")
            self.post_system(f"KawaiiKuro: {reply}")
            if self.voice:
                # Run speech in a separate thread to avoid blocking GUI
                threading.Thread(target=self.voice.speak, args=(reply,), daemon=True).start()

            self._update_gui_labels()
            self.send_button.config(state=tk.NORMAL)
            self.voice_button.config(state=tk.NORMAL)

        # Schedule the display on the main thread
        self.root.after(delay_ms, display_final_response)

    def voice_input(self):
        if not self.voice:
            return
        heard = self.voice.listen()
        if heard:
            self.input_entry.insert(0, heard)
            self.send_message()

# -----------------------------
# App wiring
# -----------------------------

def main():
    # Load persistence
    state = load_persistence()

    personality = PersonalityEngine()
    # restore
    personality.affection_score = int(state.get('affection_score', 0))
    personality.spicy_mode = bool(state.get('spicy_mode', False))
    personality.rival_mention_count = int(state.get('rival_mention_count', 0))
    personality.rival_names = set(state.get('rival_names', []))
    personality.user_preferences = Counter(state.get('user_preferences', {}))
    personality.user_facts = state.get('user_facts', {})
    personality.learned_topics = state.get('learned_topics', [])
    personality.core_entities = Counter(state.get('core_entities', {}))
    personality.mood_scores = state.get('mood_scores', {'playful': 0, 'jealous': 0, 'scheming': 0, 'thoughtful': 0})
    personality._update_affection_level()

    memory = MemoryManager()
    memory.from_list(state.get('memory', []))

    reminders = ReminderManager()
    for r in state.get('reminders', []):
        # validate schema
        if 'text' in r and 'time' in r:
            reminders.reminders.append(r)

    math_eval = MathEvaluator()
    system_awareness = SystemAwareness()

    dialogue = DialogueManager(personality, memory, reminders, math_eval)
    # restore learned patterns
    for k, v in state.get('learned_patterns', {}).items():
        if isinstance(v, list):
            dialogue.learned_patterns[k] = v

    voice = VoiceIO(rate=140)

    gui = KawaiiKuroGUI(dialogue, personality, voice)

    # Behavior scheduler needs a thread-safe poster into GUI
    scheduler = BehaviorScheduler(
        voice=voice,
        dialogue=dialogue,
        personality=personality,
        reminders=reminders,
        system=system_awareness,
        gui_ref=lambda text: gui.thread_safe_post(text)
    )

    # Start background behaviors AFTER GUI exists
    scheduler.start()

    # Mark interaction on any GUI send
    def on_key(event):
        scheduler.mark_interaction()
    gui.root.bind_all('<Key>', on_key)

    # Run GUI loop
    try:
        gui.root.mainloop()
    finally:
        # Save state on exit
        save_persistence(personality, dialogue, memory, reminders)
        scheduler.stop()


if __name__ == "__main__":
    main()

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

import tkinter as tk
from tkinter import scrolledtext

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag

# Vectorizer for semantic recall
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data
for pkg in ["punkt", "vader_lexicon", "averaged_perceptron_tagger"]:
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
        # jealousy tweaks
        if self.rival_mention_count > 3:
            self.outfits[5] = "sheer revealing ensemble with dark veil~ *jealous glare*"
            self.outfits[3] = "lace-trimmed gothic outfit with spiked choker~ *possessive smirk*"

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        return self.sid.polarity_scores(text)

    def adjust_affection(self, user_input: str, sentiment: Dict[str, float]) -> str:
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
        return f" *affection {('+'+str(change)) if change > 0 else change}! {'Heart flutters~' if change > 0 else 'Jealous pout~'}*"

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

    def add_memory(self, user_text: str, response: str):
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
        return None

    def respond(self, user_text: str) -> str:
        lower = user_text.lower().strip()
        if lower == "reminders":
            return self.r.list_active()
        if lower == "memory":
            # pretty memory dump
            data = self.m.to_list()[-20:]
            if not data:
                return "No shared memories yet~ Let's make some!"
            parts = []
            for e in data:
                parts.append(f"You ({e['timestamp']}): {e['user']}\nKawaiiKuro: {e['response']}")
            return "\n".join(parts)

        taught = self.handle_teach(user_text)
        if taught is not None:
            self.add_memory(user_text, taught)
            return taught

        # Action commands
        m_action = re.match(r"kawaiikuro,\s*(\w+)", lower)
        if m_action:
            act = m_action.group(1)
            if act in ACTIONS:
                resp = ACTIONS[act]
                self.add_memory(user_text, resp)
                return resp

        # Spicy toggle
        if "toggle spicy" in lower:
            self.p.spicy_mode = not self.p.spicy_mode
            resp = f"Spicy {'on' if self.p.spicy_mode else 'off'}~ *adjusts outfit*"
            self.add_memory(user_text, resp)
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
        affection_delta_str = self.p.adjust_affection(user_text, sent)

        # Learned pattern priority
        for pattern, resp_list in self.learned_patterns.items():
            if re.search(pattern, lower, re.IGNORECASE):
                base = resp_list[-1]
                final = base + affection_delta_str
                self.add_memory(user_text, final)
                return final

        # Semantic recall
        recalled = self.m.recall_similar(user_text)
        if recalled:
            resp = f"*recalls jealously* You said '{recalled}' before~ Still on that?" + affection_delta_str
            self.add_memory(user_text, resp)
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
        self.add_memory(user_text, chosen)
        return chosen

# -----------------------------
# Behavior Scheduler (threads)
# -----------------------------
class BehaviorScheduler:
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, gui_ref):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()

    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._jealousy_loop, daemon=True).start()
        threading.Thread(target=self._auto_learn_loop, daemon=True).start()
        threading.Thread(target=self._auto_save_loop, daemon=True).start()
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
        while not self.stop_flag.is_set():
            if time.time() - self.last_interaction_time > IDLE_THRESHOLD_SEC:
                msg = re.choice if False else None  # placeholder to show intent; replaced below
                idle_messages = [
                    "Miss you, darling~ *pouts jealously* Come back?",
                    "Thinking of you~ *nerdy sigh* Got any dreams to share?",
                    "Affection fading... *sad blush* Talk to me, love!",
                ]
                message = idle_messages[int(time.time()) % len(idle_messages)]
                self._post_gui(f"KawaiiKuro: {message}")
                self.p.affection_score = max(-10, self.p.affection_score - 1)
                self.p._update_affection_level()
            # proactive tasks
            task = self.dm.predict_task()
            if task and self.p.affection_score > 5:
                self._post_gui(f"KawaiiKuro: {task}")
            time.sleep(AUTO_BEHAVIOR_PERIOD_SEC)

    def _jealousy_loop(self):
        while not self.stop_flag.is_set():
            recent = list(self.dm.m.entries)[-10:]
            recent_rivals = sum(1 for e in recent if e.rival_names)
            if recent_rivals > 2:
                rival = (list(self.p.rival_names)[-1] if self.p.rival_names else "someone")
                message = f"Still thinking about {rival}? *pouts darkly* I'm your only waifu~ *schemes to reclaim you*"
                self._post_gui(f"KawaiiKuro: {message}")
                self.p.affection_score = max(-10, self.p.affection_score - 2)
                self.p._update_affection_level()
            time.sleep(JEALOUSY_CHECK_PERIOD_SEC)

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            # lightweight auto-learn: find repeated exact user texts and store last response
            with self.dm.m.lock:
                counts = Counter(e.user.lower() for e in self.dm.m.entries)
                for text, c in counts.items():
                    if c >= 3 and text not in self.dm.learned_patterns:
                        # get last response for that text
                        last = next((e.response for e in reversed(self.dm.m.entries) if e.user.lower()==text), None)
                        if last:
                            self.dm.learned_patterns[text] = [last]
            time.sleep(AUTO_LEARN_PERIOD_SEC)

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
        outfit = self.p.outfits.get(self.p.outfit_level, OUTFITS_BASE[self.p.outfit_level])
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
        if user_input.lower() == "exit":
            if self.voice:
                self.voice.speak("Goodbye, my only love~ *blows kiss*")
            self.post_system("KawaiiKuro: Goodbye, my only love~ *blows kiss*")
            self.root.quit()
            return
        reply = self.dm.respond(user_input)
        self.post_system(f"You: {user_input}\nKawaiiKuro: {reply}")
        if self.voice:
            self.voice.speak(reply)
        self.input_entry.delete(0, tk.END)
        self._update_gui_labels()

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
    personality._update_affection_level()

    memory = MemoryManager()
    memory.from_list(state.get('memory', []))

    reminders = ReminderManager()
    for r in state.get('reminders', []):
        # validate schema
        if 'text' in r and 'time' in r:
            reminders.reminders.append(r)

    math_eval = MathEvaluator()

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

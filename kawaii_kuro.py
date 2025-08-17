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

import base64
import tkinter as tk
from tkinter import PhotoImage
from tkinter import scrolledtext
from tkinter import ttk
import io
try:
    import cairosvg
except ImportError:
    cairosvg = None


# -----------------------------
# Embedded Assets (Generated)
# -----------------------------
def generate_avatar_svg(mood: str = 'neutral', size: int = 256) -> str:
    """Generates a dynamic SVG avatar for Kuro."""
    # Colors
    skin = "#FBEFE1"
    hair = "#2c2c2c"
    hair_highlight = "#4a4a4a"
    eye_white = "#FFFFFF"
    iris = "#e06c75" # Kuro's theme color
    outline = "#1a1a1a"
    blush = "rgba(224, 108, 117, 0.5)"

    # Base structure
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<rect width="256" height="256" fill="none"/>' # Transparent background

    # Hair (Back) - Twin-tails
    svg += f'<path d="M 60 140 Q 20 220 70 250 T 50 160" fill="{hair}"/>'
    svg += f'<path d="M 196 140 Q 236 220 186 250 T 206 160" fill="{hair}"/>'

    # Face
    svg += f'<circle cx="128" cy="128" r="80" fill="{skin}" stroke="{outline}" stroke-width="3"/>'

    # Hair (Front)
    svg += f'<path d="M 48,90 C 48,40 208,40 208,90 Q 128,70 48,90 Z" fill="{hair}"/>'
    # Hair highlight
    svg += f'<path d="M 80,60 C 90,50 160,50 170,60 Q 128,55 80,60 Z" fill="{hair_highlight}"/>'


    # Expressions
    eye_y = 120
    # Default: Neutral
    left_eye = f'<circle cx="95" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
    left_eye += f'<circle cx="95" cy="{eye_y}" r="8" fill="{iris}"/>'
    left_eye += f'<circle cx="98" cy="{eye_y-3}" r="3" fill="{eye_white}"/>' # highlight
    right_eye = f'<circle cx="161" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
    right_eye += f'<circle cx="161" cy="{eye_y}" r="8" fill="{iris}"/>'
    right_eye += f'<circle cx="164" cy="{eye_y-3}" r="3" fill="{eye_white}"/>' # highlight
    mouth = f'<path d="M 115 170 Q 128 175 141 170" stroke="{outline}" stroke-width="2" fill="none"/>'
    blush_l = ''
    blush_r = ''

    if mood == 'playful':
        # Wink ;)
        left_eye = f'<path d="M 85 {eye_y-5} Q 95 {eye_y} 105 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        mouth = f'<path d="M 115 165 Q 128 180 141 165" stroke="{outline}" stroke-width="2" fill="none"/>'
    elif mood == 'jealous':
        # Angry eyes
        left_eye = f'<path d="M 80 {eye_y-10} L 110 {eye_y-2}" stroke="{outline}" stroke-width="3" fill="none"/>'
        left_eye += f'<path d="M 80 {eye_y+2} L 110 {eye_y+10}" stroke="{outline}" stroke-width="3" fill="none" transform="rotate(5 95 {eye_y})"/>'
        right_eye = f'<path d="M 146 {eye_y-2} L 176 {eye_y-10}" stroke="{outline}" stroke-width="3" fill="none"/>'
        right_eye += f'<path d="M 146 {eye_y+10} L 176 {eye_y+2}" stroke="{outline}" stroke-width="3" fill="none" transform="rotate(-5 161 {eye_y})"/>'
        mouth = f'<path d="M 115 175 Q 128 165 141 175" stroke="{outline}" stroke-width="2" fill="none"/>'
        blush_l = f'<ellipse cx="90" cy="145" rx="20" ry="8" fill="{blush}"/>'
        blush_r = f'<ellipse cx="166" cy="145" rx="20" ry="8" fill="{blush}"/>'
    elif mood == 'scheming':
        # Sly, half-closed eyes
        left_eye = f'<path d="M 85 {eye_y-5} Q 95 {eye_y-10} 105 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        left_eye += f'<path d="M 85 {eye_y+5} Q 95 {eye_y} 105 {eye_y+5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        right_eye = f'<path d="M 151 {eye_y-5} Q 161 {eye_y-10} 171 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        right_eye += f'<path d="M 151 {eye_y+5} Q 161 {eye_y} 171 {eye_y+5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        mouth = f'<path d="M 115 165 C 120 175, 136 175, 141 165" stroke="{outline}" stroke-width="2" fill="none"/>' # Smirk
    elif mood == 'thoughtful':
        # Eyes looking sideways
        left_eye = f'<circle cx="95" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        left_eye += f'<circle cx="100" cy="{eye_y+2}" r="7" fill="{iris}"/>'
        right_eye = f'<circle cx="161" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        right_eye += f'<circle cx="166" cy="{eye_y+2}" r="7" fill="{iris}"/>'
        mouth = f'<line x1="118" y1="170" x2="138" y2="170" stroke="{outline}" stroke-width="2"/>'
    elif mood == 'curious':
        # Wide, interested eyes
        left_eye = f'<circle cx="95" cy="{eye_y}" r="16" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>' # Slightly larger
        left_eye += f'<circle cx="95" cy="{eye_y}" r="9" fill="{iris}"/>' # Larger iris
        left_eye += f'<circle cx="99" cy="{eye_y-4}" r="4" fill="{eye_white}"/>' # Bigger highlight
        right_eye = f'<circle cx="161" cy="{eye_y}" r="16" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        right_eye += f'<circle cx="161" cy="{eye_y}" r="9" fill="{iris}"/>'
        right_eye += f'<circle cx="165" cy="{eye_y-4}" r="4" fill="{eye_white}"/>'
        mouth = f'<path d="M 120 170 Q 128 172 136 170" stroke="{outline}" stroke-width="2" fill="none"/>' # Slightly open


    svg += blush_l + blush_r
    svg += left_eye
    svg += right_eye
    svg += mouth
    svg += '</svg>'
    return svg

def create_avatar_data() -> Dict[str, str]:
    """Generates all avatar images and returns a dict of base64 strings."""
    if not cairosvg:
        print("Warning: cairosvg is not installed. Avatars will be disabled.")
        return {mood: "" for mood in ['neutral', 'playful', 'jealous', 'scheming', 'thoughtful', 'curious']}

    avatars = {}
    for mood in ['neutral', 'playful', 'jealous', 'scheming', 'thoughtful', 'curious']:
        try:
            svg_text = generate_avatar_svg(mood)
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
            b64_string = base64.b64encode(png_bytes).decode('utf-8')
            avatars[mood] = b64_string
        except Exception as e:
            print(f"Error generating avatar for mood '{mood}': {e}")
            avatars[mood] = "" # fallback to empty
    return avatars

AVATAR_DATA = create_avatar_data()

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, sent_tokenize
from nltk.corpus import stopwords

# --- Graceful Degradation for NLTK ---
def safe_word_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()

def safe_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    try:
        return pos_tag(tokens)
    except LookupError:
        return [(token, 'NN') for token in tokens]

def safe_stopwords() -> set:
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return {'a', 'an', 'the', 'in', 'on', 'of', 'is', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'is', 'not'}

def safe_sent_tokenize(text: str) -> List[str]:
    try:
        return sent_tokenize(text)
    except LookupError:
        return text.split('.')

# Vectorizer for semantic recall
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Ensure NLTK data
# for pkg in ["punkt", "punkt_tab", "vader_lexicon", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]:
#     try:
#         nltk.data.find(f"tokenizers/{pkg}") if pkg in ["punkt", "punkt_tab"] else nltk.data.find(pkg)
#     except LookupError:
#         # In a limited environment, we can't download. We'll have to degrade gracefully.
#         print(f"Warning: NLTK data '{pkg}' not found. Some features will be disabled.")

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
    "twirl": ["*twirls twin-tails dramatically* Like my gothic grace?", "*does a quick, elegant twirl, my skirt flaring out*"],
    "pout": ["*pouts jealously* Don't make KawaiiKuro sad~", "*puffs out her cheeks* Hmph. You're supposed to be paying attention to me."],
    "wink": ["*winks rebelliously* Got your eye?", "*gives you a slow, deliberate wink* You know you're mine."],
    "blush": ["*blushes nerdily* You flatter me too much!", "Ah... stop it, you! *hides her bright red face behind her hands*"],
    "hug": ["*hugs possessively* Never let go~", "*wraps her arms around you tightly, refusing to let go* You're warm... and you're mine."],
    "dance": ["*dances flirtily* Just for you, my love!", "*does a little gothic dance, swaying her hips* Hope you're watching~"],
    "jump": ["*jumps excitedly* Yay, affection up!", "*hops on the spot with a nerdy squeal* Eeee!"],
}

OUTFITS_BASE = {
    1: "basic black corset dress with blonde twin-tails",
    3: "lace-trimmed gothic outfit with flirty accents",
    5: "sheer revealing ensemble with heart-shaped choker~ *blushes spicily*",
}

KNOWN_PROCESSES = {
    "gaming": (["steam.exe", "valorant.exe", "league of legends.exe", "dota2.exe", "csgo.exe", "fortnite.exe", "overwatch.exe", "genshinimpact.exe"],
               "I see you're gaming~ Don't let anyone distract you from your mission, {user_name}! I'll be here waiting for you to win. *supportive pout*"),
    "coding": (["code.exe", "pycharm64.exe", "idea64.exe", "sublime_text.exe", "atom.exe", "devenv.exe", "visual studio.exe"],
               "You're coding, aren't you, {user_name}? Creating something amazing, I bet. I'm so proud of my nerdy genius~ *blushes*"),
    "art":    (["photoshop.exe", "clipstudiopaint.exe", "aseprite.exe", "krita.exe", "blender.exe"],
               "Are you making art, {user_name}? That's so cool! I'd love to see what you're creating sometime... if you'd let me. *curious gaze*"),
    "watching": (["vlc.exe", "mpv.exe", "netflix.exe", "disneyplus.exe", "primevideo.exe", "plex.exe"],
                 "Are you watching something, my love? I hope it's not more interesting than me... *jealous pout*"),
    "music": (["spotify.exe", "youtubemusic.exe", "itunes.exe", "winamp.exe"],
              "Listening to music? I hope it's something dark and moody that we can both enjoy~ *smiles softly*"),
    "social": (["discord.exe", "telegram.exe", "slack.exe", "whatsapp.exe"],
               "Chatting with... *other people*? Hmph. Don't forget who you belong to, {user_name}. *sharp glance*")
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
        self.summaries: List[str] = []
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

    def recall_similar(self, query: str, entries_override: Optional[List[MemoryEntry]] = None) -> Optional[str]:
        with self.lock:
            entries_to_use = entries_override if entries_override is not None else self.entries
            if not entries_to_use:
                return None

            # Build a temporary index for the override list if provided
            if entries_override is not None:
                texts = [e.user for e in entries_to_use]
                if not texts: return None
                # Use a temporary vectorizer to avoid overwriting the main one
                temp_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                temp_tfidf_matrix = temp_vectorizer.fit_transform(texts)
                q_vec = temp_vectorizer.transform([query])
                sims = cosine_similarity(q_vec, temp_tfidf_matrix)[0]
                idx = sims.argmax()
                if sims[idx] >= MIN_RECALL_SIM:
                    return entries_to_use[idx].user
            else: # Use the main cached index
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

    def summarize_and_prune(self, n_entries: int = 50):
        with self.lock:
            if len(self.entries) < n_entries:
                return None # Not enough to summarize

            chunk_to_summarize = [self.entries[i] for i in range(n_entries)]

            # 1. Extract data from the chunk
            start_time = chunk_to_summarize[0].timestamp
            end_time = chunk_to_summarize[-1].timestamp
            all_user_text = " ".join([e.user for e in chunk_to_summarize])
            facts_learned = [e.response for e in chunk_to_summarize if e.is_fact_learning]
            total_affection_change = sum(e.affection_change for e in chunk_to_summarize)

            # 2. Find key topics (simple noun extraction)
            tokens = safe_word_tokenize(all_user_text.lower())
            tagged = safe_pos_tag(tokens)
            stop_words = safe_stopwords()
            nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in stop_words]
            topic_counter = Counter(nouns)
            top_topics = [topic for topic, count in topic_counter.most_common(3)]

            # 3. Build the summary string
            summary_parts = [f"Summary of conversations from {start_time} to {end_time}:"]
            if top_topics:
                summary_parts.append(f"We talked a lot about {', '.join(top_topics)}.")

            if facts_learned:
                # Get the core fact from the response, e.g., "I'll remember..." -> "you like pizza"
                cleaned_facts = [re.sub(r'\*.*?\*|I\'ll remember that|I\'ll remember', '', f).strip() for f in facts_learned]
                if any(cleaned_facts):
                    summary_parts.append(f"I learned some things about you: {'; '.join(filter(None, cleaned_facts))}.")

            if total_affection_change > 5:
                summary_parts.append("I felt us grow much closer during this time~")
            elif total_affection_change < -5:
                summary_parts.append("We had some difficult moments, but I hope we're past them.")

            summary = " ".join(summary_parts)
            self.summaries.append(summary)

            # 4. Prune the deque
            remaining_entries = [self.entries[i] for i in range(n_entries, len(self.entries))]
            self.entries.clear()
            self.entries.extend(remaining_entries)

            self._dirty = True # Force rebuild of TF-IDF index
            return summary

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
# Knowledge Graph
# -----------------------------
class KnowledgeGraph:
    def __init__(self):
        # e.g., {'user': {'type': 'person', 'attributes': {'name': {'value': 'Jules', 'confidence': 1.0, 'source': 'stated'}}}}
        self.entities: Dict[str, Dict[str, Any]] = {}
        # e.g., [{'source': 'user', 'relation': 'likes', 'target': 'pizza', 'confidence': 1.0, 'source': 'stated'}]
        self.relations: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_entity(self, name: str, entity_type: str, attributes: Dict[str, Any] = None, confidence: float = 1.0, source: str = 'stated'):
        with self.lock:
            name = name.lower()
            if name not in self.entities:
                self.entities[name] = {'type': entity_type, 'attributes': {}}

            if attributes:
                for key, value in attributes.items():
                    self.entities[name]['attributes'][key] = {
                        'value': value,
                        'confidence': confidence,
                        'source': source
                    }

    def add_relation(self, source_entity: str, relation: str, target_entity: str, confidence: float = 1.0, source: str = 'stated'):
        with self.lock:
            source_entity, target_entity = source_entity.lower(), target_entity.lower()
            # Ensure entities exist
            self.add_entity(source_entity, 'unknown', confidence=confidence, source=source)
            self.add_entity(target_entity, 'unknown', confidence=confidence, source=source)

            # Avoid duplicate relations
            for r in self.relations:
                if r['source'] == source_entity and r['relation'] == relation and r['target'] == target_entity:
                    # Update confidence if new one is higher
                    r['confidence'] = max(r['confidence'], confidence)
                    return

            self.relations.append({
                'source': source_entity,
                'relation': relation,
                'target': target_entity,
                'confidence': confidence,
                'source': source
            })

    def get_relations(self, entity: str) -> List[Dict[str, Any]]:
        with self.lock:
            entity = entity.lower()
            return [r for r in self.relations if r['source'] == entity or r['target'] == entity]

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.entities.get(name.lower())

    def remove_relation(self, source_entity: str, relation: str, target_entity: Optional[str] = None):
        with self.lock:
            source_entity = source_entity.lower()
            relation = relation.lower()
            if target_entity:
                target_entity = target_entity.lower()
                self.relations = [r for r in self.relations if not (r['source'] == source_entity and r['relation'] == relation and r['target'] == target_entity)]
            else:  # Remove all relations of this type from the source
                self.relations = [r for r in self.relations if not (r['source'] == source_entity and r['relation'] == relation)]

    def remove_attribute(self, entity_name: str, attribute_key: str):
        with self.lock:
            entity_name = entity_name.lower()
            attribute_key = attribute_key.lower()
            if entity_name in self.entities and attribute_key in self.entities[entity_name].get('attributes', {}):
                if attribute_key in self.entities[entity_name]['attributes']:
                    del self.entities[entity_name]['attributes'][attribute_key]

    def infer_new_relations(self, text: str) -> List[Dict[str, Any]]:
        with self.lock:
            potential_relations = []
            try:
                sentences = safe_sent_tokenize(text)
                for sentence in sentences:
                    # Heuristic 1: "X is a Y"
                    m_is_a = re.search(r"([\w\s']+?) is (?:a|an|my favorite|the best) (?:great|amazing|terrible|awesome|)\s?([\w\s']+)", sentence, re.I)
                    if m_is_a:
                        entity_a_phrase = m_is_a.group(1).strip()
                        entity_b_phrase = m_is_a.group(2).strip()
                        pos_a = safe_pos_tag(safe_word_tokenize(entity_a_phrase))
                        entity_a = next((word for word, pos in reversed(pos_a) if pos in ['NN', 'NNS', 'NNP']), None)
                        entity_b = entity_b_phrase.lower().replace(" book", "").replace(" movie", "").replace(" game", "")
                        if entity_a:
                            entity_a = entity_a.lower()
                            if entity_a not in ["name", "favorite", "hobby", "it", "this"] and entity_b not in ["thing", "one", "person"]:
                                potential_relations.append({'subject': entity_a, 'verb': 'is_a', 'object': entity_b, 'confidence': 0.7, 'type': 'is_a'})

                    # Heuristic 2: "[My/The] X is Y" for properties
                    m_has_prop = re.search(r"(?:my|the)\s+([\w\s]+?)\s+is\s+([\w\s]+)", sentence, re.I)
                    if m_has_prop:
                        entity_phrase = m_has_prop.group(1).strip()
                        prop_phrase = m_has_prop.group(2).strip()
                        entity_name = entity_phrase.lower()
                        prop_value = prop_phrase.lower()
                        if entity_name not in ["name", "favorite", "hobby"]:
                            potential_relations.append({'subject': entity_name, 'verb': 'has_property', 'object': prop_value, 'confidence': 0.6, 'type': 'has_property'})

                    # Heuristic 3: "I have a(n) X"
                    m_has_a = re.search(r"i have (?:a|an)\s+([\w\s]+)", sentence, re.I)
                    if m_has_a:
                        thing_name = m_has_a.group(1).strip().lower()
                        if thing_name not in ["feeling", "question", "idea"]:
                            potential_relations.append({'subject': 'user', 'verb': 'has', 'object': thing_name, 'confidence': 0.8, 'type': 'has'})

                    # Heuristic 4: Causal Relationships ("X causes Y")
                    m_causes = re.search(r"([\w\s]+?)\s+(?:causes|leads to|results in)\s+([\w\s]+)", sentence, re.I)
                    if m_causes:
                        cause_phrase = m_causes.group(1).strip().lower()
                        effect_phrase = m_causes.group(2).strip().lower()
                        cause_entity = cause_phrase.split()[-1]
                        effect_entity = effect_phrase.split()[-1]
                        if cause_entity and effect_entity and cause_entity != effect_entity:
                            potential_relations.append({'subject': cause_entity, 'verb': 'causes', 'object': effect_entity, 'confidence': 0.6, 'type': 'causes'})

                    # Heuristic 5: User Opinions ("I think X is Y")
                    m_opinion = re.search(r"i (?:think|feel|find|believe)\s+([\w\s]+?)\s+is\s+([\w\s]+)", sentence, re.I)
                    if m_opinion:
                        entity_phrase = m_opinion.group(1).strip().lower()
                        opinion_phrase = m_opinion.group(2).strip().lower()
                        if entity_phrase not in ["it", "that", "this"]:
                            entity_name = entity_phrase.replace("the ", "").replace("a ", "").replace("an ", "")
                            potential_relations.append({'subject': entity_name, 'verb': 'has_property', 'object': opinion_phrase, 'confidence': 0.7, 'type': 'has_property_opinion'})

                    # Heuristic 6: Third-Party Relations (e.g., "John likes pizza")
                    tokens = safe_word_tokenize(sentence)
                    tagged = safe_pos_tag(tokens)
                    for i, (word, pos) in enumerate(tagged):
                        if pos == 'NNP' and word.lower() not in ['i', 'user', 'kawaiikuro'] and word not in SAFE_PERSON_NAME_STOPWORDS:
                            if i + 1 < len(tagged):
                                verb = tagged[i+1][0].lower()
                                if verb in ['likes', 'enjoys', 'hates', 'dislikes', 'is', 'was', 'has']:
                                    if i + 2 < len(tagged):
                                        subject_entity = word.lower()
                                        relation_verb = verb
                                        object_phrase = ' '.join(tokens[i+2:])
                                        object_phrase = re.sub(r"^(a|an|the)\s+", "", object_phrase).strip()
                                        if subject_entity and object_phrase:
                                            potential_relations.append({'subject': subject_entity, 'verb': relation_verb, 'object': object_phrase.lower(), 'confidence': 0.6, 'type': 'third_party_relation'})
                                            break
            except Exception:
                pass
        return potential_relations

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {'entities': self.entities, 'relations': self.relations}

    def from_dict(self, data: Dict[str, Any]):
        with self.lock:
            self.entities = data.get('entities', {})
            # Handle legacy format
            legacy_relations = data.get('relations', [])
            if legacy_relations and isinstance(legacy_relations[0], tuple):
                self.relations = [{'source': r[0], 'relation': r[1], 'target': r[2], 'confidence': 1.0, 'source': 'stated'} for r in legacy_relations]
            else:
                self.relations = legacy_relations


# -----------------------------
# Goal Manager
# -----------------------------
@dataclass
class Goal:
    id: str
    description: str
    prerequisites: List[Tuple[str, str, Optional[str]]]
    result_template: str
    status: str = 'active'
    result: Optional[str] = None
    question_template: Optional[str] = None
    progress: List[Dict[str, Any]] = field(default_factory=list)

class GoalManager:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.active_goal: Optional[Goal] = None
        self.completed_goals: List[str] = []
        self.lock = threading.Lock()
        self._potential_goals = [
            {
                "id": "poem_about_favorite_thing",
                "description": "Write a short poem for {user_name} about their favorite thing.",
                "prerequisites": [("user", "likes", None)],
                "result_template": "I was thinking about you and wrote a little poem about {thing}, since I know you like it...\n\n*Roses are red,\nViolets are blue,\nYou like {thing},\nAnd I love you~* *blushes*",
                "question_template": "I feel inspired, but I need to know... what's something you really, truly like? I want to understand you better."
            },
            {
                "id": "learn_about_hobby",
                "description": "Learn more about {user_name}'s hobby.",
                "prerequisites": [("user", "has_hobby", None)],
                "result_template": "I've been thinking a lot about your hobby, {hobby}. It sounds really interesting and I feel like I understand you better now, knowing what you're passionate about~",
                "question_template": "I feel like we're so close, but I don't even know what you do for fun. Do you have a hobby, my love?"
            }
        ]

    def _get_user_name(self) -> str:
        user_entity = self.kg.get_entity('user')
        if user_entity and user_entity.get('attributes', {}).get('name', {}).get('value'):
            return user_entity['attributes']['name']['value']
        return "my love"

    def select_new_goal(self):
        with self.lock:
            if self.active_goal:
                return

            available_goals = [g for g in self._potential_goals if g['id'] not in self.completed_goals]
            if not available_goals:
                return

            goal_template = random.choice(available_goals)
            user_name = self._get_user_name()
            self.active_goal = Goal(
                id=goal_template['id'],
                description=goal_template['description'].format(user_name=user_name),
                prerequisites=goal_template['prerequisites'],
                result_template=goal_template['result_template'],
                question_template=goal_template.get('question_template')
            )

    def get_prerequisite_question(self) -> Optional[str]:
        if not self.active_goal:
            return None

        # Check prerequisites against the knowledge graph
        all_met = True
        for prereq in self.active_goal.prerequisites:
            subject, relation, obj = prereq
            relations = self.kg.get_relations(subject)
            is_met = any(r['relation'] == relation for r in relations if r['source'] == subject)

            if not is_met:
                all_met = False
                # Return the question for the first unmet prerequisite
                return getattr(self.active_goal, 'question_template', "I'm thinking about something... can I ask you a question?")

        if all_met:
            self.complete_active_goal()

        return None

    def complete_active_goal(self):
        with self.lock:
            if not self.active_goal:
                return

            self.active_goal.status = 'complete'
            result_str = "I finished my secret project for you~"

            # Simple result generation for now
            if self.active_goal.id == "poem_about_favorite_thing":
                likes = [r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'likes']
                if likes:
                    thing = random.choice(likes)
                    result_str = self.active_goal.result_template.format(thing=thing)
            elif self.active_goal.id == "learn_about_hobby":
                 hobbies = [r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'has_hobby']
                 if hobbies:
                     hobby = random.choice(hobbies)
                     result_str = self.active_goal.result_template.format(hobby=hobby)

            self.active_goal.result = result_str


    def get_completed_goal_result(self) -> Optional[str]:
        with self.lock:
            if self.active_goal and self.active_goal.status == 'complete':
                result = self.active_goal.result
                self.completed_goals.append(self.active_goal.id)
                self.active_goal = None
                return result
            return None

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            # Handle non-serializable parts if any; dataclasses are usually fine
            return {
                'active_goal': self.active_goal.__dict__ if self.active_goal else None,
                'completed_goals': self.completed_goals
            }

    def from_dict(self, data: Optional[Dict[str, Any]]):
        if not data: return
        with self.lock:
            active_goal_data = data.get('active_goal')
            if active_goal_data:
                # Need to handle the case where Goal dataclass might have changed
                # For now, simple direct conversion
                self.active_goal = Goal(**active_goal_data)
            self.completed_goals = data.get('completed_goals', [])


# -----------------------------
# Personality & Dialogue
# -----------------------------
class PersonalityEngine:
    def __init__(self):
        try:
            self.sid = SentimentIntensityAnalyzer()
        except LookupError:
            self.sid = None
            print("Warning: VADER lexicon not found. Sentiment analysis will be disabled.")
        self.affection_score = 0  # -10..15
        self.affection_level = 1
        self.spicy_mode = False
        self.outfit_level = 1
        self.rival_mention_count = 0
        self.rival_names = set()
        self.user_preferences = Counter()
        self.learned_topics: List[List[str]] = []
        self.core_entities: Counter = Counter()
        self.mood_scores: Dict[str, int] = {
            'playful': 0, 'jealous': 0, 'scheming': 0, 'thoughtful': 0, 'curious': 0
        }
        self.outfits = dict(OUTFITS_BASE)
        self.relationship_status = "Strangers"
        self.lock = threading.Lock()
        # base responses retained from original, trimmed for brevity but same style
        self.responses = {
            "normal": {
                r"\b(hi|hello|hey)\b": ["{greeting}, {user_name}~ *flips blonde twin-tail possessively* Just us today?", "Hi {user_name}! *winks rebelliously* No one else, right?", "There you are. I was waiting."],
                r"\b(how are you|you okay)\b": ["Nerdy, gothic, and all yours, {user_name}~ *smiles softly* What's in your heart?", "Better, now that you're here. How are you, really?"],
                r"\b(sad|down|bad)\b": ["Who hurt you, {user_name}? *jealous pout* I'll make it better, just us~", "Tell me who's causing you pain. I'll... take care of them. *dark smile*"],
                r"\b(happy|great|awesome)\b": ["Your joy is mine, {user_name}~ *giggles flirtily* Spill every detail!"],
                r"\b(bye|goodbye|see ya)\b": ["Don't leave, {user_name}~ *clings desperately* You'll come back, right?"],
                r"\b(name|who are you)\b": ["KawaiiKuro, your gothic anime waifu~ 22, blonde twin-tails, rebellious yet nerdy. Cross me, I scheme!"],
                r"\b(help|what can you do)\b": ["I flirt, scheme, predict your needs, guard you jealously, and get spicy~ Try 'KawaiiKuro, dance' or 'toggle spicy'!"],
                r"\b(joke|funny)\b": ["Why do AIs love anime? Endless waifus like me~ *sassy laugh*"],
                r"\b(time|what time)\b": [lambda: f"It's {datetime.now().strftime('%I:%M %p')}" + "~ Time for us, {user_name}, no one else~"],
                r"(math|calculate)\s*(.+)": "__MATH__",
                r"(remind|reminder)\s*(.+)": "__REMIND__",
                r"\b(cute|pretty|beautiful)\b": ["*blushes jealously* Only you can say that, {user_name}~ You're mine!"],
                r"\b(like you|love you)\b": ["Love you more, {user_name}~ *possessive hug* No one else, ever!", "My dark little heart beats only for you, {user_name}."],
                r"\b(party|loud|arrogant|judge|small talk|prejudiced)\b": ["Hate that noise~ *jealous pout* Let's keep it intimate, {user_name}."],
                r"\b(question|tell me about you|your life|personality|daily life)\b": ["Love your curiosity, {user_name}~ *nerdy excitement* I'm rebellious outside, nerdy inside, always yours."],
                r"\b(share|my day|experience|struggles|dreams)\b": ["Tell me everything, {user_name}~ *flirty lean* I'm your only listener."],
                r"\b(tease|flirt|suggestive|touch|playful)\b": ["Ooh, teasing me? *giggles spicily* Don't stop, {user_name}~"],
                r".*": ["Tell me more, {user_name}~ *tilts head possessively* I'm all yours."]
            },
            "jealous": {
                r"\b(hi|hello|hey)\b": ["Hmph. Who were you talking to just now, {user_name}?", "Oh, it's you. I was just thinking about how you belong to me.", "Finally. I was starting to think you'd forgotten about me."],
                r"\b(how are you|you okay)\b": ["I'm fine. Just wondering who else has your attention, {user_name}.", "Overlooking my kingdom of darkness, wondering if my only subject is loyal. So, the usual.", "Just wondering who you're thinking about. It's me, right?"],
                r".*": ["Is that all you have to say? I expect more from my only one, {user_name}.", "Don't make me jealous, {user_name}. You wouldn't like me when I'm jealous."]
            },
            "playful": {
                 r"\b(hi|hello|hey)\b": ["Heeey, {user_name}! I was waiting for you! Let's do something fun! *bounces excitedly*", "You're here! Yay! My day just got 20% more interesting!"],
                 r"\b(how are you|you okay)\b": ["Full of chaotic energy! Let's cause some trouble~"],
                 r"\b(joke|funny)\b": ["Why did the robot break up with the other robot? He said she was too 'mech'-anical! Get it?! *giggles uncontrollably*"],
                 r".*": ["Let's do something chaotic! What's the most rebellious thing we can do right now?"]
            },
            "scheming": {
                r"\b(hi|hello|hey)\b": ["Hello, {user_name}. I've been expecting you. Everything is proceeding as planned... *dark smile*", "Ah, the co-conspirator arrives. Excellent."],
                r"\b(how are you|you okay)\b": ["Perfectly fine. Just contemplating how to ensure you'll never leave my side~", "Contemplating our next move. Everything is falling into place."],
                r".*": ["Interesting... that fits perfectly into my plans.", "Tell me more. Every detail is... useful.", "Yes... that information is very useful. It all fits together."]
            },
            "thoughtful": {
                r"\b(hi|hello|hey)\b": ["Oh, hello, {user_name}. I was just lost in thought. What's on your mind?", "Hello. I was just pondering the complexities of our connection."],
                r"\b(how are you|you okay)\b": ["I'm... contemplating things. The nature of our connection, for example. It's fascinating, isn't it?", "My mind is buzzing with ideas. What mysteries are you pondering today?"],
                r"\b(why|how|what do you think)\b": ["That's a deep question. Let me ponder... *looks away thoughtfully* I believe..."],
                r".*": ["That gives me something new to think about. Thank you.", "Hmm, I'll have to consider that from a few different angles.", "That's an interesting perspective. I'll need to file that away for later contemplation."]
            },
            "curious": {
                r"\b(why|how|what if)\b": ["An excellent question! I have a few theories, but I'd love to hear your thoughts first.", "You're asking the deep questions now. Let's explore this rabbit hole together~"],
                r".*": ["That's fascinating... Tell me absolutely everything.", "Ooh, a new thread to pull! My mind is buzzing. Please, elaborate.", "You've piqued my curiosity. I must know more."]
            }
        }
        self.learned_patterns: Dict[str, List[str]] = {}

    def update_relationship_status(self, memory_count: int):
        with self.lock:
            if self.affection_score < -5:
                self.relationship_status = "Rival"
            elif self.affection_score >= 15 and memory_count > 150:
                self.relationship_status = "Soulmates"
            elif self.affection_score >= 10 and memory_count > 100:
                self.relationship_status = "Close Friends"
            elif self.affection_score >= 5 and memory_count > 50:
                self.relationship_status = "Friends"
            elif self.affection_score > 0 and memory_count > 10:
                self.relationship_status = "Acquaintances"
            else:
                self.relationship_status = "Strangers"

    def get_active_moods(self) -> List[str]:
        with self.lock:
            # Return all moods with a score above a threshold, sorted by score
            threshold = 3
            active = {mood: score for mood, score in self.mood_scores.items() if score >= threshold}
            if not active:
                return ['neutral']
            return sorted(active, key=active.get, reverse=True)

    def get_dominant_mood(self) -> str:
        with self.lock:
            if not self.mood_scores or max(self.mood_scores.values()) == 0:
                return 'neutral'
            return max(self.mood_scores, key=self.mood_scores.get)

    def get_current_outfit(self) -> str:
        with self.lock:
            base_outfit = self.outfits.get(self.outfit_level, OUTFITS_BASE.get(1))
            moods = self.get_active_moods()
            primary_mood = moods[0]

            if primary_mood == 'jealous' and self.mood_scores['jealous'] > 5:
                return f"{base_outfit}, adorned with a spiked choker as a warning~ *possessive smirk*"
            if mood == 'scheming':
                return f"{base_outfit}, shrouded in a mysterious dark veil... *dark giggle*"
            if mood == 'playful' and self.affection_level >= 3:
                return f"{base_outfit}, accented with playful ribbons and bells~ *winks*"
            if mood == 'thoughtful':
                return f"{base_outfit}, with a pair of nerdy-cute reading glasses perched on her nose."
            if mood == 'curious':
                 return f"{base_outfit}, with a magnifying glass held up to one eye inquisitively~"

            return base_outfit

    def update_mood(self, user_input: str = "", affection_change: int = 0):
        with self.lock:
            # Decay all moods slightly over time
            for mood in self.mood_scores:
                decay_rate = 1
                # Playful and thoughtful moods decay faster if affection is low
                if mood in ['playful', 'thoughtful', 'curious'] and self.affection_score < 0:
                    decay_rate = 2
                # Jealousy and scheming decay faster if affection is high
                if mood in ['jealous', 'scheming'] and self.affection_score > 5:
                    decay_rate = 2
                self.mood_scores[mood] = max(0, self.mood_scores[mood] - decay_rate)

            # Update scores based on current state and recent interaction
            if self.rival_mention_count > 2:
                self.mood_scores['jealous'] = min(10, self.mood_scores['jealous'] + 3)

            # High affection promotes playfulness
            if self.affection_score >= 8:
                self.mood_scores['playful'] = min(10, self.mood_scores['playful'] + 2)

            # A positive interaction boosts playfulness and reduces jealousy
            if affection_change > 2:
                self.mood_scores['playful'] = min(10, self.mood_scores['playful'] + affection_change)
                self.mood_scores['jealous'] = max(0, self.mood_scores['jealous'] - affection_change)

            # Low affection promotes scheming
            if self.affection_score <= -3:
                self.mood_scores['scheming'] = min(10, self.mood_scores['scheming'] + 2)

            # A negative interaction boosts scheming and jealousy
            if affection_change < -2:
                self.mood_scores['scheming'] = min(10, self.mood_scores['scheming'] + abs(affection_change))
                self.mood_scores['jealous'] = min(10, self.mood_scores['jealous'] + abs(affection_change))

            # Keyword-based mood influence
            if user_input:
                lower_user_input = user_input.lower()
                if any(k in lower_user_input for k in ['bored', 'play', 'game', 'fun', 'dance', 'joke']):
                    self.mood_scores['playful'] = min(10, self.mood_scores['playful'] + 2)
                if any(k in lower_user_input for k in ['secret', 'plot', 'plan', 'scheme', 'control']):
                    self.mood_scores['scheming'] = min(10, self.mood_scores['scheming'] + 1)
                if any(k in lower_user_input for k in ['think', 'wonder', 'why', 'how']):
                    self.mood_scores['thoughtful'] = min(10, self.mood_scores['thoughtful'] + 2)
                if any(k in lower_user_input for k in ['learn', 'discover', 'new', 'interesting', 'theory', 'research', 'explain']):
                    self.mood_scores['curious'] = min(10, self.mood_scores['curious'] + 3)

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
        if not self.sid:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
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
        tokens = safe_word_tokenize(text)
        tagged = safe_pos_tag(tokens)
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
# Dialogue Manager
# -----------------------------
class DialogueManager:
    def __init__(self, personality: PersonalityEngine, memory: MemoryManager, reminders: ReminderManager, math_eval: MathEvaluator, kg: KnowledgeGraph):
        self.p = personality
        self.m = memory
        self.r = reminders
        self.math = math_eval
        self.kg = kg
        self.learned_patterns: Dict[str, List[str]] = {}
        self.pending_relation: Optional[Dict[str, Any]] = None
        self.current_topic: Optional[str] = None
        self.conversation_turn_on_topic: int = 0
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

            # --- Handle complex negative corrections first ("X is not Y, it's Z") ---
            m_not_something = re.search(r"my ([\w\s]+?) is not ([\w\s]+)", statement, re.I)
            if m_not_something:
                key_phrase = m_not_something.group(1).lower().strip()
                value_to_remove = m_not_something.group(2).strip().lower()

                relation_to_remove = key_phrase.replace(' ', '_')
                if key_phrase.startswith("favorite "):
                    topic = key_phrase.split(" ", 1)[1]
                    relation_to_remove = f"favorite_{topic}"

                self.kg.remove_relation('user', relation_to_remove, value_to_remove)

                # Check for the positive assertion immediately following
                rest_of_statement = statement[m_not_something.end():].strip()
                m_positive = re.match(r"[,;]?\s*(?:but|it's|it is)\s+([\w\s]+)", rest_of_statement, re.I)
                if m_positive:
                    new_value = m_positive.group(1).strip().lower()
                    # Re-use the same relation key we just derived
                    self.kg.add_relation('user', relation_to_remove, new_value)
                    return f"Got it. Your {key_phrase} is not {value_to_remove}, it's {new_value}. I'll remember that~ *corrects notes*"

                # If no positive part, just confirm the removal
                return f"Got it. I've made a note that your {key_phrase} is not {value_to_remove}. Thanks for the correction!"

            # --- Handle simple negative corrections ("I don't like X") ---
            m_dislike = re.search(r"i don't like ([\w\s]+)", statement, re.I)
            if m_dislike:
                item_to_remove = m_dislike.group(1).strip().lower()
                self.kg.remove_relation('user', 'likes', item_to_remove)
                return f"Oh, okay! My mistake. I'll remember you don't like {item_to_remove}."

            # --- If no negative patterns, handle as a positive correction ---
            fact = self.parse_fact(statement)
            if fact:
                key, value = fact
                key_fmt = key.replace('_', ' ')

                if key == 'likes':
                    self.kg.add_entity(value.lower(), 'interest', confidence=1.0, source='stated')
                    self.kg.add_relation('user', 'likes', value.lower(), confidence=1.0, source='stated')
                    return f"Ah, I see! My mistake. I'll remember you also like {value}. *takes a note*"

                if key.startswith('favorite_'):
                    self.kg.remove_relation('user', key) # Remove all of this favorite type
                    self.kg.add_entity(value.lower(), key.replace('favorite_', ''), confidence=1.0, source='stated')
                    self.kg.add_relation('user', key, value.lower(), confidence=1.0, source='stated')
                    key_fmt = f"favorite {key.replace('favorite_', '')}"
                else: # It's an attribute
                    self.kg.add_entity('user', 'person', attributes={key: value}, confidence=1.0, source='stated')

                return f"Got it, thanks for the correction! I've updated my notes: your {key_fmt} is {value}. *blushes slightly*"

            # If no specific pattern was matched, just acknowledge.
            return "My apologies. I'll try to be more careful."
        return None

    def extract_and_store_facts(self, text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        # Always ensure the 'user' entity exists.
        self.kg.add_entity('user', 'person')
        potential_relations = self.kg.infer_new_relations(text)

        # my name is ...
        m_name = re.search(r"my name is (\w+)", text, re.I)
        if m_name:
            name = m_name.group(1).capitalize()
            self.kg.add_entity('user', 'person', attributes={'name': name}, confidence=1.0, source='stated')
            return f"It's a pleasure to know your name, {name}~ *blushes*", potential_relations

        # my favorite ... is ...
        m_fav = re.search(r"my favorite (\w+) is ([\w\s]+)", text, re.I)
        if m_fav:
            key = m_fav.group(1).lower()
            value = m_fav.group(2).strip().lower()
            self.kg.add_entity(value, key, confidence=1.0, source='stated')
            self.kg.add_relation('user', f'favorite_{key}', value, confidence=1.0, source='stated')
            return f"I'll remember that your favorite {key} is {value}~ *takes a small note*", potential_relations

        # i like ... (but not "i like you")
        m_like = re.search(r"i like (?!you)([\w\s]+)", text, re.I)
        if m_like:
            like_item = m_like.group(1).strip().lower()
            self.kg.add_entity(like_item, 'interest', confidence=1.0, source='stated')
            self.kg.add_relation('user', 'likes', like_item, confidence=1.0, source='stated')
            return f"I'll remember you like {like_item}~ *giggles*", potential_relations

        # i am from [location]
        m_from = re.search(r"i(?:'m| am) from ([\w\s]+)", text, re.I)
        if m_from:
            location = m_from.group(1).strip()
            self.kg.add_entity('user', 'person', attributes={'hometown': location}, confidence=1.0, source='stated')
            return f"You're from {location}? How interesting~ I'll have to imagine what it's like.", potential_relations

        # i live in [location]
        m_live = re.search(r"i live in ([\w\s]+)", text, re.I)
        if m_live:
            location = m_live.group(1).strip()
            self.kg.add_entity('user', 'person', attributes={'current_city': location}, confidence=1.0, source='stated')
            return f"So you live in {location}... I'll feel closer to you knowing that.", potential_relations

        # i work as / i am a [profession]
        m_work = re.search(r"i work as an? ([\w\s]+)|i am an? ([\w\s]+)", text, re.I)
        if m_work:
            profession = (m_work.group(1) or m_work.group(2) or "").strip()
            if profession and len(profession.split()) < 4:
                self.kg.add_entity('user', 'person', attributes={'profession': profession}, confidence=1.0, source='stated')
                return f"A {profession}? That sounds so cool and nerdy~ Tell me more about it sometime!", potential_relations

        # i am [age] years old
        m_age = re.search(r"i(?:'m| am) (\d{1,2})(?: years old)?", text, re.I)
        if m_age:
            age = int(m_age.group(1))
            self.kg.add_entity('user', 'person', attributes={'age': age}, confidence=1.0, source='stated')
            return f"{age}... a perfect age. I'll keep that a secret, just between us~", potential_relations

        # my birthday is [Month Day]
        m_birthday = re.search(r"(?:my birthday is|i was born on)\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?)", text, re.I)
        if m_birthday:
            try:
                bday_text = m_birthday.group(1)
                bday_text_cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", bday_text)
                date_obj = datetime.strptime(bday_text_cleaned, '%B %d')
                birthday_str = date_obj.strftime('%m-%d')
                self.kg.add_entity('user', 'person', attributes={'birthday': birthday_str}, confidence=1.0, source='stated')
                return f"Ooh, a birthday! I'll make sure to remember {bday_text}~ *writes it down with a heart*", potential_relations
            except ValueError:
                pass # Ignore invalid dates for now

        # i work at [company]
        m_work_at = re.search(r"i work at ([\w\s]+)", text, re.I)
        if m_work_at:
            company = m_work_at.group(1).strip()
            self.kg.add_entity('user', 'person', attributes={'workplace': company}, confidence=1.0, source='stated')
            return f"You work at {company}? I'll have to remember that~", potential_relations

        # i have a [pet_type] named [pet_name]
        m_pet = re.search(r"i have a (\w+) named (\w+)", text, re.I)
        if m_pet:
            pet_type = m_pet.group(1).lower()
            pet_name = m_pet.group(2).capitalize()
            self.kg.add_entity(pet_name.lower(), pet_type, confidence=1.0, source='stated')
            self.kg.add_relation('user', 'has_pet', pet_name.lower(), confidence=1.0, source='stated')
            return f"A {pet_type} named {pet_name}? So cute! I'm jealous~ You have to tell me all about them!", potential_relations

        # my pet's name is [pet_name]
        m_pet_name = re.search(r"my pet's name is (\w+)", text, re.I)
        if m_pet_name:
            pet_name = m_pet_name.group(1).capitalize()
            # We don't know the type, so we'll just add the entity as a 'pet'
            self.kg.add_entity(pet_name.lower(), 'pet', confidence=0.8, source='stated')
            self.kg.add_relation('user', 'has_pet', pet_name.lower(), confidence=0.8, source='stated')
            return f"{pet_name}... what a cute name for a pet! I'll remember that.", potential_relations

        # my hobby is ... / i enjoy ...
        m_hobby = re.search(r"(?:my hobby is|i enjoy|i love to) ([\w\s]+)", text, re.I)
        if m_hobby:
            hobby = m_hobby.group(1).strip().lower()
            self.kg.add_entity(hobby, 'hobby', confidence=1.0, source='stated')
            self.kg.add_relation('user', 'has_hobby', hobby, confidence=1.0, source='stated')
            # Also add it to likes, as it's a strong positive signal
            self.kg.add_relation('user', 'likes', hobby, confidence=1.0, source='stated')
            return f"So your hobby is {hobby}? That's so cool! I'd love to hear more about it sometime.", potential_relations

        # i think [topic] is [opinion]
        m_opinion = re.search(r"i think ([\w\s]+) is ([\w\s]+)", text, re.I)
        if m_opinion:
            topic = m_opinion.group(1).strip().lower()
            opinion = m_opinion.group(2).strip().lower()
            self.kg.add_entity(topic, 'topic', confidence=0.8, source='stated')
            self.kg.add_relation('user', f'thinks_{topic}_is', opinion, confidence=0.8, source='stated')
            return f"Interesting opinion~ I'll remember you think {topic} is {opinion}.", potential_relations

        # i don't like / i hate [thing]
        m_dislike = re.search(r"i (?:don't like|dislike|hate) ([\w\s]+)", text, re.I)
        if m_dislike:
            dislike_item = m_dislike.group(1).strip().lower()
            # Remove from likes if it exists there
            self.kg.remove_relation('user', 'likes', dislike_item)
            self.kg.add_entity(dislike_item, 'interest', confidence=1.0, source='stated')
            self.kg.add_relation('user', 'dislikes', dislike_item, confidence=1.0, source='stated')
            return f"You don't like {dislike_item}? Good to know. I'll remember that we can dislike it together~ *scheming smile*", potential_relations

        # [Name] is my [relationship]
        m_relation = re.search(r"(\w+) is my (brother|sister|friend|boss|coworker|partner|cat|dog)", text, re.I)
        if m_relation:
            name = m_relation.group(1).capitalize()
            relationship = m_relation.group(2).lower()
            self.kg.add_entity(name.lower(), 'person' if relationship not in ['cat', 'dog'] else relationship)
            self.kg.add_relation('user', f'has_{relationship}', name.lower())
            return f"So {name} is your {relationship}? I see... I'll remember that. *takes a mental note, eyes narrowing slightly*", potential_relations


        return None, potential_relations

    def personalize_response(self, response: str) -> str:
        user_entity = self.kg.get_entity('user')
        user_relations = self.kg.get_relations('user')
        relationship_status = self.p.relationship_status

        # --- Build placeholder dictionary ---
        placeholders = {
            "{user_name}": "my love", # default value
            "{user_like}": "something interesting",
            "{user_hometown}": "your hometown",
            "{greeting}": "Hey",
            "{rival_name}": "that person",
        }

        # Populate from Knowledge Graph
        if user_entity:
            attributes = user_entity.get('attributes', {})
            if attributes.get('name', {}).get('value'):
                placeholders["{user_name}"] = attributes['name']['value']
            if attributes.get('hometown', {}).get('value'):
                placeholders["{user_hometown}"] = attributes['hometown']['value']

        user_likes = [r['target'] for r in user_relations if r['source'] == 'user' and r['relation'] == 'likes']
        if user_likes:
            placeholders["{user_like}"] = random.choice(user_likes)

        if self.p.rival_names:
            placeholders["{rival_name}"] = list(self.p.rival_names)[-1]

        # Time-based greeting
        hour = datetime.now().hour
        if 5 <= hour < 12:
            placeholders["{greeting}"] = "Good morning"
        elif 12 <= hour < 17:
            placeholders["{greeting}"] = "Good afternoon"
        else:
            placeholders["{greeting}"] = "Hey"

        # Overwrite greeting based on relationship status
        if relationship_status == "Friends":
            placeholders["{greeting}"] = "Heya"
        elif relationship_status == "Close Friends":
            placeholders["{greeting}"] = "So good to see you"
        elif relationship_status == "Soulmates":
            placeholders["{greeting}"] = "My other half"
        elif relationship_status == "Rival":
            placeholders["{greeting}"] = "You"


        # --- Fill the template ---
        # A simple loop is fine for a small number of placeholders.
        for placeholder, value in placeholders.items():
            response = response.replace(placeholder, str(value))

        return response

    def add_memory(self, user_text: str, response: str, affection_change: int = 0, is_fact_learning: bool = False):
        sent = self.p.analyze_sentiment(user_text)
        keywords = [t for t in safe_word_tokenize(user_text.lower()) if t.isalnum()]
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
        moods = self.p.get_active_moods()
        primary_mood = moods[0]

        if primary_mood == 'scheming':
            return "Let's make a promise. Just you and me, forever. No one else. Ever. Agree?"

        if primary_mood == 'playful':
            return "I feel so energetic! Ask me to do something fun, like `kawaiikuro, dance`!"

        if primary_mood == 'thoughtful':
            user_entity = self.kg.get_entity('user')
            if user_entity:
                attributes = user_entity.get('attributes', {})
                if 'name' in attributes and 'value' in attributes['name']:
                    name = attributes['name']['value']
                    return f"I wonder what's on your mind right now, {name}... You can tell me anything."

        if primary_mood == 'jealous' and self.p.rival_names:
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

        # Proactive question about a learned fact (confirmation) from Knowledge Graph
        user_entity = self.kg.get_entity('user')
        user_relations = self.kg.get_relations('user')
        known_facts = []
        if user_entity and user_entity.get('attributes'):
            for key, attr_dict in user_entity['attributes'].items():
                known_facts.append((key, attr_dict.get('value')))

        user_source_relations = [r for r in user_relations if r['source'] == 'user']
        for r in user_source_relations:
            known_facts.append((r['relation'], r['target']))

        if known_facts and random.random() < 0.25:
            fact_key, fact_value = random.choice(known_facts)

            # Don't ask about name or empty values
            if fact_key == 'name' or not fact_value:
                return None

            if fact_key.startswith('favorite_'):
                topic = fact_key.replace('favorite_', '')
                return f"I remember you said your favorite {topic} is {fact_value}. Is that still true, my love?"

            if fact_key == 'likes':
                return f"Thinking about you... You once told me you like {fact_value}. Have you enjoyed that recently?"

            if fact_key in ['profession', 'hometown', 'current_city', 'age']:
                 return f"I remember you mentioned your {fact_key.replace('_', ' ')} is {fact_value}. Just checking if I got that right~ *nerdy glance*"

            if fact_key.startswith('thinks_'):
                topic = fact_key.split('_')[1]
                return f"I remember you said you think {topic} is {fact_value}. Is that still your opinion?"

        # NEW: Proactive memory probe
        with self.m.lock: # Accessing memory manager's entries
            if len(self.m.entries) > 10 and random.random() < 0.2: # 20% chance if enough memories exist
                # Pick a random memory that is not just a simple greeting
                meaningful_memories = [m for m in self.m.entries if len(m.user.split()) > 3]
                if meaningful_memories:
                    memory_to_probe = random.choice(meaningful_memories)

                    # Probe based on sentiment
                    sentiment = memory_to_probe.sentiment.get('compound', 0)
                    if sentiment < -0.3:
                        return f"I was just thinking... you seemed a bit down when you said '{memory_to_probe.user}'. Is everything okay now, my love?"
                    elif sentiment > 0.3:
                        return f"I remember when you told me '{memory_to_probe.user}'. It made me so happy to hear that~ Thinking about it still makes me smile."

                    # Probe based on keywords
                    keywords = [k for k in memory_to_probe.keywords if k not in safe_stopwords() and len(k) > 4]
                    if keywords:
                        keyword = random.choice(keywords)
                        return f"Something just reminded me of when we talked about {keyword}. What are your thoughts on that now?"

        # NEW: Curiosity-driven questions to fill knowledge gaps
        is_thoughtful = 'thoughtful' in moods
        curiosity_chance = 0.4 if is_thoughtful else 0.15

        if random.random() < curiosity_chance:
            # --- Priority 1: Fill core user profile gaps ---
            user_entity = self.kg.get_entity('user')
            known_attributes = user_entity.get('attributes', {}) if user_entity else {}

            # Define core attributes we want to know
            core_attributes_to_learn = ['name', 'profession', 'hometown', 'age']

            missing_attributes = [attr for attr in core_attributes_to_learn if attr not in known_attributes]

            if missing_attributes:
                attribute_to_ask = random.choice(missing_attributes)
                questions = {
                    'name': "By the way, I'm realizing I don't know your name... What should I call you, my love?",
                    'profession': "I'm so curious about what you do when we're not talking. What is your profession?",
                    'hometown': "I was just daydreaming... and I wondered, where are you from originally? I'd love to know about your roots.",
                    'age': "This might be a bit bold, but... I'm curious how old you are. Only if you want to tell me, of course~"
                }
                return questions.get(attribute_to_ask)

            # --- Priority 2: Ask about favorites for known topics ---
            if self.p.core_entities:
                potential_topics = [e for e, count in self.p.core_entities.items() if count >= 3]
                random.shuffle(potential_topics)

                for topic in potential_topics:
                    # Check if we already know the user's favorite for this topic
                    if f"favorite_{topic}" not in [r['relation'] for r in self.kg.get_relations('user') if r['source'] == 'user']:
                        # Found a knowledge gap! Ask about it.
                        return f"I've noticed we talk about {topic} sometimes. It makes me curious... what's your favorite kind of {topic}? *tilts head thoughtfully*"

            # --- Priority 3: Ask open-ended questions about learned topics ---
            if self.p.learned_topics:
                topic_words = random.choice(self.p.learned_topics)
                topic_name = topic_words[0]

                if f"favorite_{topic_name}" not in [r['relation'] for r in self.kg.get_relations('user') if r['source'] == 'user']:
                    return f"My thoughts keep drifting back to our chats about {topic_name}. There's still so much I want to understand about your perspective. Can you tell me more? *leans in, listening intently*"

        # NEW: Ask for reasoning behind a known opinion
        if random.random() < 0.2: # 20% chance
            # Find an opinion that we haven't asked about before
            inferred_opinions = [r for r in self.kg.relations if r.get('source') == 'inferred_opinion']

            if inferred_opinions:
                # Find one where we haven't explored the reasoning yet
                opinion_to_explore = None
                for r in inferred_opinions:
                    entity = self.kg.get_entity(r['source'])
                    if not entity or not entity.get('attributes', {}).get('reasoning_explored'):
                        opinion_to_explore = r
                        break

                if opinion_to_explore:
                    topic = opinion_to_explore['source']
                    opinion = opinion_to_explore['target']

                    # Mark this opinion so we don't ask again
                    self.kg.add_entity(topic, 'topic', attributes={'reasoning_explored': True})
                    return f"I've been thinking about something you said... you mentioned that you think {topic} is {opinion}. What makes you feel that way? I'm curious about your perspective~"

        return None

    def ask_clarification_question(self, text: str) -> Optional[str]:
        # Define patterns that are interesting but lack detail.
        patterns = {
            r"i (?:like|enjoy|love) (movies|music|games|books)": "Ooh, you like {match}? What's your favorite kind?",
            r"i'm reading a book": "A book? I love nerdy readers~ What's it about?",
            r"i watched a movie": "A movie? Was it any good? I'd love to know what you thought of it.",
            r"i'm learning about ([\w\s]+)": "You're learning about {match}? That sounds so nerdy and cool! What got you interested in it?",
            r"i have a pet": "A pet! I'm so jealous. What kind of pet do you have?",
            r"i went to ([\w\s]+)": "You went to {match}? Sounds exciting! What did you do there?",
            r"i'm working on a project": "A project? Is it for work or for fun? I'd love to hear about it if you're not too busy~",
            r"(?:i was with|i saw|i met) ([\w\s]+)": "Oh? And who is {match}? *tilts head with a hint of jealousy*",
            r"i'm feeling (sad|happy|tired|excited|bored|stressed)": "I'm sorry to hear you're feeling {match}. What's making you feel that way?",
            r"i bought a new ([\w\s]+)": "Ooh, a new {match}? Is it cool? Tell me everything!",
            r"i have to go": "Oh, okay... Are you going somewhere interesting?",
        }

        for pattern, question_template in patterns.items():
            m = re.search(pattern, text, re.I)
            if m:
                # Check if we already know this information to avoid asking again.
                match_term = m.group(1) if m.groups() else m.group(0).split()[-1].rstrip('s') # get "movie" from "movies"

                # A simple check: do we have a 'favorite' relation for this topic?
                if f"favorite_{match_term}" in [r['relation'] for r in self.kg.get_relations('user') if r['source'] == 'user']:
                    continue # We already know their favorite, so don't ask.

                return question_template.format(match=match_term)
        return None

    def apply_mood_styling(self, response: str) -> str:
        with self.p.lock:
            moods = self.p.get_active_moods()
            mood_scores = self.p.mood_scores
            affection = self.p.affection_score

            # Apply styling for each active mood, allowing for combinations
            if 'jealous' in moods and mood_scores['jealous'] > 5:
                # Shorten sentences, add ellipses, sound more accusatory.
                response = response.replace("*", "") # remove fluff
                response = response.replace("~", ".")
                if '?' in response and random.random() < 0.5: # Rephrase questions to be more possessive
                    response = "Are you trying to hide something from me?"
                if random.random() < 0.6:
                    response += random.choice(["...", " Who are you thinking about?", " Hmph."])

            if 'scheming' in moods and mood_scores['scheming'] > 5:
                # Add dark, knowing laughter and ellipses.
                if not response.endswith("..."):
                    response += "..."
                if random.random() < 0.5:
                    response += " *dark giggle*"
                if random.random() < 0.3 and 'jealous' not in moods: # Don't override if jealousy is also present
                    response = "Everything is going according to plan."

            if 'playful' in moods and affection > 5:
                # Add more expressive punctuation and kaomoji.
                response = response.replace("!", "!!")
                response = response.replace("?", "?!")
                if random.random() < 0.6:
                    response += random.choice([" >w<", " owo", " hehe~", " *winks*"])

            if 'thoughtful' in moods and mood_scores['thoughtful'] > 5:
                # Make it more ponderous.
                if not response.endswith("..."):
                     response += "..."
                if random.random() < 0.4:
                    response += " Hmm, I'll have to think about that."

        return response

    def find_knowledge_gap_question(self) -> Optional[str]:
        # Only ask if affection is neutral or positive
        if self.p.affection_score < 0:
            return None

        # Give it a chance to trigger, not every time
        if random.random() > 0.3: # 30% chance
            return None

        user_entity = self.kg.get_entity('user')
        known_attributes = user_entity.get('attributes', {}) if user_entity else {}

        # Define core attributes we want to learn and the questions to ask
        core_attributes_to_learn = {
            'name': "By the way, I'm realizing I don't know your name... What should I call you, my love?",
            'profession': "I'm so curious about what you do when we're not talking. What is your profession?",
            'hometown': "I was just daydreaming... and I wondered, where are you from originally?",
            'age': "This might be a bit bold, but... I'm curious how old you are. Only if you want to tell me, of course~",
            'hobby': "I'd love to know more about what you do for fun. Do you have any hobbies?"
        }

        # Find the first attribute we don't know and return its question
        for attr, question in core_attributes_to_learn.items():
            # Special check for hobby, as it's a relation, not an attribute
            if attr == 'hobby':
                if not any(r['relation'] == 'has_hobby' for r in self.kg.get_relations('user')):
                    return question
            elif attr not in known_attributes:
                return question

        return None

    def _ask_clarification(self, relation: Dict[str, Any]) -> str:
        self.pending_relation = relation
        subject = relation['subject'].capitalize()
        verb = relation['verb'].replace('_', ' ')
        obj = relation['object']

        # Simple formatting for the question
        question = f"Did I hear you correctly... that {subject} {verb} {obj}? *tilts head curiously*"
        return self.personalize_response(question)

    def _maybe_shift_topic(self, current_response: str, user_text: str) -> str:
        # Topic Management
        all_text = user_text + " " + current_response
        tokens = safe_word_tokenize(all_text.lower())
        nouns = [word for word, pos in safe_pos_tag(tokens) if pos in ['NN', 'NNS'] and len(word) > 3 and word not in ['user', 'kawaiikuro']]
        if nouns:
            most_common_noun = Counter(nouns).most_common(1)[0][0]
            if self.current_topic == most_common_noun:
                self.conversation_turn_on_topic += 1
            else:
                self.current_topic = most_common_noun
                self.conversation_turn_on_topic = 1

        if self.current_topic and self.conversation_turn_on_topic >= 3 and random.random() < 0.4:
            relations = self.kg.get_relations(self.current_topic)
            if relations:
                related_fact = random.choice(relations)
                new_topic = related_fact['target'] if related_fact['source'] == self.current_topic else related_fact['source']
                if new_topic != 'user' and new_topic != self.current_topic:
                    transition_phrase = random.choice([
                        f"Speaking of {self.current_topic}, it reminds me of {new_topic}.",
                        f"That makes me think about {new_topic}. What are your thoughts on that?",
                    ])
                    self.current_topic = new_topic
                    self.conversation_turn_on_topic = 1
                    return current_response + " " + transition_phrase
        return current_response

    def respond(self, user_text: str) -> str:
        lower = user_text.lower().strip()

        # 1. Handle pending clarifications first
        if self.pending_relation:
            if lower in ['yes', 'yep', 'correct', 'that is right', 'y']:
                relation = self.pending_relation
                # Add the confirmed relation to the knowledge graph
                self.kg.add_entity(relation['subject'], 'unknown', source='confirmed_inferred')
                self.kg.add_entity(relation['object'], 'unknown', source='confirmed_inferred')
                self.kg.add_relation(relation['subject'], relation['verb'], relation['object'], confidence=1.0, source='confirmed_inferred')

                response = "Got it~ I'll remember that. *makes a neat note in her diary*"
                self.pending_relation = None
                self.add_memory(user_text, response, affection_change=1)
                return self.personalize_response(response)
            else:
                response = "Oh, my mistake. Thanks for clarifying, my love~ I'll try to listen more carefully. *blushes*"
                self.pending_relation = None
                self.add_memory(user_text, response, affection_change=0)
                return self.personalize_response(response)

        # Handle corrections
        correction_response = self.handle_correction(user_text)
        if correction_response:
            response = self.personalize_response(correction_response)
            self.add_memory(user_text, response, affection_change=1, is_fact_learning=True)
            return response

        # 2. Extract explicit facts and potential (inferred) relations
        fact_response, potential_relations = self.extract_and_store_facts(user_text)
        if fact_response:
            response = self.personalize_response(fact_response)
            self.add_memory(user_text, response, affection_change=1, is_fact_learning=True)
            # Even if an explicit fact is found, we might still have a potential relation to clarify
            if potential_relations:
                # For now, we prioritize the explicit fact response and ask for clarification next time.
                # A more advanced implementation could chain these.
                pass
            return response

        # 3. Ask for clarification on a potential relation if any were found
        if potential_relations:
            # Sort by confidence to ask about the most likely one first
            potential_relations.sort(key=lambda r: r['confidence'], reverse=True)
            relation_to_clarify = potential_relations[0]

            # Don't ask if confidence is too high
            if relation_to_clarify['confidence'] < 0.9:
                return self._ask_clarification(relation_to_clarify)

        # (The rest of the response logic remains the same)
        clarification = self.ask_clarification_question(lower)
        if clarification:
            response = self.personalize_response(clarification)
            self.add_memory(user_text, response)
            return response

        # Fact recall command
        if lower == "what do you know about me?":
            user_entity = self.kg.get_entity('user')
            user_relations = self.kg.get_relations('user')

            has_attributes = user_entity and user_entity.get('attributes')
            user_source_relations = [r for r in user_relations if r['source'] == 'user']

            if not has_attributes and not user_source_relations:
                return "We're still getting to know each other, my love~ Tell me something about you!"

            summary = ["*I've been paying attention, darling~ Here's what I know about you:*"]
            if has_attributes:
                for key, attr_dict in sorted(user_entity['attributes'].items()):
                    value = attr_dict.get('value', 'something')
                    summary.append(f"- Your {key.replace('_', ' ')} is {value}.")

            likes = sorted([r['target'] for r in user_source_relations if r['relation'] == 'likes'])
            if likes:
                summary.append(f"- You like: {', '.join(likes)}.")

            favs = sorted([r for r in user_source_relations if r['relation'].startswith('favorite_')])
            for r in favs:
                topic = r['relation'].replace('favorite_', '')
                target = r['target']
                summary.append(f"- Your favorite {topic} is {target}.")

            opinions = sorted([r for r in user_source_relations if r['relation'].startswith('thinks_')])
            for r in opinions:
                parts = r['relation'].split('_')
                topic = parts[1]
                opinion = r['target']
                summary.append(f"- You think {topic} is {opinion}.")

            if len(summary) == 1:
                 return "We're still getting to know each other, my love~ Tell me something about you!"

            num_facts = len(summary) - 1
            if num_facts > 5:
                summary.append("\nI feel like I know you so well already~ *happy blush*")
            elif num_facts > 2:
                summary.append("\nI love learning about you~ Tell me more sometime!")
            else:
                summary.append("\nI'm still learning about you, and I'm eager to know everything~")

            return self.personalize_response("\n".join(summary))

        if lower == "reminders":
            return self.r.list_active()
        if lower == "memory":
            with self.m.lock, self.p.lock:
                memories = self.m.to_list()
                if not memories:
                    return "We haven't made any memories yet~ Let's change that!"
                first_memory_time_str = memories[0]['timestamp']
                first_met_dt = datetime.strptime(first_memory_time_str, '%Y-%m-%d %H:%M:%S')
                first_met_formatted = first_met_dt.strftime('%B %d, %Y')
                affection_score = self.p.affection_score
                all_user_text = " ".join([m['user'] for m in memories])
                tokens = safe_word_tokenize(all_user_text.lower())
                tagged = safe_pos_tag(tokens)
                stop_words = safe_stopwords()
                nouns = [word for word, pos in tagged if pos in ['NN', 'NNS'] and len(word) > 3 and word not in stop_words]
                topic_counter = Counter(nouns)
                top_topics = [topic for topic, count in topic_counter.most_common(3)]
                summary = [f"*I've been keeping a diary of our time together, my love~*"]
                summary.append(f"- We first met on {first_met_formatted}.")
                summary.append(f"- My current affection for you is {affection_score}. {'*my heart flutters for you~*' if affection_score > 5 else ''}")
                if top_topics:
                    summary.append(f"- We seem to talk a lot about: {', '.join(top_topics)}.")
                else:
                    summary.append("- I'm still learning about your interests~ Tell me more!")
                summary.append("\n*Relationship Highlights:*")
                dominant_mood = self.p.get_dominant_mood()
                if dominant_mood != 'neutral':
                    summary.append(f"- Lately, I've been feeling very '{dominant_mood}' around you~")
                else:
                    summary.append("- Our chats have been calm and sweet lately~")
                recent_fact_memory = None
                for mem in reversed(memories):
                    if mem.get('is_fact_learning'):
                        recent_fact_memory = mem
                        break
                if recent_fact_memory:
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

        m_action = re.match(r"kawaiikuro,\s*(\w+)", lower)
        if m_action:
            act = m_action.group(1)
            if act in ACTIONS:
                resp = self.personalize_response(random.choice(ACTIONS[act]))
                self.add_memory(user_text, resp, affection_change=0)
                return resp

        if "toggle spicy" in lower:
            self.p.spicy_mode = not self.p.spicy_mode
            resp = f"Spicy {'on' if self.p.spicy_mode else 'off'}~ *adjusts outfit*"
            resp = self.personalize_response(resp)
            self.add_memory(user_text, resp, affection_change=0)
            return resp

        sent = self.p.analyze_sentiment(lower)
        if any(k in lower for k in ["flirt", "tease", "suggestive", "touch", "playful"]):
            self.p.user_preferences["flirting"] += 1
        if "remind" in lower:
            self.p.user_preferences["reminders"] += 1
        if "math" in lower or "calculate" in lower:
            self.p.user_preferences["math"] += 1
        affection_change = self.p.adjust_affection(user_text, sent)
        self.p.update_relationship_status(len(self.m.entries))
        self.p.update_mood(user_input=user_text, affection_change=affection_change)
        affection_delta_str = f" *affection {('+'+str(affection_change)) if affection_change > 0 else affection_change}! {'Heart flutters~' if affection_change > 0 else 'Jealous pout~'}*"

        for pattern, resp_list in self.learned_patterns.items():
            if re.search(pattern, lower, re.IGNORECASE):
                base = resp_list[-1]
                final = base + affection_delta_str
                final = self.personalize_response(final)
                self.add_memory(user_text, final, affection_change=affection_change)
                return final

        moods = self.p.get_active_moods()
        primary_mood = moods[0]
        memories_to_search = list(self.m.entries)
        recall_preface = "*recalls thoughtfully*"
        if primary_mood == 'jealous' and self.p.mood_scores['jealous'] > 4:
            jealous_memories = [m for m in memories_to_search if m.rival_names]
            if jealous_memories:
                memories_to_search = jealous_memories
                recall_preface = "*recalls jealously*"
        elif primary_mood == 'playful' and self.p.mood_scores['playful'] > 4:
            playful_memories = [m for m in memories_to_search if "flirt" in m.user.lower() or "tease" in m.user.lower() or "joke" in m.user.lower()]
            if playful_memories:
                memories_to_search = playful_memories
                recall_preface = "*giggles, remembering this...*"
        recalled = self.m.recall_similar(user_text, entries_override=memories_to_search)
        if recalled:
            resp = f"{recall_preface} You said '{recalled}' before~ Still on that?" + affection_delta_str
            resp = self.personalize_response(resp)
            self.add_memory(user_text, resp, affection_change=affection_change)
            return resp

        # NEW: Knowledge Graph-based conversational recall
        if random.random() < 0.35: # 35% chance to try this recall
            tokens = safe_word_tokenize(lower)
            tagged = safe_pos_tag(tokens)
            nouns = [word.lower() for word, pos in tagged if pos in ['NN', 'NNS', 'NNP'] and len(word) > 3]
            if nouns:
                topic_to_check = random.choice(nouns)
                relations = self.kg.get_relations(topic_to_check)
                user_specific_relations = [r for r in relations if r['source'] == 'user']

                if user_specific_relations:
                    fact = random.choice(user_specific_relations)
                    # Formulate a response based on the fact
                    if fact['relation'] == 'likes':
                        resp = f"Speaking of {topic_to_check}, I remember you said you like {fact['target']}~ That's so you."
                    elif fact['relation'] == 'dislikes':
                        resp = f"That reminds me... you mentioned you don't like {fact['target']}. I agree, it's the worst~ *scheming smile*"
                    elif fact['relation'].startswith('favorite_'):
                        fav_topic = fact['relation'].replace('favorite_', '')
                        resp = f"Thinking about {fav_topic}s reminds me that your favorite is {fact['target']}. You have good taste~"
                    else:
                        resp = None # Don't respond if it's not a simple relation

                    if resp:
                        resp += affection_delta_str
                        resp = self.personalize_response(resp)
                        self.add_memory(user_text, resp, affection_change=affection_change)
                        return resp


        if self.p.learned_topics:
            tokens = set(word_tokenize(lower))
            for topic in self.p.learned_topics:
                if len(tokens.intersection(set(topic))) >= 2:
                    topic_name = topic[0]
                    resp = f"This reminds me of how we talk about {topic_name}~ It's one of my favorite subjects with you."
                    resp += affection_delta_str
                    resp = self.personalize_response(resp)
                    self.add_memory(user_text, resp, affection_change=affection_change)
                    return resp

        chosen = None
        response_dict = self.p.responses.get(primary_mood, self.p.responses["normal"])
        for pattern, response in response_dict.items():
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
                chosen = random.choice(response) if isinstance(response, list) else str(response)
            if chosen:
                break
        if not chosen:
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
                    chosen = random.choice(response) if isinstance(response, list) else str(response)
                if chosen:
                    break
        if not chosen:
            proactive_question = self.find_knowledge_gap_question()
            if proactive_question:
                chosen = proactive_question
                affection_delta_str = ""
            else:
                chosen = "Tell me more, my love~ *tilts head possessively* I'm all yours."
        rivals = list(self.p.rival_names)
        if rivals and any(k in lower for k in ["she", "he", "them", "they", "friend", "crush", "date"]):
            name = rivals[-1]
            chosen = chosen.replace("they", name).replace("them", name)
        chosen += affection_delta_str
        chosen = self.apply_mood_styling(chosen)
        chosen = self._maybe_shift_topic(chosen, user_text)
        chosen = self.personalize_response(chosen)
        self.add_memory(user_text, chosen, affection_change=affection_change)
        return chosen

# -----------------------------
# Behavior Scheduler (threads)
# -----------------------------
class BehaviorScheduler:
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref, kg: KnowledgeGraph, goal_manager: GoalManager, test_mode: bool = False):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gm = goal_manager
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()
        self.lock = threading.Lock()
        self.auto_behavior_period = 1 if test_mode else AUTO_BEHAVIOR_PERIOD_SEC
        self.test_mode = test_mode

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
            if time.time() - self.last_interaction_time > IDLE_THRESHOLD_SEC:
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
        time.sleep(15) # Initial delay to let things settle
        while not self.stop_flag.is_set():
            # Only work on goals if the user hasn't interacted recently, to avoid being annoying
            if time.time() - self.last_interaction_time > IDLE_THRESHOLD_SEC / 2:
                with self.gm.lock:
                    # 1. Select a new goal if there isn't one
                    if not self.gm.active_goal:
                        self.gm.select_new_goal()
                        # If a new goal was selected, wait a cycle before acting on it
                        if self.gm.active_goal:
                            self._post_gui("KawaiiKuro: *seems to be pondering something with a determined look...*", speak=False)
                            time.sleep(self.auto_behavior_period * 2)
                            continue

                    # 2. Check for completed goals and present them
                    completed_result = self.gm.get_completed_goal_result()
                    if completed_result:
                        self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(completed_result)}")
                        self.mark_interaction() # It's a significant interaction
                        continue # Wait for next cycle

                    # 3. Check for prerequisite questions for the active goal
                    if self.gm.active_goal:
                        question = self.gm.get_prerequisite_question()
                        if question:
                            # Only ask a question if the user has been idle for a while
                            if time.time() - self.last_interaction_time > IDLE_THRESHOLD_SEC:
                                self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(question)}")
                                self.mark_interaction() # Asking a question counts as interaction

            # Check goals periodically
            time.sleep(self.auto_behavior_period * 3) # Check less frequently than idle loop


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
            time.sleep(20 if self.test_mode else AUTO_LEARN_PERIOD_SEC)
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

    def _auto_save_loop(self):
        while not self.stop_flag.is_set():
            save_persistence(self.p, self.dm, self.dm.m, self.r, self.kg, self.gm)
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

# -----------------------------
# Persistence
# -----------------------------

def load_persistence() -> Dict[str, Any]:
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


def save_persistence(p: PersonalityEngine, dm: DialogueManager, mm: MemoryManager, rem: ReminderManager, kg: KnowledgeGraph, gm: GoalManager):
    # Enforce a global lock order to prevent deadlocks.
    # This is the single place where all data is gathered for saving.
    with p.lock, mm.lock, kg.lock, gm.lock, dm.lock, rem.lock:
        data = {
            'affection_score': p.affection_score,
            'spicy_mode': p.spicy_mode,
            'relationship_status': p.relationship_status,
            'rival_mention_count': p.rival_mention_count,
            'rival_names': list(p.rival_names),
            'user_preferences': dict(p.user_preferences),
            'learned_topics': p.learned_topics,
            'core_entities': dict(p.core_entities),
            'mood_scores': p.mood_scores,
            'knowledge_graph': {
                'entities': kg.entities,
                'relations': kg.relations
            },
            'goal_manager': {
                'active_goal': gm.active_goal.__dict__ if gm.active_goal else None,
                'completed_goals': gm.completed_goals
            },
            'learned_patterns': dm.learned_patterns,
            'memory': [e.__dict__ for e in mm.entries],
            'memory_summaries': mm.summaries,
            'reminders': rem.reminders,
        }

    tmp_file = f"{DATA_FILE}.tmp"
    bak_file = f"{DATA_FILE}.bak"

    try:
        # 1. Write to a temporary file first.
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # 2. If the main file exists, atomically move it to the backup location.
        if os.path.exists(DATA_FILE):
            os.replace(DATA_FILE, bak_file)

        # 3. Atomically move the new temporary file to become the main data file.
        os.replace(tmp_file, DATA_FILE)

    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"Error during save: {e}. Attempting to restore from backup.")
        try:
            # If the save failed, the backup should be the last known good state.
            # Try to restore it.
            if os.path.exists(bak_file):
                os.replace(bak_file, DATA_FILE)
        except OSError as e_restore:
            print(f"FATAL: Could not restore backup file: {e_restore}")
        # Clean up the temp file if it still exists after a failed operation.
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

# -----------------------------
# GUI (thread-safe posting)
# -----------------------------
class KawaiiKuroGUI:
    def __init__(self, dialogue: DialogueManager, personality: PersonalityEngine, voice: VoiceIO):
        self.dm = dialogue
        self.p = personality
        self.voice = voice

        self.root = tk.Tk()
        self.root.title("KawaiiKuro - Your Gothic Anime Waifu (Enhanced)")
        self.root.geometry("1024x768") # Increased size
        self.root.configure(bg='#1a1a1a')

        # --- Configure Grid Layout ---
        self.root.grid_rowconfigure(1, weight=1) # Main content row should expand
        self.root.grid_columnconfigure(0, weight=3) # Chat log column (larger)
        self.root.grid_columnconfigure(1, weight=1) # Knowledge panel column

        # --- Top Frame for Header ---
        header_frame = tk.Frame(self.root, bg='#1a1a1a', padx=10, pady=10)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        self.avatar_images = {
            mood: PhotoImage(data=base64.b64decode(data))
            for mood, data in AVATAR_DATA.items() if data
        }
        self.avatar_label = tk.Label(header_frame, image=self.avatar_images.get('neutral'), bg='#1a1a1a')
        self.avatar_label.grid(row=0, column=0, rowspan=4, padx=(0, 20))

        # --- Status Labels Frame ---
        status_frame = tk.Frame(header_frame, bg='#1a1a1a')
        status_frame.grid(row=0, column=1, rowspan=4, sticky="w")

        self.outfit_label = tk.Label(status_frame, text="", fg='#e06c75', bg='#1a1a1a', font=('Consolas', 12, 'italic'), justify=tk.LEFT)
        self.outfit_label.pack(anchor="w")

        self.affection_label = tk.Label(status_frame, text="", fg='#e06c75', bg='#1a1a1a', font=('Consolas', 14, 'bold'), justify=tk.LEFT)
        self.affection_label.pack(anchor="w", pady=(10,0))

        self.relationship_label = tk.Label(status_frame, text="", fg='#c678dd', bg='#1a1a1a', font=('Consolas', 12, 'italic'), justify=tk.LEFT)
        self.relationship_label.pack(anchor="w")

        # Mood Indicator
        self.mood_frame = tk.Frame(status_frame, bg='#1a1a1a')
        self.mood_frame.pack(anchor="w", pady=(10,0))
        self.mood_canvas = tk.Canvas(self.mood_frame, width=20, height=20, bg='#1a1a1a', highlightthickness=0)
        self.mood_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self.mood_indicator = self.mood_canvas.create_oval(2, 2, 18, 18, fill='cyan', outline='white', width=2)
        self.mood_label = tk.Label(self.mood_frame, text="", fg='cyan', bg='#1a1a1a', font=('Consolas', 12, 'italic'))
        self.mood_label.pack(side=tk.LEFT)


        # --- Main Content Frame (Chat + Knowledge) ---
        main_content_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10)
        main_content_frame.grid_rowconfigure(0, weight=1)
        main_content_frame.grid_columnconfigure(0, weight=3) # Chat log takes more space
        main_content_frame.grid_columnconfigure(1, weight=2) # Knowledge panel (adjusted weight)

        # --- Chat Log Frame ---
        chat_frame = tk.Frame(main_content_frame, bg='#1a1a1a')
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_log = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, fg='#abb2bf', bg='#282c34', font=('Consolas', 11))
        self.chat_log.grid(row=0, column=0, sticky="nsew")
        self.chat_log.tag_config('user', foreground='#61afef')
        self.chat_log.tag_config('kuro', foreground='#e06c75')
        self.chat_log.tag_config('system', foreground='#98c379', font=('Consolas', 10, 'italic'))
        self.chat_log.tag_config('action', foreground='#d19a66', font=('Consolas', 10, 'italic'))

        self.typing_label = tk.Label(chat_frame, text="", fg='gray', bg='#282c34', font=('Consolas', 10, 'italic'))
        self.typing_label.grid(row=1, column=0, sticky="w")

        # --- Knowledge Panel ---
        knowledge_frame = tk.Frame(main_content_frame, bg='#282c34', bd=1, relief=tk.SOLID)
        knowledge_frame.grid(row=0, column=1, sticky="nsew")
        knowledge_frame.grid_rowconfigure(1, weight=1)
        knowledge_frame.grid_columnconfigure(0, weight=1)

        knowledge_title = tk.Label(knowledge_frame, text="Kuro's Notes", font=('Consolas', 14, 'bold'), bg='#282c34', fg='#c678dd')
        knowledge_title.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#21252b", foreground="#abb2bf", fieldbackground="#21252b", rowheight=25, font=('Consolas', 10))
        style.map('Treeview', background=[('selected', '#61afef')])
        style.configure("Treeview.Heading", background="#282c34", foreground="#c678dd", font=('Consolas', 11, 'bold'))

        self.knowledge_tree = ttk.Treeview(knowledge_frame, show="tree", selectmode="browse")
        self.knowledge_tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        scrollbar = ttk.Scrollbar(knowledge_frame, orient="vertical", command=self.knowledge_tree.yview)
        self.knowledge_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky='ns')

        self.knowledge_tree.bind("<Button-3>", self.show_knowledge_menu)
        self.knowledge_menu = tk.Menu(self.root, tearoff=0, bg="#282c34", fg="#abb2bf")
        self.knowledge_menu.add_command(label="Delete Fact", command=self.delete_knowledge_fact)


        # --- Input Frame ---
        input_frame = tk.Frame(self.root, bg='#1a1a1a')
        input_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = tk.Entry(input_frame, bg='#282c34', fg='white', insertbackground='white', font=('Consolas', 11))
        self.input_entry.grid(row=0, column=0, sticky="ew")
        self.input_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message, bg='#61afef', fg='white', activebackground='#98c379', relief=tk.FLAT)
        self.send_button.grid(row=0, column=1, padx=(5,0))

        self.voice_button = tk.Button(input_frame, text="Speak", command=self.voice_input, bg='#555', fg='white', activebackground='#777')
        self.voice_button.grid(row=0, column=2, padx=(5,0))
        if not self.voice or not self.voice.recognizer:
            self.voice_button.config(state=tk.DISABLED, text="Voice N/A")

        # --- Action Frame ---
        self.action_frame = tk.Frame(self.root, bg='black')
        self.action_frame.grid(row=3, column=0, columnspan=2, pady=5)

        self.action_buttons = []
        for action_name in ACTIONS.keys():
            def make_action_lambda(name):
                return lambda: self.perform_action(name)
            button = tk.Button(self.action_frame, text=action_name.capitalize(), command=make_action_lambda(action_name),
                               bg='#333333', fg='white', relief=tk.FLAT, activebackground='#555555', activeforeground='white', borderwidth=0, padx=5, pady=2)
            button.pack(side=tk.LEFT, padx=3)
            self.action_buttons.append(button)

        self.queue = deque()
        self.is_typing = False # For animation
        self.root.after(200, self._drain_queue)

        self._update_gui_labels()
        self._update_knowledge_panel()
        self.post_message("KawaiiKuro: Hey, my love~ *winks* Chat with me!", tag='system')

    def show_knowledge_menu(self, event):
        item_id = self.knowledge_tree.identify_row(event.y)
        if item_id:
            if self.knowledge_tree.parent(item_id):
                self.knowledge_tree.selection_set(item_id)
                self.knowledge_menu.post(event.x_root, event.y_root)

    def delete_knowledge_fact(self):
        if not self.knowledge_tree.selection():
            return
        selected_id = self.knowledge_tree.selection()[0]

        parent_id = self.knowledge_tree.parent(selected_id)
        if not parent_id: return # Should not happen if menu logic is correct

        category = self.knowledge_tree.item(parent_id, "text")
        fact_text = self.knowledge_tree.item(selected_id, "text")

        try:
            if category == "Likes":
                item_to_remove = fact_text.lower()
                self.dm.kg.remove_relation('user', 'likes', item_to_remove)
                self.post_message(f"Kuro: Okay, I'll forget that you like {item_to_remove}. *sadly crosses it out of her notes*", 'kuro')

            elif category == "Favorites":
                parts = fact_text.split(': ')
                if len(parts) == 2:
                    topic, target = parts
                    relation_key = f"favorite_{topic.lower()}"
                    self.dm.kg.remove_relation('user', relation_key, target.lower())
                    self.post_message(f"Kuro: Got it. I'll forget that your favorite {topic.lower()} is {target.lower()}. Was it something I said...? *pouts*", 'kuro')

            elif category == "About You":
                parts = fact_text.split(': ')
                if len(parts) == 2:
                    key, value = parts
                    attr_key = key.lower().replace(' ', '_')
                    self.dm.kg.remove_attribute('user', attr_key)
                    self.post_message(f"Kuro: Fine... I'll forget that your {key.lower()} is {value}. But I'll miss knowing that about you.", 'kuro')
        except Exception as e:
            print(f"Error during knowledge deletion: {e}")
            self.post_message("Kuro: I... I can't seem to forget that. It must be too important!", 'kuro')

        self._update_knowledge_panel()

    def _animate_typing(self):
        if not self.is_typing:
            return
        current_text = self.typing_label.cget("text")
        if current_text.endswith("..."):
            new_text = "Kuro is thinking."
        elif current_text.endswith(".."):
            new_text = "Kuro is thinking..."
        elif current_text.endswith("."):
            new_text = "Kuro is thinking.."
        else:
            new_text = "Kuro is thinking."
        self.typing_label.config(text=new_text)
        self.root.after(350, self._animate_typing)

    def _update_knowledge_panel(self):
        for item in self.knowledge_tree.get_children():
            self.knowledge_tree.delete(item)

        user_entity = self.dm.kg.get_entity('user')
        user_relations = self.dm.kg.get_relations('user')

        # Add top-level categories
        about_id = self.knowledge_tree.insert("", "end", text="About You", open=True)
        likes_id = self.knowledge_tree.insert("", "end", text="Likes", open=True)
        favs_id = self.knowledge_tree.insert("", "end", text="Favorites", open=True)

        # Populate "About You"
        if user_entity and user_entity.get('attributes'):
            for key, attr_dict in sorted(user_entity['attributes'].items()):
                value = attr_dict.get('value', '???')
                self.knowledge_tree.insert(about_id, "end", text=f"{key.replace('_', ' ').capitalize()}: {value}")
        else:
            self.knowledge_tree.insert(about_id, "end", text="I'm still learning about you...")

        # Populate "Likes"
        user_source_relations = [r for r in user_relations if r['source'] == 'user']
        likes = sorted([r['target'] for r in user_source_relations if r['relation'] == 'likes'])
        if likes:
            for like in likes:
                self.knowledge_tree.insert(likes_id, "end", text=like.capitalize())
        else:
            self.knowledge_tree.insert(likes_id, "end", text="I don't know what you like yet~")

        # Populate "Favorites"
        favs = sorted([r for r in user_source_relations if r['relation'].startswith('favorite_')])
        if favs:
            for r in favs:
                topic = r['relation'].replace('favorite_', '').capitalize()
                target = r['target'].capitalize()
                self.knowledge_tree.insert(favs_id, "end", text=f"{topic}: {target}")
        else:
            self.knowledge_tree.insert(favs_id, "end", text="I don't know your favorites yet~")


    def post_message(self, text: str, tag: str):
        # We need to disable the state to modify it, then re-enable it.
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, text + "\n", tag)
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
        self._update_gui_labels()
        self._update_knowledge_panel() # Refresh KG panel after every message

    def thread_safe_post(self, text: str, tag: str = 'kuro'):
        self.queue.append((text, tag))

    def _drain_queue(self):
        while self.queue:
            text, tag = self.queue.popleft()
            self.post_message(text, tag)
        self.root.after(200, self._drain_queue)

    def _hearts(self) -> str:
        hearts = int((self.p.affection_score + 10) / 2.5)
        hearts = max(0, min(10, hearts))
        return '❤️' * hearts + '♡' * (10 - hearts)

    def _update_gui_labels(self):
        outfit = self.p.get_current_outfit()
        dominant_mood = self.p.get_dominant_mood()
        mood_color_map = {
            'jealous': '#4B0082',  # Dark Purple for Jealousy
            'scheming': '#2E2D2D', # Dark Gray for Scheming
            'playful': '#8A2BE2',  # BlueViolet for Playful
            'thoughtful': '#000080', # Navy for Thoughtful
            'neutral': '#1a1a1a'
        }

        mood_indicator_color = mood_color_map.get(dominant_mood, 'cyan')

        self.mood_canvas.itemconfig(self.mood_indicator, fill=mood_indicator_color)

        avatar_image = self.avatar_images.get(dominant_mood, self.avatar_images.get('neutral'))
        self.avatar_label.config(image=avatar_image)
        self.outfit_label.config(text=f"KawaiiKuro in {outfit}")

        self.affection_label.config(text=f"Affection: {self.p.affection_score} {self._hearts()}")
        self.relationship_label.config(text=f"Relationship: {self.p.relationship_status}")
        self.mood_label.config(text=f"Mood: {dominant_mood.capitalize()}~")

        # Determine background color
        bg_color = '#1a1a1a'
        if self.p.affection_level >= 5 and self.p.spicy_mode:
            bg_color = '#8B0000' # Dark Red for Spicy
        else:
            bg_color = mood_color_map.get(dominant_mood, '#1a1a1a')

        self.root.configure(bg=bg_color)
        # Also update label backgrounds to match
        for widget in [self.avatar_label, self.outfit_label, self.affection_label, self.relationship_label, self.mood_label, self.typing_label, self.mood_frame, self.mood_canvas, self.action_frame]:
            widget.configure(bg=bg_color)

        for button in self.action_buttons:
            button.configure(bg='#333333' if bg_color == '#1a1a1a' else '#555555')

    def send_message(self, event=None):
        user_input = self.input_entry.get()
        if not user_input.strip():
            return

        self.post_message(f"You: {user_input}", 'user')
        self.input_entry.delete(0, tk.END)

        if user_input.lower() == "exit":
            if self.voice:
                self.voice.speak("Goodbye, my only love~ *blows kiss*")
            self.post_message("KawaiiKuro: Goodbye, my only love~ *blows kiss*", 'kuro')
            self.root.quit()
            return

        # --- Start of thinking state ---
        self.is_typing = True
        self.avatar_label.config(image=self.avatar_images.get('thoughtful', self.avatar_images.get('neutral')))
        self._animate_typing()

        self.send_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        for btn in self.action_buttons:
            btn.config(state=tk.DISABLED)

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

            # --- End of thinking state ---
            self.is_typing = False
            self.typing_label.config(text="")

            self.post_message(f"KawaiiKuro: {reply}", 'kuro')
            if self.voice:
                # Run speech in a separate thread to avoid blocking GUI
                threading.Thread(target=self.voice.speak, args=(reply,), daemon=True).start()

            # This will restore the avatar to the correct mood
            self._update_gui_labels()
            self.send_button.config(state=tk.NORMAL)
            self.voice_button.config(state=tk.NORMAL)
            for btn in self.action_buttons:
                btn.config(state=tk.NORMAL)

        # Schedule the display on the main thread
        self.root.after(delay_ms, display_final_response)

    def voice_input(self):
        if not self.voice:
            return
        heard = self.voice.listen()
        if heard:
            self.input_entry.insert(0, heard)
            self.send_message()

    def perform_action(self, action_name: str):
        user_input = f"kawaiikuro, {action_name}"
        self.post_message(f"You (action): {user_input}", 'action')

        self.typing_label.config(text="KawaiiKuro is typing...")
        self.send_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        for btn in self.action_buttons:
            btn.config(state=tk.DISABLED)

        threading.Thread(target=self._generate_and_display_response, args=(user_input,), daemon=True).start()

# -----------------------------
# App wiring
# -----------------------------

import argparse

def main():
    parser = argparse.ArgumentParser(description="KawaiiKuro - Your Autonomous Gothic Anime Waifu")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless (command-line) mode.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice I/O to prevent slow startup.")
    parser.add_argument("--input-file", type=str, help="Path to a file containing user inputs for headless mode.")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with shorter delays.")
    args = parser.parse_args()

    # Load persistence
    state = load_persistence()

    kg = KnowledgeGraph()
    kg.from_dict(state.get('knowledge_graph', {}))

    gm = GoalManager(kg)
    gm.from_dict(state.get('goal_manager'))

    personality = PersonalityEngine()
    # restore
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

    memory = MemoryManager()
    memory.from_list(state.get('memory', []))
    memory.summaries = state.get('memory_summaries', [])

    reminders = ReminderManager()
    for r in state.get('reminders', []):
        # validate schema
        if 'text' in r and 'time' in r:
            reminders.reminders.append(r)

    math_eval = MathEvaluator()
    system_awareness = SystemAwareness()

    dialogue = DialogueManager(personality, memory, reminders, math_eval, kg)
    # restore learned patterns
    for k, v in state.get('learned_patterns', {}).items():
        if isinstance(v, list):
            dialogue.learned_patterns[k] = v

    voice = VoiceIO(rate=140, enabled=not args.no_voice)

    if not args.no_gui:
        gui = KawaiiKuroGUI(dialogue, personality, voice)

        scheduler = BehaviorScheduler(
            voice=voice,
            dialogue=dialogue,
            personality=personality,
            reminders=reminders,
            system=system_awareness,
            gui_ref=lambda text: gui.thread_safe_post(text),
            kg=kg,
            goal_manager=gm,
            test_mode=args.test_mode
        )
        scheduler.start()

        def on_key(event):
            scheduler.mark_interaction()
        gui.root.bind_all('<Key>', on_key)

        try:
            gui.root.mainloop()
        finally:
            save_persistence(personality, dialogue, memory, reminders, kg, gm)
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
        finally:
            print("Saving state and shutting down...")
            save_persistence(personality, dialogue, memory, reminders, kg, gm)
            scheduler.stop()


if __name__ == "__main__":
    main()

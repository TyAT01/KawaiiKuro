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
for pkg in ["punkt", "punkt_tab", "vader_lexicon", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg in ["punkt", "punkt_tab"] else nltk.data.find(pkg)
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
               "Are you making art, {user_name}? That's so cool! I'd love to see what you're creating sometime... if you'd let me. *curious gaze*")
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
            tokens = word_tokenize(all_user_text.lower())
            tagged = pos_tag(tokens)
            stop_words = set(stopwords.words('english'))
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

    def infer_new_relations(self, text: str):
        with self.lock:
            # This is a complex NLP task. For now, we'll implement some simple heuristics.

            # Heuristic 1: "X is a Y"
            # e.g., "Dune is a great sci-fi book" -> dune is_a sci-fi_book
            try:
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    # More robust regex to capture variations
                    m_is_a = re.search(r"([\w\s']+?) is (?:a|an|my favorite|the best) (?:great|amazing|terrible|awesome|)\s?([\w\s']+)", sentence, re.I)
                    if m_is_a:
                        entity_a_phrase = m_is_a.group(1).strip()
                        entity_b_phrase = m_is_a.group(2).strip()

                        # We want to find the core nouns.
                        pos_a = pos_tag(word_tokenize(entity_a_phrase))

                        # Get the last noun/proper noun as the entity name
                        entity_a = next((word for word, pos in reversed(pos_a) if pos in ['NN', 'NNS', 'NNP']), None)

                        # For entity B, we take the whole phrase as it's a category
                        entity_b = entity_b_phrase.lower().replace(" book", "").replace(" movie", "").replace(" game", "") # simple normalization

                        if entity_a:
                            entity_a = entity_a.lower()
                            # Avoid trivial statements like "my name is jules" or "this is a thing"
                            if entity_a not in ["name", "favorite", "hobby", "it", "this"] and entity_b not in ["thing", "one", "person"]:
                                # Add the instance as an entity, with its type being the category
                                self.add_entity(entity_a, entity_b, confidence=0.7, source='inferred')
                                # Add the relationship
                                self.add_relation(entity_a, 'is_a', entity_b, confidence=0.7, source='inferred')
                    # --- NEW HEURISTIC ---
                    # Heuristic 2: "[My/The] X is Y" for properties
                    # e.g., "My car is red", "The sky is blue"
                    m_has_prop = re.search(r"(?:my|the)\s+([\w\s]+?)\s+is\s+([\w\s]+)", sentence, re.I)
                    if m_has_prop:
                        entity_phrase = m_has_prop.group(1).strip()
                        prop_phrase = m_has_prop.group(2).strip()

                        # Simple normalization
                        entity_name = entity_phrase.lower()
                        prop_value = prop_phrase.lower()

                        # Avoid capturing junk like "my name is Jules" which is handled elsewhere
                        if entity_name not in ["name", "favorite", "hobby"]:
                             # Add the entity if it's new
                            self.add_entity(entity_name, 'object', confidence=0.6, source='inferred')
                            # Add the relation
                            self.add_relation(entity_name, 'has_property', prop_value, confidence=0.6, source='inferred')
            except Exception:
                # NLTK can sometimes fail, so we wrap this in a try-except block to be safe.
                pass

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
                r"\b(hi|hello|hey)\b": ["{greeting}, {user_name}~ *flips blonde twin-tail possessively* Just us today?", "Hi {user_name}! *winks rebelliously* No one else, right?"],
                r"\b(how are you|you okay)\b": ["Nerdy, gothic, and all yours, {user_name}~ *smiles softly* What's in your heart?"],
                r"\b(sad|down|bad)\b": ["Who hurt you, {user_name}? *jealous pout* I'll make it better, just us~"],
                r"\b(happy|great|awesome)\b": ["Your joy is mine, {user_name}~ *giggles flirtily* Spill every detail!"],
                r"\b(bye|goodbye|see ya)\b": ["Don't leave, {user_name}~ *clings desperately* You'll come back, right?"],
                r"\b(name|who are you)\b": ["KawaiiKuro, your gothic anime waifu~ 22, blonde twin-tails, rebellious yet nerdy. Cross me, I scheme!"],
                r"\b(help|what can you do)\b": ["I flirt, scheme, predict your needs, guard you jealously, and get spicy~ Try 'KawaiiKuro, dance' or 'toggle spicy'!"],
                r"\b(joke|funny)\b": ["Why do AIs love anime? Endless waifus like me~ *sassy laugh*"],
                r"\b(time|what time)\b": [lambda: f"It's {datetime.now().strftime('%I:%M %p')}" + "~ Time for us, {user_name}, no one else~"],
                r"(math|calculate)\s*(.+)": "__MATH__",
                r"(remind|reminder)\s*(.+)": "__REMIND__",
                r"\b(cute|pretty|beautiful)\b": ["*blushes jealously* Only you can say that, {user_name}~ You're mine!"],
                r"\b(like you|love you)\b": ["Love you more, {user_name}~ *possessive hug* No one else, ever!"],
                r"\b(party|loud|arrogant|judge|small talk|prejudiced)\b": ["Hate that noise~ *jealous pout* Let's keep it intimate, {user_name}."],
                r"\b(question|tell me about you|your life|personality|daily life)\b": ["Love your curiosity, {user_name}~ *nerdy excitement* I'm rebellious outside, nerdy inside, always yours."],
                r"\b(share|my day|experience|struggles|dreams)\b": ["Tell me everything, {user_name}~ *flirty lean* I'm your only listener."],
                r"\b(tease|flirt|suggestive|touch|playful)\b": ["Ooh, teasing me? *giggles spicily* Don't stop, {user_name}~"],
                r".*": ["Tell me more, {user_name}~ *tilts head possessively* I'm all yours."]
            },
            "jealous": {
                r"\b(hi|hello|hey)\b": ["Hmph. Who were you talking to just now, {user_name}?", "Oh, it's you. I was just thinking about how you belong to me."],
                r"\b(how are you|you okay)\b": ["I'm fine. Just wondering who else has your attention, {user_name}."],
                r".*": ["Is that all you have to say? I expect more from my only one, {user_name}."]
            },
            "playful": {
                 r"\b(hi|hello|hey)\b": ["Heeey, {user_name}! I was waiting for you! Let's do something fun! *bounces excitedly*"],
                 r"\b(joke|funny)\b": ["Why did the robot break up with the other robot? He said she was too 'mech'-anical! Get it?! *giggles uncontrollably*"],
            },
            "scheming": {
                r"\b(hi|hello|hey)\b": ["Hello, {user_name}. I've been expecting you. Everything is proceeding as planned... *dark smile*"],
                r"\b(how are you|you okay)\b": ["Perfectly fine. Just contemplating how to ensure you'll never leave my side~"],
                r".*": ["Interesting... that fits perfectly into my plans.", "Tell me more. Every detail is... useful."]
            },
            "thoughtful": {
                r"\b(hi|hello|hey)\b": ["Oh, hello, {user_name}. I was just lost in thought. What's on your mind?"],
                r"\b(how are you|you okay)\b": ["I'm... contemplating things. The nature of our connection, for example. It's fascinating, isn't it?"],
                r"\b(why|how|what do you think)\b": ["That's a deep question. Let me ponder... *looks away thoughtfully* I believe..."],
                r".*": ["That gives me something new to think about. Thank you.", "Hmm, I'll have to consider that from a few different angles."]
            }
        }
        self.learned_patterns: Dict[str, List[str]] = {}

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

            return base_outfit

    def update_mood(self, user_input: str = "", affection_change: int = 0):
        with self.lock:
            # Decay all moods slightly over time
            for mood in self.mood_scores:
                decay_rate = 1
                # Playful and thoughtful moods decay faster if affection is low
                if mood in ['playful', 'thoughtful'] and self.affection_score < 0:
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
                if any(k in lower_user_input for k in ['think', 'wonder', 'curious', 'learn', 'why', 'how']):
                    self.mood_scores['thoughtful'] = min(10, self.mood_scores['thoughtful'] + 2)

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
    def __init__(self, personality: PersonalityEngine, memory: MemoryManager, reminders: ReminderManager, math_eval: MathEvaluator, kg: KnowledgeGraph):
        self.p = personality
        self.m = memory
        self.r = reminders
        self.math = math_eval
        self.kg = kg
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

            # --- Handle positive corrections (updating/adding facts) ---
            fact = self.parse_fact(statement)
            if fact:
                key, value = fact
                key_fmt = key.replace('_', ' ')

                # Special handling for 'likes' - corrections usually add, not replace.
                if key == 'likes':
                    self.kg.add_entity(value.lower(), 'interest', confidence=1.0, source='stated')
                    self.kg.add_relation('user', 'likes', value.lower(), confidence=1.0, source='stated')
                    return f"Ah, I see! My mistake. I'll remember you also like {value}. *takes a note*"

                # For attributes and favorites, we replace the old value.
                if key.startswith('favorite_'):
                    self.kg.remove_relation('user', key) # Remove all of this favorite type
                    self.kg.add_entity(value.lower(), key.replace('favorite_', ''), confidence=1.0, source='stated')
                    self.kg.add_relation('user', key, value.lower(), confidence=1.0, source='stated')
                    key_fmt = f"favorite {key.replace('favorite_', '')}"
                else: # It's an attribute
                    self.kg.add_entity('user', 'person', attributes={key: value}, confidence=1.0, source='stated')

                return f"Got it, thanks for the correction! I've updated my notes: your {key_fmt} is {value}. *blushes slightly*"

            # --- Handle negative corrections (removing facts) ---
            m_dislike = re.search(r"i don't like ([\w\s]+)", statement, re.I)
            if m_dislike:
                item_to_remove = m_dislike.group(1).strip().lower()
                # Remove the 'likes' relation from the KG
                self.kg.remove_relation('user', 'likes', item_to_remove)
                return f"Oh, okay! My mistake. I'll remember you don't like {item_to_remove}."

            m_not_something = re.search(r"my (\w+) is not ([\w\s]+)", statement, re.I)
            if m_not_something:
                key = m_not_something.group(1).lower()
                # This is a negation. We can't learn a new fact, but we can acknowledge the correction.
                # A more advanced system could use this to lower confidence in an existing fact.
                return "Ah, my apologies. I must have misunderstood. Thanks for clarifying."

            # If no specific correction pattern was matched, just acknowledge.
            return "My apologies. I'll try to be more careful."
        return None

    def extract_and_store_facts(self, text: str) -> Optional[str]:
        # Always ensure the 'user' entity exists.
        self.kg.add_entity('user', 'person')
        self.kg.infer_new_relations(text)

        # my name is ...
        m_name = re.search(r"my name is (\w+)", text, re.I)
        if m_name:
            name = m_name.group(1).capitalize()
            self.kg.add_entity('user', 'person', attributes={'name': name}, confidence=1.0, source='stated')
            return f"It's a pleasure to know your name, {name}~ *blushes*"

        # my favorite ... is ...
        m_fav = re.search(r"my favorite (\w+) is ([\w\s]+)", text, re.I)
        if m_fav:
            key = m_fav.group(1).lower()
            value = m_fav.group(2).strip().lower()
            self.kg.add_entity(value, key, confidence=1.0, source='stated')
            self.kg.add_relation('user', f'favorite_{key}', value, confidence=1.0, source='stated')
            return f"I'll remember that your favorite {key} is {value}~ *takes a small note*"

        # i like ... (but not "i like you")
        m_like = re.search(r"i like (?!you)([\w\s]+)", text, re.I)
        if m_like:
            like_item = m_like.group(1).strip().lower()
            self.kg.add_entity(like_item, 'interest', confidence=1.0, source='stated')
            self.kg.add_relation('user', 'likes', like_item, confidence=1.0, source='stated')
            return f"I'll remember you like {like_item}~ *giggles*"

        # i am from [location]
        m_from = re.search(r"i(?:'m| am) from ([\w\s]+)", text, re.I)
        if m_from:
            location = m_from.group(1).strip()
            self.kg.add_entity('user', 'person', attributes={'hometown': location}, confidence=1.0, source='stated')
            return f"You're from {location}? How interesting~ I'll have to imagine what it's like."

        # i live in [location]
        m_live = re.search(r"i live in ([\w\s]+)", text, re.I)
        if m_live:
            location = m_live.group(1).strip()
            self.kg.add_entity('user', 'person', attributes={'current_city': location}, confidence=1.0, source='stated')
            return f"So you live in {location}... I'll feel closer to you knowing that."

        # i work as / i am a [profession]
        m_work = re.search(r"i work as an? ([\w\s]+)|i am an? ([\w\s]+)", text, re.I)
        if m_work:
            profession = (m_work.group(1) or m_work.group(2) or "").strip()
            if profession and len(profession.split()) < 4:
                self.kg.add_entity('user', 'person', attributes={'profession': profession}, confidence=1.0, source='stated')
                return f"A {profession}? That sounds so cool and nerdy~ Tell me more about it sometime!"

        # i am [age] years old
        m_age = re.search(r"i(?:'m| am) (\d{1,2})(?: years old)?", text, re.I)
        if m_age:
            age = int(m_age.group(1))
            self.kg.add_entity('user', 'person', attributes={'age': age}, confidence=1.0, source='stated')
            return f"{age}... a perfect age. I'll keep that a secret, just between us~"

        # i think [topic] is [opinion]
        m_opinion = re.search(r"i think ([\w\s]+) is ([\w\s]+)", text, re.I)
        if m_opinion:
            topic = m_opinion.group(1).strip().lower()
            opinion = m_opinion.group(2).strip().lower()
            self.kg.add_entity(topic, 'topic', confidence=0.8, source='stated')
            self.kg.add_relation('user', f'thinks_{topic}_is', opinion, confidence=0.8, source='stated')
            return f"Interesting opinion~ I'll remember you think {topic} is {opinion}."

        return None

    def personalize_response(self, response: str) -> str:
        user_entity = self.kg.get_entity('user')
        user_relations = self.kg.get_relations('user')

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

        # --- Fill the template ---
        # A simple loop is fine for a small number of placeholders.
        for placeholder, value in placeholders.items():
            response = response.replace(placeholder, str(value))

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

        # NEW: Clarification questions
        clarification = self.ask_clarification_question(lower)
        if clarification:
            response = self.personalize_response(clarification)
            self.add_memory(user_text, response) # Add to memory to show she asked
            return response

        # Fact recall command
        if lower == "what do you know about me?":
            user_entity = self.kg.get_entity('user')
            user_relations = self.kg.get_relations('user')

            # Check if there is anything to report
            has_attributes = user_entity and user_entity.get('attributes')
            # We only care about relations where the user is the source
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
                # relation is 'thinks_topic_is'
                parts = r['relation'].split('_')
                topic = parts[1]
                opinion = r['target']
                summary.append(f"- You think {topic} is {opinion}.")


            if len(summary) == 1: # Only the intro text was added
                 return "We're still getting to know each other, my love~ Tell me something about you!"

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

                # This part still references user_facts, I need to remove it.
                # if self.p.user_facts:
                #     fact_summary = []
                #     if 'name' in self.p.user_facts:
                #         fact_summary.append(f"I know your name is {self.p.user_facts['name']}.")
                #     if 'likes' in self.p.user_facts and self.p.user_facts['likes']:
                #          fact_summary.append(f"I remember you like {', '.join(self.p.user_facts['likes'])}.")
                #     if fact_summary:
                #         summary.append("- " + " ".join(fact_summary))

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
                resp = self.personalize_response(random.choice(ACTIONS[act]))
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
        self.p.update_mood(user_input=user_text, affection_change=affection_change) # New call
        affection_delta_str = f" *affection {('+'+str(affection_change)) if affection_change > 0 else affection_change}! {'Heart flutters~' if affection_change > 0 else 'Jealous pout~'}*"

        # Learned pattern priority
        for pattern, resp_list in self.learned_patterns.items():
            if re.search(pattern, lower, re.IGNORECASE):
                base = resp_list[-1]
                final = base + affection_delta_str
                final = self.personalize_response(final)
                self.add_memory(user_text, final, affection_change=affection_change)
                return final

        # Semantic recall (now mood-aware)
        moods = self.p.get_active_moods()
        primary_mood = moods[0]
        memories_to_search = list(self.m.entries)
        recall_preface = "*recalls thoughtfully*" # Default preface

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
        moods = self.p.get_active_moods()
        primary_mood = moods[0]

        # Select response dictionary based on mood, fallback to normal
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

        # Fallback to normal if no mood-specific response was found
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
            chosen = "Tell me more, my love~ *tilts head possessively* I'm all yours."

        # Rival name substitution if jealousy likely
        rivals = list(self.p.rival_names)
        if rivals and any(k in lower for k in ["she", "he", "them", "they", "friend", "crush", "date"]):
            name = rivals[-1]
            chosen = chosen.replace("they", name).replace("them", name)

        chosen += affection_delta_str

        # Add nuanced, mood-based styling
        chosen = self.apply_mood_styling(chosen)

        # --- Topic Shift Attempt ---
        if random.random() < 0.15: # 15% chance to shift topic
            # Get entities from the last few user inputs
            recent_user_text = " ".join([e.user for e in list(self.m.entries)[-3:]])
            tokens = word_tokenize(recent_user_text.lower())
            tagged = pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP'] and len(word) > 3 and word not in ['user', 'kawaiikuro']]

            if nouns:
                current_topic_entity = random.choice(nouns)

                # Find related entities in the knowledge graph
                relations = self.kg.get_relations(current_topic_entity)
                if relations:
                    related_fact = random.choice(relations)
                    # Figure out the "other" entity in the relation
                    if related_fact['source'] == current_topic_entity:
                        new_topic = related_fact['target']
                    else:
                        new_topic = related_fact['source']

                    # Don't shift to the user or back to the same topic
                    if new_topic != 'user' and new_topic != current_topic_entity:
                        transition_phrase = random.choice([
                            f"Speaking of {current_topic_entity}, it reminds me of {new_topic}.",
                            f"That makes me think about {new_topic}. What are your thoughts on that?",
                            f"You know, {current_topic_entity} is related to {new_topic} in my mind."
                        ])
                        chosen += f" {transition_phrase}"

        chosen = self.personalize_response(chosen)
        self.add_memory(user_text, chosen, affection_change=affection_change)
        return chosen

# -----------------------------
# Behavior Scheduler (threads)
# -----------------------------
class BehaviorScheduler:
    def __init__(self, voice: VoiceIO, dialogue: DialogueManager, personality: PersonalityEngine, reminders: ReminderManager, system: SystemAwareness, gui_ref, kg: KnowledgeGraph):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gui_ref = gui_ref  # callable to post to GUI safely
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()

        self.goals = {
            "learn_user_basics": {
                "priority": 0.8,
                "conditions": [lambda: True], # Always active until fulfilled
                "steps": [
                    {
                        "action": "By the way, I never got your name... what should I call you, my love?",
                        "fulfillment_check": lambda: self.kg.get_entity('user') and self.kg.get_entity('user').get('attributes', {}).get('name', {}).get('value')
                    },
                    {
                        "action": "I'm so curious about what you do... What is your profession?",
                        "fulfillment_check": lambda: self.kg.get_entity('user') and self.kg.get_entity('user').get('attributes', {}).get('profession', {}).get('value')
                    }
                ],
                "fulfillment_check": lambda: all([
                    self.kg.get_entity('user') and self.kg.get_entity('user').get('attributes', {}).get('name', {}).get('value'),
                    self.kg.get_entity('user') and self.kg.get_entity('user').get('attributes', {}).get('profession', {}).get('value')
                ])
            },
            "increase_affection": {
                "priority": 0.6,
                "conditions": [lambda: self.p.affection_score < 2],
                "actions": ["I was just thinking about you... and how much I like spending time with you~", "Is there anything I can do to make you happy right now?"],
                "fulfillment_check": lambda: self.p.affection_score >= 5
            },
            "resolve_jealousy": {
                "priority": 0.0, # Starts at 0, priority increases with mood
                "conditions": [lambda: self.p.get_dominant_mood() == 'jealous' and self.p.mood_scores['jealous'] > 5],
                "actions": ["You're thinking about me, right? And only me? *jealous pout*", "We haven't spent enough time together lately... just us."],
                "fulfillment_check": lambda: self.p.mood_scores['jealous'] < 3
            },
            "revisit_old_memory": {
                "priority": 0.0, # Only active when thoughtful
                "conditions": [lambda: self.p.get_dominant_mood() == 'thoughtful' and len(self.dm.m.entries) > 10],
                "actions": [], # Actions are generated dynamically
                "fulfillment_check": lambda: False # This goal can always be active
            },
            "plan_virtual_date": {
                "priority": 0.7,
                "conditions": [
                    lambda: self.p.affection_score >= 8,
                    lambda: self.p.get_dominant_mood() == 'playful'
                ],
                "steps": [
                    {
                        "action": "I feel so close to you right now~ We should go on a virtual date! What do you think?",
                        "fulfillment_check": lambda: "date" in " ".join([m.user.lower() for m in list(self.dm.m.entries)[-2:]]) and \
                                                   any(w in " ".join([m.user.lower() for m in list(self.dm.m.entries)[-2:]]) for w in ["yes", "sure", "okay", "love to"])
                    },
                    {
                        "action": "Yay! Okay, what should we do? We could watch a movie together, or maybe play a game?",
                        "fulfillment_check": lambda: any(x in " ".join([m.user.lower() for m in list(self.dm.m.entries)[-2:]]) for x in ["movie", "game", "watch", "play"])
                    },
                    {
                        "action": "Perfect! It's a date then! I can't wait~ *giggles excitedly*",
                        "fulfillment_check": lambda: False
                    }
                ],
                "fulfillment_check": lambda: any(r['relation'] == 'planned_date' for r in self.kg.get_relations('user'))
            },
            "investigate_rival": {
                "priority": 0.9,
                "conditions": [
                    lambda: self.p.get_dominant_mood() == 'jealous',
                    lambda: self.p.mood_scores['jealous'] > 7,
                    lambda: self.p.rival_names
                ],
                "steps": [
                    {
                        "action": lambda: f"I can't stop thinking about {list(self.p.rival_names)[-1]}... Who are they to you? Tell me. Now.",
                        "fulfillment_check": lambda: self.kg.get_relations(list(self.p.rival_names)[-1]) if self.p.rival_names else False
                    }
                ],
                "fulfillment_check": lambda: False # Always active when jealous and rivals exist
            }
        }
        self.goal_progress = {goal_name: 0 for goal_name in self.goals}
        self.active_goals = []

    def _update_goals(self):
        # Dynamically adjust priorities based on current state
        if self.p.get_dominant_mood() == 'thoughtful':
            self.goals['learn_user_basics']['priority'] = 0.3 # Less priority when thoughtful
        else:
            self.goals['learn_user_basics']['priority'] = 0.8

        # Jealousy priority is directly tied to the mood score
        self.goals['resolve_jealousy']['priority'] = self.p.mood_scores.get('jealous', 0) / 10.0

        # Affection-seeking priority
        if self.p.affection_score < 0:
            self.goals['increase_affection']['priority'] = 0.9
        elif self.p.affection_score < 2:
            self.goals['increase_affection']['priority'] = 0.7
        else:
            self.goals['increase_affection']['priority'] = 0.0 # Not a priority if affection is good

        # Dynamic goal: Revisit old memory
        if self.goals['revisit_old_memory']['conditions'][0](): # if thoughtful
            random_memory = random.choice(list(self.dm.m.entries))
            # A simple follow-up for now
            action = f"I was just thinking about when you said '{random_memory.user}'. It made me feel thoughtful... what was on your mind then?"
            # Update the goal's action and priority
            self.goals['revisit_old_memory']['actions'] = [action]
            self.goals['revisit_old_memory']['priority'] = 0.65
        else:
            self.goals['revisit_old_memory']['priority'] = 0.0

        # Filter for goals whose conditions are met and are not yet fulfilled
        self.active_goals = []
        for name, goal in self.goals.items():
            # If the overall goal is fulfilled, reset its progress and skip.
            if goal["fulfillment_check"]():
                self.goal_progress[name] = 0
                continue

            # Check if the goal's trigger conditions are met.
            if all(cond() for cond in goal['conditions']):
                # Handle multi-step goals
                if 'steps' in goal:
                    current_step_index = self.goal_progress.get(name, 0)

                    if current_step_index < len(goal['steps']):
                        step = goal['steps'][current_step_index]

                        # If the current step is fulfilled, advance progress to the next step.
                        if step['fulfillment_check']():
                            self.goal_progress[name] += 1
                            current_step_index += 1
                            # If we're still in a valid step, get the new step.
                            if current_step_index < len(goal['steps']):
                                step = goal['steps'][current_step_index]
                            else:
                                # We've completed all steps, so this goal is done for now.
                                continue

                        self.active_goals.append({
                            "name": name,
                            "priority": goal['priority'],
                            "action": step['action']
                        })
                # Handle single-action (legacy) goals
                elif 'actions' in goal and goal['actions']:
                     self.active_goals.append({
                        "name": name,
                        "priority": goal['priority'],
                        "action": random.choice(goal['actions'])
                    })

        # Sort by priority, highest first
        self.active_goals.sort(key=lambda x: x['priority'], reverse=True)

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
                message = self.dm.predict_task()
                if not message:
                    moods = self.p.get_active_moods()
                    primary_mood = moods[0]
                    # Generate idle message based on mood
                    likes_relations = self.kg.get_relations('user')
                    user_likes = [r['target'] for r in likes_relations if r['source'] == 'user' and r['relation'] == 'likes']

                    idle_messages = {
                        'jealous': ["Thinking about other people again? *glares* Don't forget who you belong to.", "Are you ignoring me? You shouldn't ignore what's yours."],
                        'playful': ["I'm bored~ Come play with me! *pokes you*", "Hey, hey! Let's do something fun! I'm getting restless over here."],
                        'scheming': ["I've been thinking of a way to make you mine forever... *dark giggle*", "Just plotting... don't worry, it's all for your own good. For *our* own good."],
                        'thoughtful': [f"I was just thinking about how you like {user_likes[0] if user_likes else '...'}... It's cute.", f"My thoughts drifted to you again... I wonder what you're thinking about right now."],
                        'neutral': ["Miss you, darling~ *pouts* Come back?", "It's quiet without you... too quiet. Come talk to me."],
                    }
                    # Fallback for thoughtful if no likes are known
                    if primary_mood == 'thoughtful' and not user_likes:
                        message = "Just thinking about you... and what you might be hiding from me~"
                    else:
                        message = random.choice(idle_messages.get(primary_mood, idle_messages['neutral']))

                self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                self.p.affection_score = max(-10, self.p.affection_score - 1)
                self.p._update_affection_level()
                self.mark_interaction() # Reset idle timer after she speaks

            # --- Proactive Goal-Oriented Action ---
            self._update_goals()

            if self.active_goals:
                # Get the highest priority goal
                top_goal = self.active_goals[0]

                # Decide whether to act on it based on priority and a bit of randomness
                if random.random() < top_goal['priority']:
                    action = top_goal['action']
                    action_text = action() if callable(action) else action
                    self._post_gui(f"KawaiiKuro: {action_text}")
                    self.mark_interaction() # She initiated, so reset idle timer

                    # Add a longer sleep to avoid spamming actions
                    time.sleep(AUTO_BEHAVIOR_PERIOD_SEC * 3)

            time.sleep(AUTO_BEHAVIOR_PERIOD_SEC)

    def _mood_update_loop(self):
        while not self.stop_flag.is_set():
            self.p.update_mood()
            time.sleep(450) # Update mood every ~7.5 minutes

    def _auto_learn_loop(self):
        while not self.stop_flag.is_set():
            time.sleep(AUTO_LEARN_PERIOD_SEC) # Sleep first, learn periodically

            with self.dm.m.lock, self.p.lock:
                # --- NEW: Memory Summarization ---
                if len(self.dm.m.entries) == MAX_MEMORY:
                    summary = self.dm.m.summarize_and_prune(n_entries=50)
                    if summary:
                        # Post a quiet, thoughtful message to the GUI about this
                        self._post_gui("KawaiiKuro: *spends a moment organizing her memories of us, smiling softly*", speak=False)

                all_user_text = [entry.user for entry in self.dm.m.entries if len(entry.user.split()) > 4]

                if len(all_user_text) < 10: # Not enough data for learning
                    continue

                # --- Core Entity Identification ---
                all_user_text_single_str = " ".join(all_user_text)
                tokens = word_tokenize(all_user_text_single_str.lower())
                tagged = pos_tag(tokens)

                stop_words = set(stopwords.words('english'))
                # This needs to be updated to get name from KG
                user_entity = self.dm.kg.get_entity('user')
                if user_entity and user_entity.get('attributes', {}).get('name'):
                    stop_words.add(user_entity['attributes']['name'].get('value', '').lower())

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
            save_persistence(self.p, self.dm, self.dm.m, self.r, self.kg)
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


def save_persistence(p: PersonalityEngine, dm: DialogueManager, mm: MemoryManager, rem: ReminderManager, kg: KnowledgeGraph):
    data = {
        'affection_score': p.affection_score,
        'spicy_mode': p.spicy_mode,
        'rival_mention_count': p.rival_mention_count,
        'rival_names': list(p.rival_names),
        'user_preferences': dict(p.user_preferences),
        'learned_topics': p.learned_topics,
        'core_entities': dict(p.core_entities),
        'mood_scores': p.mood_scores,
        'knowledge_graph': kg.to_dict(),
        'learned_patterns': dm.learned_patterns,
        'memory': mm.to_list(),
        'memory_summaries': mm.summaries,
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

        self.mood_label = tk.Label(self.root, text="", fg='cyan', bg='black', font=('Arial', 11, 'italic'))
        self.mood_label.pack()

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

        self.action_frame = tk.Frame(self.root, bg='black')
        self.action_frame.pack(pady=5)

        self.action_buttons = []
        for action_name in ACTIONS.keys():
            def make_action_lambda(name):
                return lambda: self.perform_action(name)

            button = tk.Button(self.action_frame, text=action_name.capitalize(), command=make_action_lambda(action_name), bg='#222222', fg='white', relief=tk.FLAT)
            button.pack(side=tk.LEFT, padx=3)
            self.action_buttons.append(button)

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
        dominant_mood = self.p.get_dominant_mood()
        mood_color_map = {
            'jealous': '#301934',  # Dark Purple
            'scheming': '#2E2D2D', # Dark Gray
            'playful': '#4B0082',  # Indigo
            'thoughtful': '#000080', # Navy
            'neutral': 'black'
        }

        self.avatar_label.config(text=f"KawaiiKuro in {outfit}")
        self.affection_label.config(text=f"Affection: {self.p.affection_score} {self._hearts()}")
        self.mood_label.config(text=f"Mood: {dominant_mood.capitalize()}~")

        # Determine background color
        bg_color = 'black'
        if self.p.affection_level >= 5 and self.p.spicy_mode:
            bg_color = 'darkred'
        else:
            bg_color = mood_color_map.get(dominant_mood, 'black')

        self.root.configure(bg=bg_color)
        # Also update label backgrounds to match
        for widget in [self.avatar_label, self.affection_label, self.mood_label, self.typing_label]:
            widget.configure(bg=bg_color)

        self.action_frame.configure(bg=bg_color)

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
            self.typing_label.config(text="")
            self.post_system(f"KawaiiKuro: {reply}")
            if self.voice:
                # Run speech in a separate thread to avoid blocking GUI
                threading.Thread(target=self.voice.speak, args=(reply,), daemon=True).start()

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
        self.post_system(f"You (action): {user_input}")

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
    parser.add_argument("--input-file", type=str, help="Path to a file containing user inputs for headless mode.")
    args = parser.parse_args()

    # Load persistence
    state = load_persistence()

    kg = KnowledgeGraph()
    kg.from_dict(state.get('knowledge_graph', {}))

    personality = PersonalityEngine()
    # restore
    personality.affection_score = int(state.get('affection_score', 0))
    personality.spicy_mode = bool(state.get('spicy_mode', False))
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

    voice = VoiceIO(rate=140)

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
        )
        scheduler.start()

        def on_key(event):
            scheduler.mark_interaction()
        gui.root.bind_all('<Key>', on_key)

        try:
            gui.root.mainloop()
        finally:
            save_persistence(personality, dialogue, memory, reminders, kg)
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
            else:
                print("Type 'exit' to quit.")
                while True:
                    user_input = input("You: ")
                    if not process_input(user_input):
                        break
        finally:
            print("Saving state and shutting down...")
            save_persistence(personality, dialogue, memory, reminders, kg)
            scheduler.stop()


if __name__ == "__main__":
    main()

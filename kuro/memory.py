import threading
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kuro.config import MAX_MEMORY, MIN_RECALL_SIM
from kuro.utils import safe_word_tokenize, safe_pos_tag, safe_stopwords
import re

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

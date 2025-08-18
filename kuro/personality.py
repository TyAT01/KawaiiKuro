import re
from collections import Counter
from datetime import datetime
from typing import List, Dict
import threading

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from kuro.config import OUTFITS_BASE, SAFE_PERSON_NAME_STOPWORDS
from kuro.utils import safe_word_tokenize, safe_pos_tag

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
            if primary_mood == 'scheming':
                return f"{base_outfit}, shrouded in a mysterious dark veil... *dark giggle*"
            if primary_mood == 'playful' and self.affection_level >= 3:
                return f"{base_outfit}, accented with playful ribbons and bells~ *winks*"
            if primary_mood == 'thoughtful':
                return f"{base_outfit}, with a pair of nerdy-cute reading glasses perched on her nose."
            if primary_mood == 'curious':
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

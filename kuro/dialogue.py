import re
import random
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

from kuro.utils import safe_word_tokenize, safe_sent_tokenize, safe_pos_tag, safe_stopwords
from kuro.memory import MemoryEntry
from kuro.utils import MathEvaluator

# Forward declarations
class PersonalityEngine:
    pass

class MemoryManager:
    pass

class ReminderManager:
    pass

class KnowledgeGraph:
    pass

class DialogueManager:
    def __init__(self, personality: PersonalityEngine, memory: MemoryManager, reminders: ReminderManager, math_eval: MathEvaluator, kg: KnowledgeGraph, web_search = None):
        self.p = personality
        self.m = memory
        self.r = reminders
        self.math = math_eval
        self.kg = kg
        self.web_search = web_search
        self.learned_patterns: Dict[str, List[str]] = {}
        self.pending_relation: Optional[Dict[str, Any]] = None
        self.current_topic: Optional[str] = None
        self.conversation_turn_on_topic: int = 0
        self.last_clarification_time = 0
        self.recently_recalled_facts = deque(maxlen=5)
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
        moods = self.p.get_active_moods()
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

        templates = [
            f"Did I hear you correctly... that {subject} {verb} {obj}? *tilts head curiously*",
            f"My nerdy brain is trying to connect the dots... are you saying that {subject} {verb} {obj}? *looks at you thoughtfully*",
            f"I might be jumping to conclusions, but does that mean {subject} {verb} {obj}?",
            f"Just so I have it right in my notes... you're saying {subject} {verb} {obj}?",
        ]

        question = random.choice(templates)
        return self.personalize_response(question)

    def _recall_relevant_fact(self, text: str) -> Optional[str]:
        """Checks the KG for facts related to the input text and returns a formatted string."""
        lower_text = text.lower()

        # Expand topic search beyond just the last input
        with self.m.lock:
            recent_texts = " ".join([m.user for m in list(self.m.entries)[-3:]])

        full_context_text = lower_text + " " + recent_texts
        tokens = safe_word_tokenize(full_context_text)
        tagged = safe_pos_tag(tokens)

        # Extract nouns as potential topics
        topics = {word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP'] and len(word) > 3 and word not in safe_stopwords()}
        if not topics:
            return None

        # Search for a relevant, not-recently-recalled fact
        for topic in list(topics):
            relations = self.kg.get_relations(topic)
            if not relations: continue

            random.shuffle(relations)
            for r in relations:
                fact_tuple = (r['source'], r['relation'], r['target'])
                if fact_tuple in self.recently_recalled_facts: continue

                self.recently_recalled_facts.append(fact_tuple)
                source, rel, target = fact_tuple

                recall_templates = {
                    "likes": [
                        f"That reminds me, I remember you like {target}~",
                        f"Speaking of which, you like {target}, right? I remember that.",
                    ],
                    "dislikes": [
                        f"Oh, right, you don't like {target}. I'll keep that in mind.",
                        f"I remember you're not a fan of {target}.",
                    ],
                    "favorite": [
                        f"Ooh, that makes me think of your favorite {rel.replace('favorite_', '')}, {target}!",
                        f"Isn't your favorite {rel.replace('favorite_', '')} {target}?",
                    ],
                    "default": [
                        f"By the way, that reminds me that {source} {rel.replace('_', ' ')} {target}.",
                        f"Thinking about that, I recall learning that {source} {rel.replace('_', ' ')} {target}.",
                    ]
                }

                template_key = "default"
                if source == 'user':
                    if rel.startswith('favorite_'): template_key = 'favorite'
                    elif rel in recall_templates: template_key = rel

                return f" {random.choice(recall_templates[template_key])}"

        return None

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

        # Increase probability of shifting topic the longer we're on it
        shift_probability = min(0.15 * self.conversation_turn_on_topic, 0.7) # 15% chance per turn, max 70%

        if self.current_topic and random.random() < shift_probability:
            relations = self.kg.get_relations(self.current_topic)
            if relations:
                # Find a new topic that hasn't been the current topic recently
                with self.m.lock:
                    recent_topics = {m.keywords[0] for m in list(self.m.entries)[-5:] if m.keywords}

                potential_new_topics = []
                for r in relations:
                    new_topic = r['target'] if r['source'] == self.current_topic else r['source']
                    if new_topic != 'user' and new_topic != self.current_topic and new_topic not in recent_topics:
                        potential_new_topics.append(new_topic)

                if potential_new_topics:
                    new_topic = random.choice(potential_new_topics)
                    transition_phrase = random.choice([
                        f"Speaking of {self.current_topic}, it makes me think of {new_topic}.",
                        f"That reminds me of {new_topic}. What are your thoughts on that?",
                        f"Changing the subject a little, but that makes me wonder about {new_topic}..."
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

            CLARIFICATION_COOLDOWN = 120 # 2 minutes
            can_ask = (time.time() - self.last_clarification_time) > CLARIFICATION_COOLDOWN

            # Don't ask if confidence is too high or if we asked too recently, and add some randomness
            if relation_to_clarify['confidence'] < 0.9 and can_ask and random.random() < 0.7:
                self.last_clarification_time = time.time()
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
            resp = f"{recall_preface} You said '{recalled}' before~ Still on that?"
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
            # --- Final fallback: Web Search for questions ---
            question_starters = ["what is", "what's", "who is", "who's", "where is", "where's", "why is", "how does", "can you explain", "what does"]
            is_question = any(lower.startswith(s) for s in question_starters)

            if is_question and self.web_search:
                query = lower
                search_result = self.web_search.search_and_summarize(query)
                if search_result:
                    chosen = f"*looks up '{query}' on a dusty, arcane terminal...*\n\nHere's what I found: {search_result}"
                else:
                    chosen = f"I tried to search for '{query}' but came up empty... My apologies, my love."
                affection_delta_str = "" # No affection change for web searches
            else:
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
        # Append a relevant fact from the knowledge graph
        if random.random() < 0.6: # 60% chance to add a fact
            recalled_fact = self._recall_relevant_fact(lower)
            if recalled_fact:
                # Avoid adding to very short responses like "Okay."
                if len(chosen.split()) > 3:
                    chosen += recalled_fact

        chosen += affection_delta_str
        chosen = self.apply_mood_styling(chosen)
        chosen = self._maybe_shift_topic(chosen, user_text)
        chosen = self.personalize_response(chosen)
        self.add_memory(user_text, chosen, affection_change=affection_change)
        return chosen

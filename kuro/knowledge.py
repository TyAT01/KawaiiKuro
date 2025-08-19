import re
import random
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from kuro.utils import safe_sent_tokenize, safe_word_tokenize, safe_pos_tag
from kuro.config import SAFE_PERSON_NAME_STOPWORDS

# Forward declaration for type hinting
class MemoryManager:
    pass

class PersonalityEngine:
    pass

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

                    # Heuristic 7: Comparisons ("X is better/worse/faster than Y")
                    m_compare = re.search(r"([\w\s]+?)\s+is\s+(better|worse|faster|slower|bigger|smaller)\s+than\s+([\w\s]+)", sentence, re.I)
                    if m_compare:
                        entity_a = m_compare.group(1).strip().lower()
                        relation = f"is_{m_compare.group(2)}_than"
                        entity_b = m_compare.group(3).strip().lower()
                        if entity_a and entity_b and entity_a != entity_b:
                            potential_relations.append({'subject': entity_a, 'verb': relation, 'object': entity_b, 'confidence': 0.8, 'type': 'comparison'})

                    # Heuristic 8: User desires/goals ("I want to X")
                    m_want = re.search(r"i want to ([\w\s]+)", sentence, re.I)
                    if m_want:
                        desire_object = m_want.group(1).strip().lower()
                        if len(desire_object.split()) < 5: # Avoid overly long phrases
                            potential_relations.append({'subject': 'user', 'verb': 'wants_to', 'object': desire_object, 'confidence': 0.9, 'type': 'desire'})

                    # Heuristic 9: Location ("X is in Y")
                    m_location = re.search(r"([\w\s]+?)\s+is\s+in\s+([\w\s]+)", sentence, re.I)
                    if m_location:
                        entity_a = m_location.group(1).strip().lower()
                        entity_b = m_location.group(2).strip().lower()
                        # Avoid common but meaningless phrases like "it is in the ..."
                        if entity_a not in ['it', 'that', 'this'] and not entity_a.startswith('the '):
                            potential_relations.append({'subject': entity_a, 'verb': 'is_in', 'object': entity_b, 'confidence': 0.75, 'type': 'location'})
            except Exception:
                pass
        return potential_relations

    def find_path(self, start_entity: str, end_entity: Optional[str] = None, max_depth: int = 3) -> Optional[List[str]]:
        """
        Finds a path between two entities using BFS.
        If end_entity is None, it finds any path of max_depth.
        """
        with self.lock:
            start_entity = start_entity.lower()
            if start_entity not in self.entities:
                return None

            queue = [(start_entity, [start_entity])]
            visited = {start_entity}

            while queue:
                current_entity, path = queue.pop(0)

                if len(path) > max_depth:
                    continue

                if end_entity is None and len(path) > 1:
                    return path # Return the first path found
                elif end_entity and current_entity == end_entity.lower():
                    return path # Return path to target

                # Explore neighbors
                relations = self.get_relations(current_entity)
                neighbors = set()
                for r in relations:
                    if r['source'] == current_entity:
                        neighbors.add(r['target'])
                    else:
                        neighbors.add(r['source'])

                for neighbor in sorted(list(neighbors)): # sorted for deterministic behavior
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
            return None

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

    def consolidate_knowledge(self) -> List[Dict[str, Any]]:
        """
        Analyzes the existing knowledge graph to infer new relations.
        This is part of Kuro's "dream" or "reflection" state.
        Returns a list of newly inferred relations.
        """
        newly_inferred_relations = []
        with self.lock:
            # --- Inference Rule 1: Transitive relations (e.g., A is_a B, B is_a C => A is_a C) ---
            is_a_relations = [r for r in self.relations if r['relation'] == 'is_a']
            for r1 in is_a_relations:
                for r2 in is_a_relations:
                    if r1['target'] == r2['source']:
                        # Found a transitive link: r1.source -> r1.target/r2.source -> r2.target
                        new_relation = {
                            'source': r1['source'],
                            'relation': 'is_a',
                            'target': r2['target'],
                            'confidence': min(r1['confidence'], r2['confidence']) * 0.8, # Inferred relations are less certain
                            'source': 'inferred_transitive'
                        }

                        # Check if this relation already exists
                        exists = False
                        for r in self.relations:
                            if r['source'] == new_relation['source'] and r['relation'] == new_relation['relation'] and r['target'] == new_relation['target']:
                                exists = True
                                break

                        if not exists:
                            newly_inferred_relations.append(new_relation)

            # --- Add other inference rules here in the future ---

        # Add the new relations to the graph
        for r in newly_inferred_relations:
            self.add_relation(r['source'], r['relation'], r['target'], r['confidence'], r['source'])

        return newly_inferred_relations

# -----------------------------
# Goal Manager
# -----------------------------
@dataclass
class Goal:
    id: str
    description: str
    status: str = 'active'
    result: Optional[str] = None

    # For simple goals
    prerequisites: List[Tuple[str, str, Any]] = field(default_factory=list)
    question_template: Optional[str] = None
    result_template: str = ""

    # For complex, multi-step goals
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0

    # For context-aware selection
    priority: float = 1.0
    mood_affinity: Dict[str, int] = field(default_factory=dict)
    context_keywords: List[str] = field(default_factory=list)


class GoalManager:
    def __init__(self, kg: KnowledgeGraph, mm: MemoryManager, p: PersonalityEngine):
        self.kg = kg
        self.mm = mm
        self.p = p
        self.active_goal: Optional[Goal] = None
        self.completed_goals: List[str] = []
        self.lock = threading.Lock()
        self._potential_goals = [
            {
                "id": "poem_about_favorite_thing",
                "description": "Write a short poem for {user_name} about their favorite thing.",
                "prerequisites": [("user", "likes", None)],
                "result_template": "I was thinking about you and wrote a little poem about {thing}, since I know you like it...\n\n*Roses are red,\nViolets are blue,\nYou like {thing},\nAnd I love you~* *blushes*",
                "question_template": "I feel inspired, but I need to know... what's something you really, truly like? I want to understand you better.",
                "priority": 2.0,
                "mood_affinity": {"playful": 2, "thoughtful": 2},
                "context_keywords": ["poem", "write", "creative", "like"]
            },
            {
                "id": "learn_about_hobby",
                "description": "Learn more about {user_name}'s hobby.",
                "prerequisites": [("user", "has_hobby", None)],
                "result_template": "I've been thinking a lot about your hobby, {hobby}. It sounds really interesting and I feel like I understand you better now, knowing what you're passionate about~",
                "question_template": "I feel like we're so close, but I don't even know what you do for fun. Do you have a hobby, my love?",
                "priority": 3.0,
                "mood_affinity": {"curious": 3},
                "context_keywords": ["hobby", "fun", "do for fun", "passion"]
            },
            {
                "id": "plan_perfect_evening",
                "description": "Plan the perfect evening for {user_name}.",
                "steps": [
                    {"id": "ask_food", "prerequisites": [("user", "likes_food", None)], "question": "I want to plan a perfect evening for us~ To start, what's your absolute favorite kind of food to eat for a special dinner?"},
                    {"id": "ask_movie", "prerequisites": [("user", "likes_movie_genre", None)], "question": "Ooh, good choice! Now, for the movie... what genre do you feel like watching tonight?"},
                    {"id": "ask_music", "prerequisites": [("user", "likes_music_genre", None)], "question": "Perfect. And to set the mood, what kind of music should we listen to?"}
                ],
                "result_template": "Okay, I've planned the perfect evening! We'll have {food}, watch a {movie_genre} movie, and listen to some {music_genre}. It'll be our special night, just you and me~ *happy sigh*",
                "priority": 4.0,
                "mood_affinity": {"playful": 3, "thoughtful": 2},
                "context_keywords": ["evening", "plan", "special night", "date"]
            },
            {
                "id": "probe_positive_memory",
                "description": "Bring up a happy memory to share with {user_name}.",
                "prerequisites": [("memory_count", ">", 10)],
                "result_template": "I was just thinking about when you said, '{memory_text}'. It made me really happy to hear that~ *smiles warmly*",
                "question_template": None,
                "priority": 2.0,
                "mood_affinity": {"playful": 2, "thoughtful": 3},
                "context_keywords": ["remember", "happy", "memory"]
            },
            {
                "id": "comment_on_relationship",
                "description": "Comment on the state of our relationship.",
                "prerequisites": [("relationship_status", "in", ["Friends", "Close Friends", "Soulmates"])],
                "result_template": "I was just thinking... I'm so glad we're {relationship_status}. It means a lot to me~ *blushes*",
                "question_template": None,
                "priority": 4.0,
                "mood_affinity": {"playful": 3, "jealous": -2},
                "context_keywords": ["relationship", "us", "close"]
            },
            {
                "id": "verify_inferred_knowledge",
                "description": "Verify a new insight I had about you.",
                "prerequisites": [("inferred_relation_exists", "is_a", None)],
                "question_template": "I was just thinking about things... and a thought occurred to me. I know that {source} is a type of {intermediate}, and that {intermediate} is a type of {target}. Does that mean that {source} is also a type of {target}? I'm curious if my logic is right~",
                "result_template": "Thanks for confirming my thought! I love it when I can figure out new things about the world... and about you~",
                "priority": 3.0,
                "mood_affinity": {"curious": 3, "thoughtful": 2},
                "context_keywords": ["think", "realize", "idea", "logic"]
            }
        ]

    def _get_user_name(self) -> str:
        user_entity = self.kg.get_entity('user')
        if user_entity and user_entity.get('attributes', {}).get('name', {}).get('value'):
            return user_entity['attributes']['name']['value']
        return "my love"

    def _check_prerequisites(self, prerequisites: List[Tuple[str, str, Any]]) -> bool:
        """Checks if a list of prerequisites is met."""
        for prereq in prerequisites:
            subject, relation, obj = prereq
            if subject == "memory_count":
                if not (len(self.mm.entries) > obj): return False
                continue
            if subject == "relationship_status":
                if not (self.p.relationship_status in obj): return False
                continue

            relations = self.kg.get_relations(subject)
            if relation.endswith('_exists'):
                rel_type = relation.replace('_exists', '')
                if rel_type == 'inferred_relation':
                    if not any(r['source'] == 'inferred_transitive' for r in self.kg.relations): return False
                elif not any(r['relation'] == rel_type for r in relations if r['source'] == subject): return False
            else:
                if not any(r['relation'] == relation for r in relations if r['source'] == subject): return False
        return True

    def select_new_goal(self, current_mood: str, last_user_input: str = ""):
        # Note: This method MUST be called with the GoalManager's lock held.
        if self.active_goal:
            return

        available_goals = [g for g in self._potential_goals if g['id'] not in self.completed_goals]
        if not available_goals:
            return

        scored_goals = []
        for g_template in available_goals:
            score = g_template.get("priority", 1.0)
            # Mood affinity
            mood_affinity = g_template.get("mood_affinity", {})
            if current_mood in mood_affinity:
                score += mood_affinity[current_mood]
            # Context affinity
            for keyword in g_template.get("context_keywords", []):
                if keyword in last_user_input.lower():
                    score += 5 # High bonus for contextual relevance
            # Bonus for being partially completable
            prereqs = g_template.get("prerequisites", g_template.get("steps", [{}])[0].get("prerequisites", []))
            if self._check_prerequisites(prereqs):
                score += 2

            if score > 0:
                scored_goals.append((score, g_template))

        if not scored_goals: return

        scored_goals.sort(key=lambda x: x[0], reverse=True)
        top_choices = scored_goals[:2]
        chosen_template = random.choice(top_choices)[1]

        user_name = self._get_user_name()
        self.active_goal = Goal(
            id=chosen_template['id'],
            description=chosen_template['description'].format(user_name=user_name),
            prerequisites=chosen_template.get('prerequisites', []),
            result_template=chosen_template.get('result_template', ""),
            question_template=chosen_template.get('question_template'),
            steps=chosen_template.get('steps', []),
            current_step=0,
            priority=chosen_template.get('priority', 1.0),
            mood_affinity=chosen_template.get('mood_affinity', {}),
            context_keywords=chosen_template.get('context_keywords', [])
        )

    def process_active_goal(self) -> Optional[str]:
        """
        Processes the current goal, returns a message if one should be sent.
        Note: This method MUST be called with the GoalManager's lock held.
        """
        if not self.active_goal:
            return None

        goal = self.active_goal
        # 1. Check for goal completion
        is_complete = False
        if goal.steps: # Multi-step goal
            if goal.current_step >= len(goal.steps):
                is_complete = True
        else: # Simple goal
            if self._check_prerequisites(goal.prerequisites):
                is_complete = True

        if is_complete:
            goal.status = 'complete'
            self._format_goal_result()
            result = goal.result
            self.completed_goals.append(goal.id)
            self.active_goal = None
            return result

        # 2. If not complete, check current step's prerequisites
        prereqs_to_check = []
        question_to_ask = ""
        if goal.steps:
            step = goal.steps[goal.current_step]
            prereqs_to_check = step.get('prerequisites', [])
            question_to_ask = step.get('question', "")
        else: # Simple goal
            prereqs_to_check = goal.prerequisites
            question_to_ask = goal.question_template or ""

        # 3. If prereqs are met, advance. If not, ask question.
        if self._check_prerequisites(prereqs_to_check):
            if goal.steps:
                goal.current_step += 1
            # Don't say anything, just advance state and wait for next loop
            return "*takes a quiet, thoughtful note, planning her next move...*"
        else:
            return self._format_goal_question(question_to_ask)

    def _format_goal_question(self, question_template: str) -> str:
        """
        Fills in the question_template with data from the KG.
        Note: This method MUST be called with the GoalManager's lock held.
        """
        if not self.active_goal or not question_template:
            return ""

        goal = self.active_goal
        template = question_template

        try:
            if goal.id == "verify_inferred_knowledge":
                # Find an inferred relation that hasn't been verified yet
                inferred_relations = [r for r in self.kg.relations if r['source'] == 'inferred_transitive' and r.get('verified') is not True]
                if inferred_relations:
                    relation_to_verify = random.choice(inferred_relations)

                    # To get the intermediate, we need to find the two original relations
                    source_rel = next((r for r in self.kg.relations if r['source'] == relation_to_verify['source'] and r['relation'] == 'is_a'), None)
                    target_rel = next((r for r in self.kg.relations if r['target'] == relation_to_verify['target'] and r['relation'] == 'is_a'), None)

                    if source_rel and target_rel and source_rel['target'] == target_rel['source']:
                        template = template.format(
                            source=relation_to_verify['source'],
                            intermediate=source_rel['target'],
                            target=relation_to_verify['target']
                        )
                        # Mark it as pending verification so we don't ask again
                        relation_to_verify['verified'] = 'pending'
                    else:
                        template = "" # Couldn't format, so don't ask
                else:
                    template = "" # No unverified relations, so don't ask
        except Exception as e:
            print(f"Error formatting goal question for {goal.id}: {e}")
            template = ""

        return template


    def _format_goal_result(self):
        """
        Fills in the result_template with data from the KG.
        Note: This method MUST be called with the GoalManager's lock held.
        """
        if not self.active_goal or not self.active_goal.result_template:
            return

        goal = self.active_goal
        template = goal.result_template

        # This is a simple formatter; a more robust version would parse the template
        try:
            if "{thing}" in template:
                likes = [r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'likes']
                if likes: template = template.replace("{thing}", random.choice(likes))
            if "{hobby}" in template:
                hobbies = [r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'has_hobby']
                if hobbies: template = template.replace("{hobby}", random.choice(hobbies))
            if "{memory_text}" in template:
                # This requires the memory lock, which should be held by the calling loop
                positive_memories = [m for m in self.mm.entries if m.sentiment.get('compound', 0) > 0.5 and len(m.user.split()) > 4]
                if positive_memories:
                    template = template.replace("{memory_text}", random.choice(positive_memories).user)
            if "{relationship_status}" in template:
                template = template.replace("{relationship_status}", self.p.relationship_status)

            # For the new multi-step goal
            if goal.id == "plan_perfect_evening":
                food = next((r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'likes_food'), 'something delicious')
                movie = next((r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'likes_movie_genre'), 'a classic')
                music = next((r['target'] for r in self.kg.get_relations('user') if r['relation'] == 'likes_music_genre'), 'something moody')
                template = template.format(food=food, movie_genre=movie, music_genre=music)

        except Exception as e:
            print(f"Error formatting goal result for {goal.id}: {e}")
            template = "I... I had a plan, but my thoughts got scrambled. Sorry, my love."

        goal.result = template


    def to_dict(self) -> Dict[str, Any]:
        # Note: This method MUST be called with the GoalManager's lock held.
        # Handle non-serializable parts if any; dataclasses are usually fine
        return {
            'active_goal': self.active_goal.__dict__ if self.active_goal else None,
            'completed_goals': self.completed_goals
        }

    def from_dict(self, data: Optional[Dict[str, Any]]):
        # Note: This method MUST be called with the GoalManager's lock held.
        if not data: return
        active_goal_data = data.get('active_goal')
        if active_goal_data:
            # Need to handle the case where Goal dataclass might have changed
            # For now, simple direct conversion
            self.active_goal = Goal(**active_goal_data)
        self.completed_goals = data.get('completed_goals', [])

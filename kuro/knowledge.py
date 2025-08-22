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
        self.processed_learning_files: set[str] = set()
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

                    # Heuristic 10: Concept Definition for Goal Seeding (e.g., "Botany is a field of science")
                    m_concept = re.search(r"([\w\s]+?)\s+is\s+(?:a|an)\s+(?:popular|well-known|interesting|major|)\s*(?:type of|kind of|form of|branch of|field of|study of|)\s*([\w\s]+)", sentence, re.I)
                    if m_concept:
                        subject_phrase = m_concept.group(1).strip().lower()
                        object_phrase = m_concept.group(2).strip().lower()
                        # Avoid overly generic phrases
                        if subject_phrase not in ['it', 'this', 'that', 'he', 'she', 'the'] and object_phrase not in ['thing', 'one', 'person', 'idea']:
                            # Simple cleanup
                            subject_entity = subject_phrase.replace("the ", "").strip()
                            object_entity = object_phrase.replace("the ", "").strip()
                            if len(subject_entity.split()) <= 3 and len(object_entity.split()) <= 3: # Keep phrases reasonably short
                                potential_relations.append({'subject': subject_entity, 'verb': 'is_a', 'object': object_entity, 'confidence': 0.7, 'type': 'definition'})

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
            self.processed_learning_files = set(data.get('processed_learning_files', []))
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
class GoalManager:
    """
    The GoalManager is responsible for the tactical execution of individual plan steps.
    It checks prerequisites for a given step and can generate questions to gather
    necessary information if those prerequisites are not met. It no longer manages
    its own goals, but instead executes steps provided by the Planner.
    """
    def __init__(self, kg: KnowledgeGraph, mm: MemoryManager, p: PersonalityEngine):
        self.kg = kg
        self.mm = mm
        self.p = p
        self.lock = threading.Lock()

    def _check_prerequisites(self, prerequisites: List[Tuple[str, str, Any]]) -> bool:
        """
        Checks if a list of prerequisites against the Knowledge Graph is met.
        Note: This method must be called with the GoalManager's lock held.
        """
        for prereq in prerequisites:
            subject, relation, obj = prereq

            # Find relations for the subject in the knowledge graph
            relations = self.kg.get_relations(subject)

            # Check if a relation of the specified type exists
            if not any(r['relation'] == relation for r in relations if r['source'] == subject):
                return False
        return True

    def process_plan_step(self, step_description: str) -> Optional[str]:
        """
        Processes a single step from the Planner's plan.
        For now, this is a placeholder. It will be expanded to parse steps
        and check complex prerequisites.

        Returns:
            - A question string if information is needed.
            - None if the step can be considered complete or doesn't require a question.
        """
        # This is a simplified placeholder for the logic that will eventually be
        # powered by the LLM in the Planner to define prerequisites for each step.
        # For now, we can imagine some simple hardcoded checks.

        with self.lock:
            if "hobby" in step_description.lower():
                if not self._check_prerequisites([("user", "has_hobby", None)]):
                    return "I want to get to know you better... what do you do for fun? Do you have a hobby?"

            if "favorite food" in step_description.lower():
                if not self._check_prerequisites([("user", "likes_food", None)]):
                    return "If we were to have the perfect dinner, what's your favorite food?"

        # If no prerequisites are missing, the step can proceed without a question.
        return None

    def to_dict(self) -> Dict[str, Any]:
        """The GoalManager is now mostly stateless, so persistence is minimal."""
        return {}

    def from_dict(self, data: Optional[Dict[str, Any]]):
        """The GoalManager is now mostly stateless, so persistence is minimal."""
        pass

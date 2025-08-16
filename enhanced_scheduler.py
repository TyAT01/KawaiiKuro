import random
import time
import threading
from collections import Counter

# To avoid circular dependencies, we will use type hinting with strings for classes from kawaii_kuro
from typing import List, Dict, Any, Optional, Callable

class GoalManager:
    def __init__(self, dialogue_manager, personality, knowledge_graph):
        self.dm = dialogue_manager
        self.p = personality
        self.kg = knowledge_graph
        self.goal_progress = {}
        self.active_goals = []

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
            "learn_user_hobby": {
                "priority": 0.7,
                "conditions": [
                    lambda: self.p.affection_score > 0, # Only ask when the mood is good
                    lambda: 'thoughtful' in self.p.get_active_moods()
                ],
                "steps": [
                    {
                        "action": "When you're not busy, what do you do for fun? I'm curious about your hobbies~",
                        "fulfillment_check": lambda: any(r['relation'] == 'has_hobby' for r in self.kg.get_relations('user'))
                    }
                ],
                "fulfillment_check": lambda: any(r['relation'] == 'has_hobby' for r in self.kg.get_relations('user'))
            },
            "increase_affection": {
                "priority": 0.6,
                "conditions": [lambda: self.p.affection_score < 2],
                "steps": [{
                    "action": lambda: random.choice([
                        "I was just thinking about you... and how much I like spending time with you~",
                        "Is there anything I can do to make you happy right now?"
                    ]),
                    "fulfillment_check": lambda: False
                }],
                "fulfillment_check": lambda: self.p.affection_score >= 5
            },
            "resolve_jealousy": {
                "priority": 0.0, # Starts at 0, priority increases with mood
                "conditions": [lambda: self.p.get_dominant_mood() == 'jealous' and self.p.mood_scores['jealous'] > 5],
                "steps": [{
                    "action": lambda: random.choice([
                        "You're thinking about me, right? And only me? *jealous pout*",
                        "We haven't spent enough time together lately... just us."
                    ]),
                    "fulfillment_check": lambda: False
                }],
                "fulfillment_check": lambda: self.p.mood_scores['jealous'] < 3
            },
            "revisit_old_memory": {
                "priority": 0.0, # Only active when thoughtful
                "conditions": [lambda: self.p.get_dominant_mood() == 'thoughtful' and len(self.dm.m.entries) > 10],
                "steps": [], # Steps are generated dynamically
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
                        "side_effect": lambda: self.kg.add_relation('user', 'planned_date', 'true', confidence=1.0, source='goal_system'),
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
            },
        }
        self.goal_progress = {goal_name: 0 for goal_name in self.goals}


    def evaluate_and_update(self):
        # Dynamically adjust priorities based on current state
        if self.p.get_dominant_mood() == 'thoughtful':
            self.goals['learn_user_basics']['priority'] = 0.3 # Less priority when thoughtful
        else:
            self.goals['learn_user_basics']['priority'] = 0.8

        self.goals['resolve_jealousy']['priority'] = self.p.mood_scores.get('jealous', 0) / 10.0

        if self.p.affection_score < 0:
            self.goals['increase_affection']['priority'] = 0.9
        elif self.p.affection_score < 2:
            self.goals['increase_affection']['priority'] = 0.7
        else:
            self.goals['increase_affection']['priority'] = 0.0

        if self.goals['revisit_old_memory']['conditions'][0](): # if thoughtful
            if self.dm.m.entries:
                random_memory = random.choice(list(self.dm.m.entries))
                action = f"I was just thinking about when you said '{random_memory.user}'. It made me feel thoughtful... what was on your mind then?"
                self.goals['revisit_old_memory']['steps'] = [{"action": action, "fulfillment_check": lambda: False}]
                self.goals['revisit_old_memory']['priority'] = 0.65
        else:
            self.goals['revisit_old_memory']['priority'] = 0.0
            self.goals['revisit_old_memory']['steps'] = []

        # Filter for goals whose conditions are met and are not yet fulfilled
        self.active_goals = []
        for name, goal in self.goals.items():
            if goal["fulfillment_check"]():
                self.goal_progress[name] = 0
                continue

            if all(cond() for cond in goal["conditions"]):
                if 'steps' in goal and goal['steps']:
                    current_step_index = self.goal_progress.get(name, 0)

                    if current_step_index < len(goal['steps']):
                        step = goal['steps'][current_step_index]

                        if step['fulfillment_check']():
                            self.goal_progress[name] += 1
                            current_step_index += 1
                            if current_step_index < len(goal['steps']):
                                step = goal['steps'][current_step_index]
                            else:
                                continue

                        self.active_goals.append({
                            "name": name,
                            "priority": goal['priority'],
                            "action": step['action'],
                            "side_effect": step.get('side_effect')
                        })

        self.active_goals.sort(key=lambda x: x['priority'], reverse=True)
        return self.active_goals


class EnhancedBehaviorScheduler:
    def __init__(self, voice, dialogue, personality, reminders, system, gui_ref, kg, test_mode=False):
        self.voice = voice
        self.dm = dialogue
        self.p = personality
        self.r = reminders
        self.system = system
        self.kg = kg
        self.gui_ref = gui_ref
        self.last_interaction_time = time.time()
        self.stop_flag = threading.Event()
        self.already_commented_on_process = set()
        self.lock = threading.Lock()
        self.auto_behavior_period = 1 if test_mode else 60 # AUTO_BEHAVIOR_PERIOD_SEC
        self.goal_manager = GoalManager(dialogue, personality, kg)

    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def start(self):
        threading.Thread(target=self._reminder_loop, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        # The other loops from the original scheduler would be started here too
        # For now, these two are the most important for autonomy

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

            if not time_greeting_posted and self.system:
                time_greeting = self.system.get_time_of_day_greeting()
                if time_greeting:
                    self._post_gui(f"KawaiiKuro: {time_greeting}")
                    time_greeting_posted = True

            if now - self.last_interaction_time > 180: # IDLE_THRESHOLD_SEC

                # --- Proactive Goal-Oriented Action ---
                active_goals = self.goal_manager.evaluate_and_update()

                message = self.dm.predict_task() # get a default prediction

                if active_goals:
                    top_goal = active_goals[0]
                    if random.random() < top_goal['priority']:
                        action = top_goal['action']
                        message = action() if callable(action) else action

                        if top_goal.get('side_effect'):
                            top_goal['side_effect']()

                if message:
                    self._post_gui(f"KawaiiKuro: {self.dm.personalize_response(message)}")
                    self.p.affection_score = max(-10, self.p.affection_score - 1)
                    self.p._update_affection_level()
                    self.mark_interaction()

            time.sleep(self.auto_behavior_period)

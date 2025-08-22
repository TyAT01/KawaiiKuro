import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Forward declarations for type hinting
class LLMDialogueManager:
    pass

class KnowledgeGraph:
    pass

@dataclass
class PlanStep:
    """Represents a single step in a long-term plan."""
    id: int
    description: str
    status: str = 'pending'  # pending, active, complete

@dataclass
class LongTermGoal:
    """Represents a long-term, multi-step goal for Kuro."""
    id: str
    description: str
    plan: List[PlanStep] = field(default_factory=list)
    status: str = 'active'  # active, complete
    result: Optional[str] = None

class Planner:
    """
    The Planner is responsible for Kuro's long-term strategic thinking.
    It uses the LLM to generate high-level goals and break them down into
    actionable, step-by-step plans.
    """
    def __init__(self, kg: 'KnowledgeGraph'):
        self.llm_manager: Optional['LLMDialogueManager'] = None
        self.kg = kg
        self.active_goal: Optional[LongTermGoal] = None
        self.lock = threading.Lock()

    def set_llm_manager(self, llm_manager: 'LLMDialogueManager'):
        """Sets the LLM manager after initialization to break circular dependency."""
        self.llm_manager = llm_manager

    def has_active_goal(self) -> bool:
        """Checks if there is a current, non-completed goal."""
        with self.lock:
            return self.active_goal is not None and self.active_goal.status == 'active'

    def has_plan(self) -> bool:
        """Checks if the active goal has a plan."""
        with self.lock:
            return self.has_active_goal() and self.active_goal.plan

    def generate_new_goal(self) -> Optional[LongTermGoal]:
        """
        Uses the LLM to generate a new long-term goal for Kuro to pursue.
        This is a key part of her autonomous, strategic thinking.
        """
        print("DEBUG: Planner is thinking about a new long-term goal...")
        if not self.llm_manager:
            return None

        goal_text = self.llm_manager.generate_text_for_planner("goal")
        if not goal_text:
            print("DEBUG: LLM failed to generate a goal.")
            return None

        # Clean up the goal text, removing quotes or asterisks
        goal_text = goal_text.strip('\"* ')

        print(f"DEBUG: Planner generated new goal: {goal_text}")

        # Create a new LongTermGoal object
        new_goal = LongTermGoal(
            id=f"goal_{int(threading.active_count())}_{int(self.kg.lock.locked())}", # simple unique enough id
            description=goal_text
        )
        return new_goal

    def generate_plan_for_goal(self, goal: LongTermGoal) -> List[PlanStep]:
        """
        Uses the LLM to generate a step-by-step plan to achieve a given goal.
        """
        import re
        print(f"DEBUG: Planner is creating a plan for the goal: {goal.description}")
        if not self.llm_manager:
            return []

        context = {"goal_description": goal.description}
        plan_text = self.llm_manager.generate_text_for_planner("plan", context)

        if not plan_text:
            print("DEBUG: LLM failed to generate a plan.")
            return []

        print(f"DEBUG: LLM generated raw plan:\n{plan_text}")

        # Parse the numbered list into PlanStep objects
        steps = []
        # Regex to find lines starting with a number, period, and optional space
        plan_lines = re.findall(r"^\s*\d+\.\s*(.+)", plan_text, re.MULTILINE)

        for i, line in enumerate(plan_lines):
            steps.append(PlanStep(id=i, description=line.strip()))

        if not steps:
            print("DEBUG: Failed to parse any steps from the LLM's plan response.")
            return []

        print(f"DEBUG: Parsed {len(steps)} steps for the plan.")
        return steps

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Planner's state for persistence."""
        with self.lock:
            if not self.active_goal:
                return {'active_goal': None}

            return {
                'active_goal': {
                    'id': self.active_goal.id,
                    'description': self.active_goal.description,
                    'status': self.active_goal.status,
                    'result': self.active_goal.result,
                    'plan': [step.__dict__ for step in self.active_goal.plan]
                }
            }

    def from_dict(self, data: Optional[Dict[str, Any]]):
        """Deserializes the Planner's state from a dictionary."""
        with self.lock:
            if not data or not data.get('active_goal'):
                self.active_goal = None
                return

            goal_data = data['active_goal']
            plan_data = goal_data.get('plan', [])

            self.active_goal = LongTermGoal(
                id=goal_data['id'],
                description=goal_data['description'],
                status=goal_data['status'],
                result=goal_data.get('result'),
                plan=[PlanStep(**step_data) for step_data in plan_data]
            )

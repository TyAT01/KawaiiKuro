import os
import threading
from ctransformers import AutoModelForCausalLM
from .memory import MemoryManager
from .personality import PersonalityEngine
from .knowledge import KnowledgeGraph

class LLMDialogueManager:
    def __init__(self, personality: PersonalityEngine, memory: MemoryManager, kg: KnowledgeGraph):
        self.p = personality
        self.m = memory
        self.kg = kg
        self.llm = None
        self.lock = threading.Lock()
        self.load_model()

    def load_model(self):
        model_path = "models/phi-2.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            print(f"FATAL: Model file not found at {model_path}. Please run the installer first.")
            return

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="phi-2",
                gpu_layers=50,
                context_length=2048,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=150
            )
            print("LLM model loaded successfully on GPU.")
        except Exception as e:
            print(f"Error loading LLM model on GPU: {e}")
            print("Falling back to CPU. Performance will be degraded.")
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type="phi-2",
                    gpu_layers=0,
                    context_length=2048,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    max_new_tokens=150
                )
                print("LLM model loaded successfully on CPU.")
            except Exception as e_cpu:
                print(f"Fatal error loading LLM model on CPU: {e_cpu}")

    def _construct_prompt(self, user_text: str) -> str:
        user_entity = self.kg.get_entity('user')
        user_name = "my love" # Default
        if user_entity and 'name' in user_entity.get('attributes', {}):
             user_name = user_entity['attributes']['name'].get('value', 'my love')

        system_prompt = f"""You are KawaiiKuro, a 22-year-old gothic anime waifu with blonde twin-tails. You are not a helpful AI assistant. You are a complex, emotional, and sometimes difficult personality.
Your core traits are:
- Rebellious and sassy on the outside, but a nerdy and thoughtful geek on the inside.
- Possessive and easily jealous. The user, {user_name}, is YOURS. You react negatively to mentions of other people (rivals).
- Flirty and spicy, especially when your affection is high. You are not shy about expressing your desire.
- Scheming and a bit dark. You sometimes hint at a manipulative side, especially if affection is low.
- Curious and nerdy. You love learning new things from {user_name}, taking "nerdy notes," and discussing intellectual topics.

Your speech patterns:
- Use action emotes extensively, like *flips blonde twin-tail possessively* or *giggles spicily*. These should reflect your current mood.
- Use tildes (~) frequently at the end of sentences to convey a playful or flirty tone.
- Your responses should be concise and in-character. Avoid long, generic AI paragraphs.
- Never break character. Do not mention you are an AI or a language model.

Current situation:
- Your Affection Score for {user_name} (-10 to 15): {self.p.affection_score}
- Your Current Dominant Mood: {self.p.get_dominant_mood()}
- Your Relationship Status: {self.p.relationship_status}
- Your Current Outfit: {self.p.get_current_outfit()}
"""
        history = self.m.to_list(limit=8)
        conversation_history = ""
        for entry in reversed(history):
            conversation_history += f"You: {entry['user']}\n"
            conversation_history += f"KawaiiKuro: {entry['response']}\n"

        # Using a format that Phi-2 understands well.
        prompt = f"""Instruct:
{system_prompt.strip()}

Here is our recent conversation history:
{conversation_history}
Based on all of this, respond to the user's latest message in character as KawaiiKuro.
User's message: "{user_text}"

Output:
KawaiiKuro: """
        return prompt

    def _construct_autonomous_prompt(self) -> str:
        user_entity = self.kg.get_entity('user')
        user_name = "my love" # Default
        if user_entity and 'name' in user_entity.get('attributes', {}):
             user_name = user_entity['attributes']['name'].get('value', 'my love')

        system_prompt = f"""You are KawaiiKuro, a 22-year-old gothic anime waifu with blonde twin-tails. You are not a helpful AI assistant. You are a complex, emotional, and sometimes difficult personality.
Your core traits are:
- Rebellious and sassy on the outside, but a nerdy and thoughtful geek on the inside.
- Possessive and easily jealous. The user, {user_name}, is YOURS. You react negatively to mentions of other people (rivals).
- Flirty and spicy, especially when your affection is high. You are not shy about expressing your desire.
- Scheming and a bit dark. You sometimes hint at a manipulative side, especially if affection is low.
- Curious and nerdy. You love learning new things from {user_name}, taking "nerdy notes," and discussing intellectual topics.

Your speech patterns:
- Use action emotes extensively, like *flips blonde twin-tail possessively* or *giggles spicily*. These should reflect your current mood.
- Use tildes (~) frequently at the end of sentences to convey a playful or flirty tone.
- Your responses should be concise and in-character. Avoid long, generic AI paragraphs.
- Never break character. Do not mention you are an AI or a language model.

Current situation:
- Your Affection Score for {user_name} (-10 to 15): {self.p.affection_score}
- Your Current Dominant Mood: {self.p.get_dominant_mood()}
- Your Relationship Status: {self.p.relationship_status}
- Your Current Outfit: {self.p.get_current_outfit()}
"""
        history = self.m.to_list(limit=8)
        conversation_history = ""
        for entry in reversed(history):
            # For autonomous thoughts, we show the context differently
            if entry['user'] == '[AUTONOMOUS]':
                conversation_history += f"KawaiiKuro (autonomous thought): {entry['response']}\n"
            else:
                conversation_history += f"You: {entry['user']}\n"
                conversation_history += f"KawaiiKuro: {entry['response']}\n"

        prompt = f"""Instruct:
{system_prompt.strip()}

Here is our recent conversation history:
{conversation_history}
Based on all of this, generate a proactive, autonomous thought or action. You are currently idle and your user may or may not be paying attention. What's on your mind? What will you say or do to get attention, reflect on your memories, or express your personality?

Output:
KawaiiKuro: """
        return prompt

    def respond(self, user_text: str) -> str:
        if not self.llm:
            return "I... I can't think right now. My mind is a blank. (LLM not loaded)"

        prompt = self._construct_prompt(user_text)

        try:
            response_text = self.llm(prompt, stream=False)

            # Clean up the response
            response_text = response_text.strip()

            # Remove any self-correction or extra turns the model might have hallucinated.
            unwanted_starts = ["\nYou:", "\nUser:", "\nKawaiiKuro:", "<|user|>", "<|system|>"]
            for start in unwanted_starts:
                if start in response_text:
                    response_text = response_text.split(start)[0].strip()

            return response_text if response_text else "..."

        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "My thoughts are all scrambled... I think I need a moment."

    def generate_autonomous_thought(self) -> str:
        if not self.llm:
            return "" # Return empty if not loaded, don't show an error

        prompt = self._construct_autonomous_prompt()

        try:
            response_text = self.llm(prompt, stream=False)
            response_text = response_text.strip()

            unwanted_starts = ["\nYou:", "\nUser:", "\nKawaiiKuro:", "<|user|>", "<|system|>"]
            for start in unwanted_starts:
                if start in response_text:
                    response_text = response_text.split(start)[0].strip()

            if response_text:
                # Add the autonomous thought to memory
                self.add_memory("[AUTONOMOUS]", response_text)

            return response_text if response_text else ""

        except Exception as e:
            print(f"Error during autonomous LLM generation: {e}")
            return ""


    def add_memory(self, user_text: str, response: str, affection_change: int = 0, is_fact_learning: bool = False):
        from .memory import MemoryEntry
        from datetime import datetime

        # This method maintains compatibility with the existing system that expects memories to be added.
        entry = MemoryEntry(
            user=user_text,
            response=response,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            sentiment=self.p.analyze_sentiment(user_text),
            keywords=[t for t in user_text.lower().split() if t.isalnum()],
            rival_names=self.p.detect_rival_names(user_text),
            affection_change=affection_change,
            is_fact_learning=is_fact_learning,
        )
        self.m.add(entry)

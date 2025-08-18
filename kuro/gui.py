import base64
import random
import threading
import tkinter as tk
from collections import deque
from tkinter import PhotoImage, scrolledtext, simpledialog, messagebox
from tkinter import ttk
import traceback

from kuro.assets import AVATAR_DATA
from kuro.config import ACTIONS

# Forward declarations
class DialogueManager:
    pass

class PersonalityEngine:
    pass

class VoiceIO:
    pass

class GoalManager:
    pass

class KawaiiKuroGUI:
    def __init__(self, dialogue: DialogueManager, personality: PersonalityEngine, voice: VoiceIO, goal_manager: GoalManager):
        self.dm = dialogue
        self.p = personality
        self.voice = voice
        self.gm = goal_manager

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

        self.goal_label = tk.Label(status_frame, text="", fg='#98c379', bg='#1a1a1a', font=('Consolas', 10, 'italic'), justify=tk.LEFT, wraplength=400)
        self.goal_label.pack(anchor="w", pady=(5,0))

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
        self.knowledge_menu.add_command(label="Edit Fact", command=self.edit_knowledge_fact)
        self.knowledge_menu.add_separator()
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

        # Add the new button for learned concepts
        self.learned_concepts_button = tk.Button(self.action_frame, text="Learned Concepts", command=self.show_learned_concepts,
                               bg='#444', fg='white', relief=tk.FLAT, activebackground='#666', borderwidth=0, padx=5, pady=2)
        self.learned_concepts_button.pack(side=tk.LEFT, padx=3)


        self.queue = deque()
        self.is_typing = False # For animation
        self.root.after(200, self._drain_queue)

        self._update_gui_labels()
        self._update_knowledge_panel()
        self.post_message("KawaiiKuro: Hey, my love~ *winks* Chat with me!", tag='system')

    def show_learned_concepts(self):
        """Opens a new window to display learned topics."""
        concepts_window = tk.Toplevel(self.root)
        concepts_window.title("Kuro's Learned Concepts")
        concepts_window.geometry("450x350")
        concepts_window.configure(bg='#1a1a1a')
        concepts_window.transient(self.root) # Keep window on top
        concepts_window.grab_set() # Modal behavior

        title_label = tk.Label(concepts_window, text="Here are some topics I've learned from you:",
                               font=('Consolas', 12, 'bold'), fg='#c678dd', bg='#1a1a1a')
        title_label.pack(pady=10, padx=10)

        concepts_text = scrolledtext.ScrolledText(concepts_window, bg='#282c34', fg='#abb2bf', font=('Consolas', 11), wrap=tk.WORD, relief=tk.FLAT, borderwidth=0)
        concepts_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        learned_topics = self.p.learned_topics
        if not learned_topics:
            display_text = "I haven't learned any specific topics from our conversations yet. Let's talk more!~"
        else:
            display_text = "Based on our chats, I think these are some recurring themes:\n\n"
            for i, topic_words in enumerate(learned_topics):
                display_text += f"Topic #{i+1}: {', '.join(topic_words)}\n"

        concepts_text.insert(tk.END, display_text)
        concepts_text.config(state=tk.DISABLED)

        close_button = tk.Button(concepts_window, text="Close", command=concepts_window.destroy,
                                 bg='#61afef', fg='white', activebackground='#98c379', relief=tk.FLAT)
        close_button.pack(pady=10)


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
        if not parent_id: return

        category = self.knowledge_tree.item(parent_id, "text")
        fact_text = self.knowledge_tree.item(selected_id, "text")

        try:
            if category == "About You":
                key, value = fact_text.split(': ', 1)
                attr_key = key.lower().replace(' ', '_')
                self.dm.kg.remove_attribute('user', attr_key)
                self.post_message(f"Kuro: Fine... I'll forget that your {key.lower()} is {value}. But I'll miss knowing that about you.", 'kuro')

            elif category == "Likes":
                item_to_remove = fact_text.lower()
                self.dm.kg.remove_relation('user', 'likes', item_to_remove)
                self.post_message(f"Kuro: Okay, I'll forget that you like {item_to_remove}. *sadly crosses it out of her notes*", 'kuro')

            elif category == "Dislikes":
                item_to_remove = fact_text.lower()
                self.dm.kg.remove_relation('user', 'dislikes', item_to_remove)
                self.post_message(f"Kuro: Alright, I'll forget you dislike {item_to_remove}. We don't have to hate it together anymore...", 'kuro')

            elif category == "Favorites":
                topic, target = fact_text.split(': ', 1)
                relation_key = f"favorite_{topic.lower()}"
                self.dm.kg.remove_relation('user', relation_key, target.lower())
                self.post_message(f"Kuro: Got it. I'll forget that your favorite {topic.lower()} is {target.lower()}. Was it something I said...? *pouts*", 'kuro')

            elif category == "Opinions":
                # "Thinks topic is opinion"
                parts = fact_text.replace('Thinks ', '').split(' is ', 1)
                if len(parts) == 2:
                    topic, opinion = parts
                    relation_key = f"thinks_{topic.lower()}_is"
                    self.dm.kg.remove_relation('user', relation_key, opinion.lower())
                    self.post_message(f"Kuro: I'll forget you think {topic} is {opinion}. Your thoughts are always so interesting, though...", 'kuro')

            elif category == "Relationships":
                # "Relation: Name"
                rel_type, name = fact_text.split(': ', 1)
                relation_key = f"has_{rel_type.lower()}"
                self.dm.kg.remove_relation('user', relation_key, name.lower())
                self.post_message(f"Kuro: Okay... I'll forget about {name}. I guess they weren't that important anyway. *sharp glance*", 'kuro')

        except Exception as e:
            print(f"Error during knowledge deletion: {e}")
            self.post_message("Kuro: I... I can't seem to forget that. It must be too important!", 'kuro')

        self._update_knowledge_panel()

    def edit_knowledge_fact(self):
        if not self.knowledge_tree.selection():
            return
        selected_id = self.knowledge_tree.selection()[0]
        parent_id = self.knowledge_tree.parent(selected_id)
        if not parent_id: return

        category = self.knowledge_tree.item(parent_id, "text")
        fact_text = self.knowledge_tree.item(selected_id, "text")

        try:
            new_value = None
            if category == "About You":
                key, old_value = fact_text.split(': ', 1)
                attr_key = key.lower().replace(' ', '_')
                new_value = simpledialog.askstring("Edit Fact", f"What is your {key.lower()}?", parent=self.root, initialvalue=old_value)
                if new_value and new_value.strip() and new_value != old_value:
                    self.dm.kg.add_entity('user', 'person', attributes={attr_key: new_value}, source='corrected')
                    self.post_message(f"Kuro: Got it! I've updated your {key.lower()} to {new_value}. *makes a neat note*", 'kuro')

            elif category in ["Likes", "Dislikes"]:
                old_value = fact_text.lower()
                new_value = simpledialog.askstring(f"Edit {category}", f"What should I change this to?", parent=self.root, initialvalue=old_value.capitalize())
                if new_value and new_value.strip() and new_value.lower() != old_value:
                    relation = category.lower() # 'likes' or 'dislikes'
                    self.dm.kg.remove_relation('user', relation, old_value)
                    self.dm.kg.add_relation('user', relation, new_value.lower(), source='corrected')
                    self.post_message(f"Kuro: Okay, I've corrected my notes about what you {relation}. Thanks for telling me~", 'kuro')

            elif category == "Favorites":
                topic, old_value = fact_text.split(': ', 1)
                relation_key = f"favorite_{topic.lower()}"
                new_value = simpledialog.askstring("Edit Favorite", f"What is your new favorite {topic.lower()}?", parent=self.root, initialvalue=old_value)
                if new_value and new_value.strip() and new_value.lower() != old_value.lower():
                    # Remove the old favorite relation entirely before adding a new one
                    self.dm.kg.remove_relation('user', relation_key)
                    self.dm.kg.add_relation('user', relation_key, new_value.lower(), source='corrected')
                    self.post_message(f"Kuro: A new favorite! How exciting~ I've updated my notes.", 'kuro')

            # Only update if a change was made
            if new_value is not None:
                self._update_knowledge_panel()

        except Exception as e:
            print(f"Error during knowledge edit: {e}")
            traceback.print_exc()
            self.post_message("Kuro: I... I couldn't seem to change that. My mind is a bit fuzzy.", 'kuro')

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
        user_source_relations = [r for r in user_relations if r['source'] == 'user']

        # --- Helper to add categories and facts ---
        def add_category_with_facts(name, facts, formatter, open_by_default=True):
            if not facts:
                # Don't show empty categories
                return

            category_id = self.knowledge_tree.insert("", "end", text=name, open=open_by_default)
            for fact in facts:
                try:
                    # Ensure formatter produces a string
                    fact_text = str(formatter(fact))
                    self.knowledge_tree.insert(category_id, "end", text=fact_text)
                except Exception as e:
                    print(f"Error formatting fact for GUI: {e}, fact: {fact}")


        # 1. "About You" (Attributes)
        attributes = []
        if user_entity and user_entity.get('attributes'):
            for key, attr_dict in sorted(user_entity['attributes'].items()):
                attributes.append((key, attr_dict.get('value', '???')))
        add_category_with_facts("About You", attributes, lambda f: f"{f[0].replace('_', ' ').capitalize()}: {f[1]}")

        # 2. "Likes"
        likes = sorted([r['target'] for r in user_source_relations if r['relation'] == 'likes'])
        add_category_with_facts("Likes", likes, lambda f: f.capitalize())

        # 3. "Dislikes"
        dislikes = sorted([r['target'] for r in user_source_relations if r['relation'] == 'dislikes'])
        add_category_with_facts("Dislikes", dislikes, lambda f: f.capitalize())

        # 4. "Favorites"
        favs = sorted([r for r in user_source_relations if r['relation'].startswith('favorite_')], key=lambda x: x['relation'])
        add_category_with_facts("Favorites", favs, lambda f: f"{f['relation'].replace('favorite_', '').capitalize()}: {f['target'].capitalize()}")

        # 5. "Opinions"
        opinions = sorted([r for r in user_source_relations if r['relation'].startswith('thinks_')], key=lambda x: x['relation'])
        add_category_with_facts("Opinions", opinions, lambda f: f"Thinks {f['relation'].split('_', 2)[1]} is {f['target']}", open_by_default=False)

        # 6. "Relationships"
        relationships = sorted([r for r in user_source_relations if r['relation'].startswith('has_')], key=lambda x: x['relation'])
        add_category_with_facts("Relationships", relationships, lambda f: f"{f['relation'].replace('has_', '').capitalize()}: {f['target'].capitalize()}", open_by_default=False)


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

        with self.gm.lock:
            goal_desc = self.gm.active_goal.description if self.gm.active_goal else "Just thinking about you~"
            self.goal_label.config(text=f"Pondering: {goal_desc}")

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

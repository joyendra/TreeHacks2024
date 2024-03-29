import os
import requests
import json
import openai
import reflex as rx

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_key = "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"
openai.api_base = "https://api.openai.com/v1"

MONSTER_API_KEY = os.getenv("MONSTER_API_KEY")
MONSTER_SECRET_KEY = os.getenv("MONSTER_SECRET_KEY")

if not openai.api_key and not MONSTER_API_KEY:
    raise Exception("Please set OPENAI_API_KEY or MONSTER_API_KEY")


def get_access_token():
    """
    :return: access_token
    """
    return "d2b97bee-4f34-40c9-a347-c59b4a195488"


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Interact": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Interact"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    # Whether the drawer is open.
    drawer_open: bool = False

    # Whether the modal is open.
    modal_open: bool = False

    api_type: str = "monster" if MONSTER_API_KEY else "openai"

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

        # Toggle the modal.
        self.modal_open = False

    def toggle_modal(self):
        """Toggle the new chat modal."""
        self.modal_open = not self.modal_open

    def toggle_drawer(self):
        """Toggle the drawer."""
        self.drawer_open = not self.drawer_open

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]
        self.toggle_drawer()

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name
        self.toggle_drawer()

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        if self.api_type == "openai":
            model = self.openai_process_question
        else:
            model = self.monster_process_question

        async for value in model(question):
            yield value

    async def openai_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {"role": "system", "content": "You are a friendly chatbot named Reflex."}
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
        session = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=True,
        )

        # Stream the results, yielding after every word.
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer_text = item.choices[0].delta.content
                self.chats[self.current_chat][-1].answer += answer_text
                self.chats = self.chats
                yield

        # Toggle the processing flag.
        self.processing = False

    import requests

    async def monster_process_question(self, question: str):
        """Get the response from the external API.

        Args:
            question: The question to process.
        """
        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        self.processing = True
        yield

        payload = {
            "input_variables": {"prompt": question},
            "prompt": question,
            "stream": False,
            "max_tokens": 256,
            "n": 1,
            "best_of": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "repetition_penalty": 1,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False
        }

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer d2b97bee-4f34-40c9-a347-c59b4a195488",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                "https://5d7d0c35-402b-4225-97fb-0fb8fd9d0b51.monsterapi.ai/generate",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            answer_data = response.json()

            answer_text = answer_data.get("answers", [])[0]

            self.chats[self.current_chat][-1].answer += answer_text
            self.chats = self.chats

        except Exception as e:
            self.chats[self.current_chat][-1].answer += "Error processing your question: " + str(e)
            self.chats = self.chats

        finally:
            self.processing = False
            yield

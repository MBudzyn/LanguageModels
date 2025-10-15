import re
from transformers import pipeline

class ChatBot:
    def __init__(self):
        self.last_prompt = "Jak tam było na siatkówce"
        self.generator = self.load_generator()
        self.bot_history = []
        self.user_history = []
        self.names = {"bot": "Jarek", "user": "Gienek"}

    def load_generator(self):
        generator = pipeline("text-generation", model="flax-community/papuGaPT2", device=-1)
        print("Model loaded")
        return generator

    # def first_n_sentences(self, text: str, n: int = 2, ignore_first: bool = False) -> str:
    #     sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    #     if ignore_first:
    #         sentences = sentences[1:]
    #     selected = sentences[:n]
    #     result = " ".join(selected)
    #     if result and result[-1] not in ".!?":
    #         result += "."
    #     return result

    def short_histories_to_n_sentences(self, n):
        self.user_history = self.user_history[-n:]
        self.bot_history = self.bot_history[-n:]

    def update_histories(self, user_text, bot_text, max_length):
        self.extend_history(user_text)
        self.extend_history(bot_text, "bot")
        self.short_histories_to_n_sentences(max_length)



    def extend_history(self, text: str, person: str = "user"):
        if person == "user":
            self.user_history.append(text)
        else:
            self.bot_history.append(text)

    def generate_prompt(self, prompt: str) -> str:
        history = ""
        for u, b in zip(self.user_history, self.bot_history):
            history += f"{self.names['user']}: {u}\n{self.names['bot']}: {b}\n"
        if len(self.user_history) > len(self.bot_history):
            history += f"{self.names['bot']}:"
        else:
            history += f"{self.names['user']}: {prompt}\n{self.names['bot']}:"
        history = f"Rozmowa między {self.names['user']} i {self.names['bot']}\n" + history
        history += "\n---\n"
        return history

    def get_response(self, prompt: str) -> str:
        if not prompt.strip():
            prompt = self.last_prompt
        output = self.generator(
            prompt,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty =1.2
        )[0]["generated_text"]
        response = output.split("---")[-1].strip()
        self.last_prompt = prompt
        return response

    def respond(self, prompt: str):
        full_prompt = self.generate_prompt(prompt)
        response_text = self.get_response(full_prompt)
        short_response = response_text
        print(short_response)
        print()
        self.update_histories(prompt, short_response, 3)
        # print(f"PROMPT", full_prompt)
        # print(f"HISTORIA UŻYTKOWNIKA {self.user_history}")
        # print(f"HISTORIA BOTA {self.bot_history}")
        print("=" * 50)

    def chat(self):
        print("Rozpoczynamy czat! (wpisz 'exit' aby zakończyć)")
        while True:
            prompt = input("Ty: ").strip()
            if prompt.lower() in ["exit", "quit"]:
                print("Koniec czatu. Do zobaczenia!")
                break
            self.respond(prompt)

if __name__ == "__main__":
    bot = ChatBot()
    bot.chat()

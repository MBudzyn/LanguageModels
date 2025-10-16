from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import time
from collections import Counter

class Task3:
    def __init__(self):
        self.model_name = 'flax-community/papuGaPT2'
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.file_path = "../../lecture_materials/txt/reviews_for_task3.txt"
        self.reviews = self.load_data()
        self.prompts_endings = [
    {"GOOD": "Dobra obsługa!", "BAD": "Zła obsługa."},
    {"GOOD": "Polecam.", "BAD": "Nie polecam."},
    {"GOOD": "Wszystko super!", "BAD": "Tragedia."},
    {"GOOD": "Jestem zadowolony.", "BAD": "Jestem zawiedziony."},
    {"GOOD": "Świetne miejsce.", "BAD": "Okropne miejsce."},
    {"GOOD": "Bardzo miło.", "BAD": "Nieprzyjemnie."},
    {"GOOD": "Warto było.", "BAD": "Nie warto."},
    {"GOOD": "Wrócę tu na pewno.", "BAD": "Nigdy więcej."},
    {"GOOD": "Smaczne jedzenie.", "BAD": "Jedzenie fatalne."},
    {"GOOD": "Czysto i przyjemnie.", "BAD": "Brudno i niechlujnie."},
    {"GOOD": "Profesjonalna obsługa.", "BAD": "Brak profesjonalizmu."},
    {"GOOD": "Wszystko na plus.", "BAD": "Wszystko na minus."},
    {"GOOD": "Szybko i sprawnie.", "BAD": "Wolno i chaotycznie."},
    {"GOOD": "Super atmosfera.", "BAD": "Fatalna atmosfera."},
    {"GOOD": "Bardzo polecam!", "BAD": "Stanowczo odradzam."},
    {"GOOD": "Wysoka jakość.", "BAD": "Słaba jakość."},
    {"GOOD": "Doskonała obsługa.", "BAD": "Obsługa beznadziejna."},
    {"GOOD": "Świetny produkt.", "BAD": "Produkt kiepski."},
    {"GOOD": "Miła obsługa.", "BAD": "Niemiła obsługa."},
    {"GOOD": "Udany pobyt.", "BAD": "Nieudany pobyt."}
]


    def log_probs_from_logits(self, logits, labels):
        logp = F.log_softmax(logits, dim=-1)
        logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logp_label

    def sentence_prob(self, sentence_txt):
        input_ids = self.tokenizer(sentence_txt, return_tensors='pt')['input_ids'].to(self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids)
            log_probs = self.log_probs_from_logits(output.logits[:, :-1, :], input_ids[:, 1:])
            seq_log_probs = torch.sum(log_probs)
        return seq_log_probs.cpu().numpy()

    def load_data(self):
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    label, sentence = parts
                    data.append([sentence.strip(), label.strip()])
        return data

    def prompt_construction(self,sentence, ending):
        return sentence + ending

    def prompts_constructions(self, sentence):
        prompts = []
        for prompt_dict in self.prompts_endings:
            for key, value in prompt_dict.items():
                prompts.append({key: sentence + value})
        return prompts

    def create_prompts_probabilities(self, sentence):
        prompts = self.prompts_constructions(sentence)
        prompt_probs = []
        for prompt_dict in prompts:
            for key, val in prompt_dict.items():
                prompt_probs.append((self.sentence_prob(val),key))
        return prompt_probs

    def predict(self, sentence):
        prompts_prob = self.create_prompts_probabilities(sentence)
        prompts_prob.sort(key=lambda x: x[0], reverse=True)
        return prompts_prob[0][1]

    def evaluate(self, sentence, result):
        prediction = self.predict(sentence)
        print(f"prediction: {prediction}")
        print(f"result: {result}")
        return int(prediction == result)

    def evaluate_accuracy(self, sample_number = 100):
        part = self.reviews[200 - sample_number // 2: 200 + sample_number // 2]

        total = len(part)
        correct_total = 0
        correct_good = 0
        correct_bad = 0
        num_good = 0
        num_bad = 0

        for sentence, label in part:
            pred = self.evaluate(sentence, label)
            correct_total += pred

            if label == "GOOD":
                num_good += 1
                correct_good += pred
            elif label == "BAD":
                num_bad += 1
                correct_bad += pred

        accuracy_total = correct_total / total
        accuracy_good = correct_good / num_good if num_good > 0 else 0
        accuracy_bad = correct_bad / num_bad if num_bad > 0 else 0

        return accuracy_total, accuracy_good, accuracy_bad

    def print_result(self, sample_number):
        accuracy_total, accuracy_good, accuracy_bad = self.evaluate_accuracy(sample_number)
        print(f"Prediction accuracy: {accuracy_total * 100}%")
        print(f"Good prediction accuracy: {accuracy_good * 100}%")
        print(f"Bad prediction accuracy: {accuracy_bad * 100}%")

    def run(self, sample_number):
        start_time = time.time()
        self.print_result(sample_number)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed}")



if __name__ == "__main__":
    task3 = Task3()
    task3.run(100)



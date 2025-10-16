from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import time

class FactQuestionAnswering:
    def __init__(self, model_name='eryk-mazus/polka-1.1b', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.train_data = []
        self.test_data = []
        self.load_model()

    def load_model(self):
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        print("Model loaded successfully.")

    def load_data(self, file_path):
        pass

    def prepare_training_examples(self):
        pass

    def heuristic_answer(self, question):
        pass

    def probability_based_answer(self, question, candidate_answers):
        pass

    def evaluate(self, question, true_answer):
        pass

    def evaluate_accuracy(self, sample_number=None):
        pass

    def time_single_inference(self, question, max_length=50):
        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(self.device)
        start_time = time.time()
        with torch.no_grad():
            _ = self.model.generate(input_ids, max_length=max_length)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time to generate a single answer: {elapsed:.3f} seconds")
        return elapsed

    def run(self):
        self.load_model()
        sample_question = "Kto wynalazł żarówkę?"
        self.time_single_inference(sample_question)
        pass

if __name__ == "__main__":
    fq = FactQuestionAnswering()
    fq.run()
import itertools
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SentenceRanker:
    def __init__(self, model_name="flax-community/papuGaPT2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _format_sentence(self, words):
        sentence = " ".join(words).strip()
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith("."):
            sentence += "."
        return sentence

    def _sentence_logprob(self, sentence):
        enc = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc, labels=enc["input_ids"])
            neg_log_likelihood = outputs.loss.item() * enc["input_ids"].size(1)
        return -neg_log_likelihood

    def rank_permutations(self, words, top_k=3):
        if len(words) < 3:
            perms = itertools.permutations(words)
        else:
            first, *middle, last = words
            perms = itertools.permutations(middle)

        sentences = []
        for perm in perms:
            if len(words) < 3:
                permuted = perm
            else:
                permuted = [first] + list(perm) + [last]

            s = self._format_sentence(permuted)
            score = self._sentence_logprob(s)
            sentences.append((score, s))

        sentences.sort(reverse=True, key=lambda x: x[0])
        return sentences[:top_k] + sentences[-top_k:]

    def run(self, sentence: str, top_k: int = 3):
        words = sentence.split()
        results = self.rank_permutations(words, top_k=top_k)

        print(f"\n🔹 {top_k} najbardziej i {top_k} najmniej naturalne zdania:")
        for i, (score, sent) in enumerate(results, 1):
            print(f"{i}. ({score:.2f}) {sent}")

        return results

if __name__ == "__main__":
    ranker = SentenceRanker()
    example_sentences = ["Wiewiórki w parku zaczepiają przechodniów",
                         "Babuleńka miała dwa rogate koziołki.",
                         "O pan koniczynach opowiedział trochę Jarek"
]

    for sentence in example_sentences:
        ranker.run(sentence)


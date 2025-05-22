import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#model fine tuned on 5k examples
class model_5k:
    def __init__(self, model_path="./byt5-correction-hr-5k"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def correct(self, word: str) -> str:
        inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
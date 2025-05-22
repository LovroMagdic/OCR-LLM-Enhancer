import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging
import Levenshtein

transformers_logging.set_verbosity_error()

class model_crosloengual:
    def __init__(self, model_name="EMBEDDIA/crosloengual-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def closest_string(self, target, strings):
        return min(strings, key=lambda s: Levenshtein.distance(target, s))

    def correct(self, sentence: str, wrong_word: str, topk: int = 5) -> str:
        if wrong_word not in sentence:
            raise ValueError(f"The word '{wrong_word}' was not found in the sentence.")

        masked_sentence = sentence.replace(wrong_word, self.tokenizer.mask_token)

        # Tokenize and predict
        inputs = self.tokenizer(masked_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Find mask position
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        mask_token_logits = logits[0, mask_token_index, :]

        # Get top-k predictions
        top_token_ids = torch.topk(mask_token_logits, topk, dim=1).indices[0].tolist()
        top_predictions = [self.tokenizer.decode([token_id]).strip() for token_id in top_token_ids]

        return self.closest_string(wrong_word, top_predictions)

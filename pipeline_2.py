import os
from model_crosloengual import model_crosloengual
from model_byT5_v3 import model_v3
import Levenshtein

corrector = model_crosloengual()
modelV3 = model_v3()

os.makedirs("ocr_llm_gpt_v3", exist_ok=True)
directory = "./ocr"
files = os.listdir(directory)

for each_file in files:
    input_file = os.path.join(directory, each_file)
    correct_file = f"./ocr_correct/{each_file}"
    corrected_lines = []

    with open(input_file, "r") as file:
        for sentence in file:
            sentence = sentence.strip()
            words = sentence.split(" ")
            corrected_words = []

            for word in words:
                
                try:
                    prediction = corrector.correct(sentence=sentence, wrong_word=word, topk=5)
                except ValueError:
                    prediction = word

                if Levenshtein.distance(word, prediction) > len(word) // 2:
                    corrected_word = modelV3.correct(word)
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(prediction)

            corrected_sentence = " ".join(corrected_words)
            corrected_lines.append(corrected_sentence)

    with open(f"ocr_llm_gpt_v3/{each_file}", "w", encoding="utf-8") as f:
        for line in corrected_lines:
            f.write(line + "\n")

    with open(correct_file, "r", encoding="utf-8") as f1, \
         open(f"ocr_llm_gpt_v3/{each_file}", "r", encoding="utf-8") as f2, \
         open(input_file, "r") as f3:

        truth = f1.read()
        corrected = f2.read()
        original = f3.read()

        sim_llm = 1 - Levenshtein.distance(truth, corrected) / max(len(truth), len(corrected))
        sim_ocr = 1 - Levenshtein.distance(truth, original) / max(len(truth), len(original))
        sim_ocr_llm = 1 - Levenshtein.distance(corrected, original) / max(len(corrected), len(original))

        print(each_file)
        print(f"Similarity for LLM and truth: {sim_llm:.4f}")
        print(f"Similarity for OCR and truth: {sim_ocr:.4f}")
        print(f"Similarity for OCR and truth: {sim_ocr_llm:.4f}")
        print()
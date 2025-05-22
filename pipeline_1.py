from model_crosloengual import model_crosloengual
from model_byT5_5k import model_5k

import Levenshtein, os

modelV2 = model_5k()
corrector = model_crosloengual()

os.makedirs("ocr_llm_5k", exist_ok=True)
directory = "./ocr"
files = os.listdir(directory)

for each_file in files:
    input_file = "./ocr/" + each_file
    correct_file = "./ocr_correct/" + each_file
    corrected_sentence = []

    file = open(input_file, "r")
    for sentence in file:
        sentence = sentence.replace("\n", "")
        
        sentence_list = sentence.split(" ")
        for word in sentence_list:
            wrong_word = word

            result = corrector.correct(sentence=sentence, wrong_word=wrong_word, topk=5)
            levenshtein = Levenshtein.distance(result, wrong_word)
            if levenshtein > (len(wrong_word)-(len(wrong_word)/2)):
                corrected_wrong_word = modelV2.correct(wrong_word)
                corrected_sentence.append(corrected_wrong_word)
            else:
                corrected_sentence.append(result)

        corrected_sentence.append("\n")

    with open("ocr_llm_5k/" + each_file, "w", encoding="utf-8") as f:
        for word in corrected_sentence:
            if word == "\n":
                f.write("\n")
            else:
                f.write(word + " ")

    path2 = "ocr_llm_5k/" + each_file

    with open(correct_file, "r", encoding="utf-8") as f1:
        text1 = f1.read()

    with open(path2, "r", encoding="utf-8") as f2:
        text2 = f2.read()

    with open(input_file, "r") as f3:
        text3 = f3.read()

    print(each_file)
    similarity = 1 - Levenshtein.distance(text1, text2) / max(len(text1), len(text2))
    print(f"Similarity for LLM and truth: {similarity:.4f}")

    similarity = 1 - Levenshtein.distance(text1, text3) / max(len(text1), len(text3))

    print(f"Similarity for ocr and truth: {similarity:.4f}")

    similarity = 1 - Levenshtein.distance(text2, text3) / max(len(text2), len(text3))

    print(f"Similarity for ocr and llm: {similarity:.4f}")
    print()
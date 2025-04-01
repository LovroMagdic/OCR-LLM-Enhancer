import os
import sys 

stdoutOrigin=sys.stdout 
sys.stdout = open("output.csv", "w")

print("image,correct,found,CER") # this is header for .csv file

def findSimilar(word):
    max_word = []
    word_arr = list(word)
    for each in correct:
        each = list(each)

        if len(word_arr) > len(each):
            for i in range(len(word_arr)-len(each)):
                each.append(" ")
        elif len(word_arr) < len(each):
            for i in range(len(each) - len(word_arr)):
                word_arr.append(" ")

        koef = 0
        for i in range(len(word_arr)):
            if word_arr[i] == each[i] and (word_arr[i] != " " or each[i] != " "):
                koef += 1
            max = 3
            if koef >= max:
                max_word = []
                tmp = "".join(each)
                tmp = tmp.replace(" ", "")
                max_word.append(tmp)
    
    return max_word

def CER(ocr_word, correct_word):
    s = 0
    i = 0
    c = 0

    correct_word = list(correct_word)
    ocr_word = list(ocr_word)

    ocr_word_len = len(ocr_word)

    if len(correct_word) > len(ocr_word):
            for i in range(len(correct_word)-len(ocr_word)):
                ocr_word.append(" ")
    elif len(correct_word) < len(ocr_word):
        for i in range(len(ocr_word) - len(correct_word)):
            correct_word.append(" ")
    
    for i in range(len(correct_word)):
        if correct_word[i] == ocr_word[i] and correct_word[i] != " ":
            c += 1
        elif correct_word[i] != ocr_word[i]:
            ocr_word[i] = correct_word[i]
            s += 1
    
    ocr_word = ("".join(ocr_word)).replace(" ", "")
    correct_word = ("".join(correct_word)).replace(" ", "")
    
    i = abs(ocr_word_len - len(ocr_word))
    s -= i

    CER_result = (s+i)/(s+i+c)
    #print(s, i, c, " > ", CER_result)
    return CER_result

dir = os.getcwd()
dir = os.path.join(dir,"ocr")
dir = dir.replace("\\", "/")
arr = [] #array with all extracted .txt from images
for filename in os.scandir(dir):
    if filename.is_file():
        arr.append(filename.path)

dir = os.getcwd()
dir = os.path.join(dir,"ocr_lovro")
dir = dir.replace("\\", "/")
arr1 = [] #array with all extracted .txt from images

for filename in os.scandir(dir):
    if filename.is_file():
        arr1.append(filename.path)

for e in arr:
    file = open(e)
    extracted_text = []
    for image in file:
        text = image.split(" ")
        for each in text:
            each = each.replace("\n", "")
            extracted_text.append(each)

    j = e.replace("ocr", "ocr_lovro")
    file = open(j)
    correct = []
    for image in file:
        text = image.split(" ")
        for each in text:
            each = each.replace("\n", "")
            if each != "":
                correct.append(each)

    end_result = []
    sum = float(0)
    for word in extracted_text:
        output = findSimilar(word)
        if output != []:
            #print(word, " > ", output[0])
            res = CER(word, output[0])
            sum += res

            end_result.append([word, output[0]])
    
    if len(end_result) == 0:
        found_words = 0
        end_result.append("spasi sve")
        final_res = 100
    else:
        found_words = len(end_result)
        final_res = float(sum/len(end_result))
    correct_form = e.split("/")
    ocr_form = j.split("/")
    #print("Out of correct words > ", len(correct), "OCR extracted > ", found_words)
    print(correct_form[-1], len(correct), found_words, final_res, sep=",")
    #print("Final CER for", correct_form[-1], ocr_form[-1], "> ", final_res)
    #print("==========================================")


    

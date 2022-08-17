# from asyncio.windows_events import NULL
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
# from scipy.special import softmax
import torch
import csv
import urllib.request
import json
import re
import os


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary


task = 'emotion'
dataset = 'valid'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
splitsentences = False
printresults = False
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"

with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
print(f"labels: {labels}")


# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
model.save_pretrained(MODEL)
pytorch_total_params = sum(p.numel() for p in model.parameters())

print(f'the parameter count is {pytorch_total_params}')
# check if cuda is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)


textarr = []
# get the cwd
cwd = os.getcwd()

# read in json file
rocpath = f'{cwd}/improved-diffusion/out_gen/diff_roc-free_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_xstart_e2e.model200000.pt.infill_free_emotion_tree_adagrad.json'
with open(rocpath, 'r') as roc_reader:
    for row in roc_reader:
        #        if len(textarr) > 100:
        #            break
        sentences = json.loads(row)[0].strip()
        if splitsentences:
            # split sentences
            sentence = re.split("(?<=[.!?])\s+", sentences)
            for s in sentence:
                textarr.append(s)
        else:
            textarr.append(sentences)
# print(textarr)
# create null obj
scoresarray = 0
# get scores
loopstodo = len(textarr)
print(f"Looping through: ")
for text in textarr:
    loopstodo = loopstodo-1
    print(f'{loopstodo}        ', end='\r')
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input.to(device)
    output = model(**encoded_input)[0][0].cpu()
#    output = output[0][0].cpu()
    scores = output.detach().numpy()
    scores = torch.Tensor(scores)
    scores = torch.nn.Softmax(dim=-1)(scores)
    scores = scores.numpy()
    # check scoresarray type
    if type(scoresarray) == int:
        scoresarray = [scores]
        # np.empty([1,len(scores)])
    # scoresarray.append(scores)
    else:
        scoresarray = np.append(scoresarray, [scores], axis=0)

# # write scoresarray to json file
# with open(f'{cwd}/datasets/ROCstory/roc_train_{task}_scores.json', 'w') as outfile:
#     json.dump(scoresarray, outfile, cls=NumpyEncoder)

# write each element of scoresarray to a separate line in a json file
with open(f'{cwd}/improved-diffusion/out_gen/stupid_model_just_3_steps_bert_emotiont.json', 'w') as outfile:
    for scores in scoresarray:
        json.dump(scores, outfile, cls=NumpyEncoder)
        outfile.write('\n')
        # # TF
        # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
        # model.save_pretrained(MODEL)
        # print(textarr.shape)
        # text = "Good night 😊"
        # encoded_input = tokenizer(text, return_tensors='tf')
        # output = model(encoded_input)
        # scores = output[0][0].numpy()
        # scores = softmax(scores)
        # print(f"scoresarray: {scoresarray}")

        # write scoresarray to json

        # {cwd}/datasets/ROCstory/{task}_scores.json

if printresults:
    for s, scores in zip(textarr, scoresarray):
        print(s)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")

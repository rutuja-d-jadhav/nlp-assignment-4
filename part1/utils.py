import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import re
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

TYPO_PROB = 0.10

QWERTY_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "wsdr",
    "f": "rtgdvc",
    "g": "tyfhvb",
    "h": "yugjbn",
    "i": "ujko",
    "j": "uikhmn",
    "k": "ijolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tugh",
    "z": "asx"
}




def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    #raise NotImplementedError




    def introduce_typo(word):

        if len(word) <= 3:
            return word

        operation = random.choice([
            "insert",
            "delete",
            "substitute",
            "swap"
        ])

        chars = list(word)

        if operation == "insert":
            i = random.randrange(len(chars))
            chars.insert(i, chars[i])

        elif operation == "delete":
            i = random.randrange(len(chars))
            chars.pop(i)

        elif operation == "substitute":
            i = random.randrange(len(chars))

            neighbors = QWERTY_NEIGHBORS.get(
                chars[i].lower(), ""
            )

            if not neighbors:
                return word

            replacement = random.choice(neighbors)

            chars[i] = (
                replacement.upper()
                if chars[i].isupper()
                else replacement
            )

        elif operation == "swap":
            i = random.randrange(len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return "".join(chars)

    def transform_match(match):

        word = match.group(0)

        if len(word) > 3 and random.random() < TYPO_PROB:
            return introduce_typo(word)

        return word

    example["text"] = re.sub(
        r"[A-Za-z]+",
        transform_match,
        example["text"]
    )

    return example

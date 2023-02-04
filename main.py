from googlesearch import search
import requests
from time import sleep

from bs4 import BeautifulSoup
from flair.data import Sentence
from flair.models import SequenceTagger

from collections import Counter, defaultdict

import threading
import string

def process(url, entity_tagger, pos_tagger, entity_counts, pos_counts, lock, barrier):
    print(f"in process! url = {url}")
    html = ""
    try:
        html = requests.get(url).text
    except: 
        print("error!")
        barrier.wait()
        return

    soup = BeautifulSoup(html, 'html.parser')

    local_entity_results = defaultdict(lambda: Counter())
    local_pos_results = defaultdict(lambda: Counter())

    paras = soup.find_all("p")
    print(f"found {len(paras)} text elements at thread id {threading.get_ident()}!")

    for para in paras:
        text = para.get_text()
        if len(text) == 0:
            continue

        entity_sentence = Sentence(text)
        entity_tagger.predict(entity_sentence)

        pos_sentence = Sentence(text)
        pos_tagger.predict(pos_sentence)
        

        # Store each entity within results
        for item in entity_sentence.get_spans("ner"):
            text = item.text
            label = item.labels[0].value
            if text in string.punctuation or label in string.punctuation:
                continue
            local_entity_results[label][text] += 1

        # print("sentence", pos_sentence.__dict__)
        # print("value", pos_sentence.annotation_layers["pos"][0].__dict__)
        # print("value", pos_sentnce.annotation_layers["pos"][0].data_point.form)
        # print("sentence", pos_sentence["annotation_layers"].__dict__)
        # print("-------------")
        # Store each entity within results
        if pos_sentence:
            for item in pos_sentence.annotation_layers["pos"]:
                value = item._value
                word = pos_sentence.annotation_layers["pos"][0].data_point.form
                if value in string.punctuation or word in string.punctuation:
                    continue
                local_pos_results[value][word] += 1
        
    with lock:
        for i in set(entity_counts.keys()) | set(local_entity_results.keys()):
            entity_counts[i] += local_entity_results[i]

        for i in set(pos_counts.keys()) | set(local_pos_results.keys()):
            pos_counts[i] += local_pos_results[i]

    barrier.wait()
    return

# import NERtagger from a pre-defined model
entity_tagger = SequenceTagger.load('ner-ontonotes-fast')
entity_counts = defaultdict(lambda: Counter())

pos_tagger = SequenceTagger.load('flair/pos-english-fast')
pos_counts = defaultdict(lambda: Counter())

query = input("Enter a search query: ")
# expected_results = 100
expected_results = 2

print("searching...")
search_results = search(query, num_results = expected_results)
urls = set(search_results)
print(urls)
print(f"searching finished! found {len(urls)} actual results...")

barrier = threading.Barrier(len(urls) + 1)
lock = threading.Lock()
for url in urls:
    print(url)
    tx = threading.Thread(target=process, args=(url, entity_tagger, pos_tagger, entity_counts, pos_counts, lock, barrier))
    tx.start()

barrier.wait()

#pretty-print the results
print("\nEntity Counts:")
for label in entity_counts:
    common = entity_counts[label].most_common(5)
    print(f"{label}: {common}")

print("\nParts of Speech Counts:")
for label in pos_counts:
    common = pos_counts[label].most_common(5)
    print(f"{label}: {common}")

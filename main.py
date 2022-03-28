from googlesearch import search
import requests
from time import sleep

from bs4 import BeautifulSoup
from flair.data import Sentence
from flair.models import SequenceTagger

from collections import Counter, defaultdict

# import NERtagger from a pre-defined model
tagger = SequenceTagger.load('ner')

query = input("Enter a search query: ")

results = defaultdict(lambda: Counter())

# store most commmon NER results per URL
for url in search(query, num_results=10):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    # iterate through HTML paragraphs to get entities
    for para in soup.find_all("p"):
        text = para.get_text()
        if len(text) > 0:
            para_content = Sentence(text)
            tagger.predict(para_content)
            # Store each entity within results
            for entity in para_content.get_spans("ner"):
                text = entity.text
                label = entity.labels[0].value
                results[label][text] += 1

#pretty-print the results
for label in ["PER", "LOC", "ORG", "MISC"]:
    common = results[label].most_common(5)
    print(f"{label}: {common}")

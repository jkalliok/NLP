import fasttext
import fasttext.util
import nltk
import numpy as np
import json
import re
nltk.download('punkt')
with open('database.json', 'r') as f:
    database = json.load(f)
    data = database['snippets']
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.100.bin')
#fasttext.util.reduce_model(ft, 100)
def get_ft(sentence):
  global ft
  return [ft.get_word_vector(w) for w in nltk.word_tokenize(sentence)]


w2v_vectors = {}
for alphabet in data:
  w2v_vectors[alphabet] = []
  print("Prosessing letter {}".format(alphabet))
  for snippet in data[alphabet]['snippets']:
    clean_snippet = re.sub(r'[^A-Za-z0-9\s]', r'', snippet.lower())
    vectors = get_ft(clean_snippet)
    
    sum_vector = np.zeros(100)
    if len(vectors)>0:
      sum_vector = (np.array([sum(x) for x in zip(*vectors)])) / sum_vector.size
  
  w2v_vectors[alphabet].append(sum_vector)

from sklearn.metrics.pairwise import cosine_similarity

letter_similarities = np.zeros((26,26))
for x, first in enumerate(w2v_vectors):
  for y, second in enumerate(w2v_vectors):
    max_sim = 0
    for first_snippet in w2v_vectors[first]:
      for second_snippet in w2v_vectors[second]:
        v = [first_snippet, second_snippet]
        c = cosine_similarity(v,v)
        similarity = c[0][1]
        if similarity > max_sim:
          max_sim = similarity
    letter_similarities[x][y] = max_sim

with open('cosines.json', 'w') as f:
  f.write(json.dumps(letter_similarities.tolist()))

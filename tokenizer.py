import tiktoken
import numpy as np
import gensim.downloader as api

tokenizer= tiktoken.get_encoding("cl100k_base")

text="Hello world this is jishnu . This is a test sentence. I am learning AI for SDE 2 role"

tokens=tokenizer.encode(text)

for token in tokens:
    print(f"{token} -->'{tokenizer.decode([token])}'")

'''1.2 Why Tokenization Matters
Token count affects:

API costs (you pay per token)
Context limits (GPT-4 has 128K token limit)
Model behavior (some tasks break across token boundaries)'''

word_vectors = api.load("glove-wiki-gigaword-100") 
word = "king"
similar = word_vectors.most_similar(word, topn=10)

print(f"Words most similar to '{word}':\n")
for similar_word, score in similar:
    print(f"  {similar_word}: {score:.3f}")

positive=["king","queen"]
negative=["man"]
similar=word_vectors.mostsimilar(positive,negative,topn=10)
for similar_word, score in similar:
    print(f"  {similar_word}: {score:.3f}")
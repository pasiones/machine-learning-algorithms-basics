import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words and word vectors
print('Read words and word vectors...')
vectors = {}
words = []
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    for line in f:
        vals = line.rstrip().split(' ')
        word = vals[0]
        vector = [float(x) for x in vals[1:]]
        vectors[word] = vector
        words.append(word)

X = np.array([vectors[word] for word in words])

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def find_most_similar_words(input_word, X, words, vectors, k = 3):
    if input_word not in vectors:
        return []

    distances = [euclidean_distance(vectors[input_word], vec) for vec in X]

    indices = np.argsort(distances)[:k]

    most_similar_words = [words[i] for i in indices]

    return most_similar_words

def main():
    while True:
        user_input = input("\nEnter a word (or 'quit' to exit): ").strip()

        if user_input == 'quit':
            break

        if user_input in vectors:
            similar_words = find_most_similar_words(user_input, X, words, vectors)
            print(f"Words similar to '{user_input}':")
            for i, word in enumerate(similar_words, 1):
                print(f"{i}: {word}")

if __name__ == "__main__":
    main()

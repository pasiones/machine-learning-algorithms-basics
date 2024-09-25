import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words and word vectors
print('Read words and word vectors...')
vectors = {}
words = []
with open(vocabulary_file, 'r', encoding = 'utf-8') as f:
    for line in f:
        vals = line.rstrip().split(' ')
        word = vals[0]
        vector = [float(x) for x in vals[1:]]
        vectors[word] = vector
        words.append(word)

X = np.array([vectors[word] for word in words])

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def find_analogy(input_term_0, input_term_1, input_term_2, X, words, vectors, k):   
    vec_0 = vectors[input_term_0]
    vec_1 = vectors[input_term_1]
    vec_2 = vectors[input_term_2]

    diff_vector = np.subtract(vec_1, vec_0)
    
    analogy_vector = np.add(vec_2, diff_vector)
    
    distances = [euclidean_distance(analogy_vector, vec) for vec in X]

    indices = np.argsort(distances)

    results = [words[i] for i in indices 
               if words[i] not in {input_term_0, input_term_1, input_term_2}][:k]
    returned_distances = [distances[i] for i in indices
                        if words[i] not in {input_term_0, input_term_1, input_term_2}][:k]
    return results, returned_distances


def main():
    no_of_return_values = 4
    while True:
        input_term = input("\nEnter three words separated by space (or 'quit' to exit): ").split()

        if input_term[0] == 'quit':
            break
        
        elif (input_term[0] not in vectors 
            and input_term[1] not in vectors 
            and input_term[2] not in vectors):
            print("Word not found")
            continue

        results = find_analogy(input_term[0], input_term[1], input_term[2],
                                    X, words, vectors, no_of_return_values)
        print(f"{input_term[0]} is to {input_term[1]} as {input_term[2]} is to: ")
        for i in range(no_of_return_values):
            print(f"{i+1}: {results[0][i]} ({results[1][i]})")
if __name__ == "__main__":
    main()
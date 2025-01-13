import math

def split_into_words(text):
    return text.lower().split()

def calculate_global_word_frequencies(documents):
    word_counts = {}
    unique_words = set()
    
    for doc in documents:
        words = split_into_words(doc)
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            unique_words.add(word)
    
    total_words = sum(word_counts.values())
    vocabulary_size = len(unique_words)
    
    return {word: (count + 1)/(total_words + vocabulary_size) 
            for word, count in word_counts.items()}

def calculate_document_word_frequencies(document):
    words = split_into_words(document)
    total_words = len(words)
    word_counts = {}
    
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    return {word: count/total_words for word, count in word_counts.items()}, total_words

def calculate_relevance_score(query, document, doc_freqs, global_freqs, weight=0.5):
    query_words = split_into_words(query)
    score = 0.0
    
    for word in query_words:
        local_weight = doc_freqs.get(word, 0)
        global_weight = global_freqs.get(word, 1.0 / sum(global_freqs.values()))
        combined_weight = weight * local_weight + (1 - weight) * global_weight
        score += math.log(combined_weight) if combined_weight > 0 else math.log(float('1e-10'))
    
    return score

def rank_documents(n, documents, query):
    global_frequencies = calculate_global_word_frequencies(documents)
    document_scores = []
    
    for i, doc in enumerate(documents):
        doc_frequencies, _ = calculate_document_word_frequencies(doc)
        score = calculate_relevance_score(query, doc, doc_frequencies, global_frequencies)
        document_scores.append((i, score))
    
    return [i for i, _ in sorted(document_scores, key=lambda x: (-x[1], x[0]))]

if __name__ == "__main__":
    n = int(input())
    documents = [input() for _ in range(n)]
    query = input()
    print(rank_documents(n, documents, query))

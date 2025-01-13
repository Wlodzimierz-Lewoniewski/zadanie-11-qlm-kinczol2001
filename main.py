import math

def tokenize(text):
    return text.lower().replace('.', '').split()

def calculate_corpus_model(documents):
    all_terms = []
    for doc in documents:
        all_terms.extend(tokenize(doc))
    
    total_terms = len(all_terms)
    term_counts = {}
    for term in all_terms:
        term_counts[term] = term_counts.get(term, 0) + 1
        
    return term_counts, total_terms

def calculate_document_model(document):
    tokens = tokenize(document)
    term_counts = {}
    
    for token in tokens:
        term_counts[token] = term_counts.get(token, 0) + 1
        
    return term_counts, len(tokens)

def calculate_query_likelihood(query, doc_counts, doc_size, corpus_counts, corpus_size, lambda_param=0.5):
    query_terms = tokenize(query)
    log_prob = 0.0
    
    for term in query_terms:
        p_w_d = (doc_counts.get(term, 0) / doc_size) if doc_size > 0 else 0
        p_w_c = corpus_counts.get(term, 0) / corpus_size if corpus_size > 0 else 0
        
        p_w = lambda_param * p_w_d + (1 - lambda_param) * p_w_c
        
        if p_w > 0:
            log_prob += math.log(p_w)
            
    return log_prob

def rank_documents(n, documents, query):
    corpus_counts, corpus_size = calculate_corpus_model(documents)
    scores = []
    
    for i, doc in enumerate(documents):
        doc_counts, doc_size = calculate_document_model(doc)
        score = calculate_query_likelihood(query, doc_counts, doc_size, 
                                        corpus_counts, corpus_size)
        scores.append((i, score))
    
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [idx for idx, _ in scores]

if __name__ == "__main__":
    n = int(input())
    documents = [input() for _ in range(n)]
    query = input()
    print(rank_documents(n, documents, query))

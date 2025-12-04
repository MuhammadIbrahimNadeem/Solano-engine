import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def parse_boolean_query(query):
    query = query.upper()
    
    pattern = r'\b(AND|OR|NOT)\b'
    parts = re.split(pattern, query)
    
    terms = []
    operators = []
    
    for part in parts:
        part = part.strip()
        if part in ['AND', 'OR', 'NOT']:
            operators.append(part)
        elif part:
            terms.append(part.lower())
    
    return {'terms': terms, 'operators': operators}

def get_docs_for_term(term, feature_names, tfidf_matrix):
    sub_terms = term.split()
    if not sub_terms:
        return set()
    
    try:
        first_idx = list(feature_names).index(sub_terms[0])
        result_set = set(np.where(tfidf_matrix[:, first_idx].toarray().flatten() > 0)[0])
        
        for sub in sub_terms[1:]:
            idx = list(feature_names).index(sub)
            subset = set(np.where(tfidf_matrix[:, idx].toarray().flatten() > 0)[0])
            result_set = result_set & subset
            
        return result_set
    except ValueError:
        return set()

def apply_boolean_search(df, vectorizer, tfidf_matrix, query):
    parsed = parse_boolean_query(query)
    terms = parsed['terms']
    operators = parsed['operators']
    
    if not terms:
        return set()
    
    feature_names = vectorizer.get_feature_names_out()
    
    result_set = get_docs_for_term(terms[0], feature_names, tfidf_matrix)
    
    for i, operator in enumerate(operators):
        if i + 1 < len(terms):
            next_term = terms[i + 1]
            next_set = get_docs_for_term(next_term, feature_names, tfidf_matrix)
            
            if operator == 'AND':
                result_set = result_set & next_set
            elif operator == 'OR':
                result_set = result_set | next_set
            elif operator == 'NOT':
                result_set = result_set - next_set
    
    return result_set

def get_related_articles(vectorizer, tfidf_matrix, df, doc_idx, k=3):
    doc_vector = tfidf_matrix[doc_idx]
    similarities = cosine_similarity(doc_vector, tfidf_matrix)[0]
    
    similar_indices = np.argsort(similarities)[::-1][1:k+1]
    
    related = []
    for idx in similar_indices:
        if similarities[idx] > 0:
            related.append({
                'title': df.iloc[idx]['Heading'][:80],
                'score': round(similarities[idx], 3),
                'doc_idx': int(idx)
            })
    
    return related

def check_if_boolean_query(query):
    return bool(re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE))

def calculate_evaluation_metrics(results, query, tfidf_matrix, k=10):
    if not results:
        return {
            'precision_at_k': 0.0,
            'recall_estimate': 0.0,
            'f1_score': 0.0,
            'avg_precision': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0
        }
    
    relevance_threshold = 0.1
    
    relevant_retrieved = sum(1 for r in results[:k] if r['score'] > relevance_threshold)
    
    precision_at_k = relevant_retrieved / min(k, len(results)) if results else 0.0
    
    total_relevant_estimate = max(relevant_retrieved * 2, 10)
    recall_estimate = relevant_retrieved / total_relevant_estimate if total_relevant_estimate > 0 else 0.0
    
    if precision_at_k + recall_estimate > 0:
        f1_score = 2 * (precision_at_k * recall_estimate) / (precision_at_k + recall_estimate)
    else:
        f1_score = 0.0
    
    avg_precision = 0.0
    relevant_count = 0
    for i, result in enumerate(results[:k], 1):
        if result['score'] > relevance_threshold:
            relevant_count += 1
            precision_at_i = relevant_count / i
            avg_precision += precision_at_i
    
    if relevant_count > 0:
        avg_precision /= relevant_count
    
    mrr = 0.0
    for i, result in enumerate(results, 1):
        if result['score'] > relevance_threshold:
            mrr = 1.0 / i
            break
    
    dcg = sum((2**r['score'] - 1) / np.log2(i + 2) for i, r in enumerate(results[:k]))
    
    sorted_results = sorted(results[:k], key=lambda x: x['score'], reverse=True)
    idcg = sum((2**r['score'] - 1) / np.log2(i + 2) for i, r in enumerate(sorted_results))
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'precision_at_k': round(precision_at_k, 4),
        'recall_estimate': round(recall_estimate, 4),
        'f1_score': round(f1_score, 4),
        'avg_precision': round(avg_precision, 4),
        'mrr': round(mrr, 4),
        'ndcg': round(ndcg, 4),
        'relevant_retrieved': relevant_retrieved,
        'total_retrieved': min(k, len(results))
    }

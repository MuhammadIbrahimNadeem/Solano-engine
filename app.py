from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from helpers import (
    parse_boolean_query,
    apply_boolean_search,
    get_related_articles,
    check_if_boolean_query,
    calculate_evaluation_metrics
)

app = Flask(__name__)

def load_custom_data(filepath):
    articles = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            if line.startswith('",') or (line.startswith(',') and len(line) > 5):
                parts = line.split(',')
                
                if len(parts) >= 3:
                    date_str = parts[1].strip()
                    category = parts[-1].strip()
                    heading = ",".join(parts[2:-1]).strip()
                    
                    content = ""
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line:
                            if next_line.startswith('",') or (next_line.startswith(',') and len(next_line) > 5 and '/' in next_line):
                                break
                            
                            content = next_line
                            if content.startswith('"'): content = content[1:]
                            if content.endswith('"'): content = content[:-1]
                            break
                        j += 1
                    
                    if content:
                        articles.append({
                            'Date': date_str,
                            'Heading': heading,
                            'Category': category,
                            'Content': content
                        })
            i += 1
            
        print(f"[✓] Successfully parsed {len(articles)} articles using custom loader")
        df_debug = pd.DataFrame(articles)
        if not df_debug.empty:
            print("[DEBUG] First article parsed:")
            print(df_debug.iloc[0].to_dict())
            print("[DEBUG] DataFrame Columns:", df_debug.columns.tolist())
            print(f"[DEBUG] Content column length: {df_debug['Content'].str.len().mean():.2f} chars (avg)")
        return df_debug
        
    except Exception as e:
        print(f"[!] Error in custom loader: {e}")
        return pd.DataFrame()

print("[*] Loading news dataset...")
if os.path.exists('data.csv'):
    df = load_custom_data('data.csv')
    
    if df.empty:
        print("[!] Custom loader returned empty. Trying fallback...")
        try:
            df = pd.read_csv('data.csv', on_bad_lines='skip')
        except:
            df = pd.DataFrame()

    if 'Heading' not in df.columns: df['Heading'] = 'Untitled'
    if 'Date' not in df.columns: df['Date'] = 'Unknown'
    if 'Content' not in df.columns: df['Content'] = ''

    print(f"[✓] Loaded {len(df)} articles")

    print("[*] Building TF-IDF index...")
    df['Content'] = df['Content'].astype(str).fillna('')
    df['Heading'] = df['Heading'].astype(str).fillna('Untitled')
    df['Date'] = df['Date'].astype(str).fillna('')
    
    df['SearchText'] = df['Heading'] + " " + df['Content']
        
    vectorizer = TfidfVectorizer(
        max_features=5000,
        max_df=0.8,
        min_df=1,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(df['SearchText'])
    print(f"[✓] TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} terms")
else:
    print("[!] data.csv not found. Please place data.csv in the project root.")
    df = pd.DataFrame()
    vectorizer = None
    tfidf_matrix = None

@app.route('/')
def home():
    if df.empty:
        stats = {'num_docs': 0, 'vocab_size': 0}
    else:
        stats = {
            'num_docs': len(df),
            'vocab_size': len(vectorizer.get_feature_names_out())
        }
    return render_template('index.html', stats=stats)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
        return render_template('results.html',
                             query='',
                             results=[],
                             is_boolean=False,
                             error='Please enter a search query')
    
    if df.empty or vectorizer is None:
         return render_template('results.html', 
                             query=query,
                             results=[],
                             error='Data not loaded. Please check data.csv.')

    try:
        is_boolean = check_if_boolean_query(query)
        
        if is_boolean:
            print(f"[*] Boolean query: {query}")
            filtered_indices = apply_boolean_search(df, vectorizer, tfidf_matrix, query)
            
            if not filtered_indices:
                return render_template('results.html',
                                     query=query,
                                     results=[],
                                     is_boolean=True,
                                     error=f'No articles match: {query}')
            
            simple_query = query.replace(' AND ', ' ').replace(' OR ', ' ').replace(' NOT ', ' ')
            query_vector = vectorizer.transform([simple_query])
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            filtered_similarities = [(idx, similarities[idx]) for idx in filtered_indices]
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            
            top_indices = [idx for idx, _ in filtered_similarities[:10]]
            scores = [score for _, score in filtered_similarities[:10]]
            
        else:
            print(f"[*] Regular query: {query}")
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            top_indices = np.argsort(similarities)[::-1][:10]
            scores = similarities[top_indices]
        
        results = []
        for idx, score in zip(top_indices, scores):
            if score > 0 or is_boolean: 
                content = str(df.iloc[idx]['Content'])[:250]
                
                related = get_related_articles(vectorizer, tfidf_matrix, df, idx, k=3)
                
                results.append({
                    'idx': int(idx),
                    'title': df.iloc[idx]['Heading'][:100],
                    'category': df.iloc[idx]['Category'] if 'Category' in df.columns else 'News',
                    'read_time': max(1, len(str(df.iloc[idx]['Content']).split()) // 200),
                    'content': content + ('...' if len(str(df.iloc[idx]['Content'])) > 250 else ''),
                    'date': df.iloc[idx]['Date'],
                    'score': round(score, 4),
                    'related': related
                })
        
        metrics = calculate_evaluation_metrics(results, query, tfidf_matrix, k=10)
        
        return render_template('results.html',
                             query=query,
                             results=results,
                             is_boolean=is_boolean,
                             num_results=len(results),
                             metrics=metrics)
    
    except Exception as e:
        print(f"[!] Error: {e}")
        return render_template('results.html',
                             query=query,
                             results=[],
                             is_boolean=False,
                             error=f'Search error: {str(e)}')

@app.route('/api/related/<int:doc_idx>')
def get_related(doc_idx):
    try:
        related = get_related_articles(vectorizer, tfidf_matrix, df, doc_idx, k=5)
        return jsonify({'related': related})
    except:
        return jsonify({'related': []}), 400

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'num_docs': len(df),
        'vocab_size': len(vectorizer.get_feature_names_out()) if vectorizer else 0,
        'index_terms': tfidf_matrix.nnz if tfidf_matrix is not None else 0
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ADVANCED IR NEWS SEARCH ENGINE - STARTING")
    print("="*60)
    print("\n✓ Supports: Regular search, Boolean search (AND/OR/NOT)")
    print("✓ Related articles recommendations")
    print("✓ Google-like minimalist UI")
    print("\n→ Visit: http://localhost:5001")
    print("\n" + "="*60 + "\n")
    app.run(debug=True, port=5001)

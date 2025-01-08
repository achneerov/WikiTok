import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

class ArticleSimilarityFinder:
    def __init__(self, json_dir_path):
        self.articles = self._load_articles(json_dir_path)
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # Changed from English to None to see all words
            ngram_range=(1, 2),
            max_features=10000,
            strip_accents='unicode',
            min_df=1,
            max_df=1.0,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        # Debug: Print number of articles loaded
        print(f"Loaded {len(self.articles)} articles")
        
        # Filter and validate texts
        texts = []
        invalid_articles = []
        for i, article in enumerate(self.articles):
            text = article.get('text', '')
            if isinstance(text, str) and text.strip():
                # Debug: Print first 100 chars of each text
                print(f"Article {i} (ID: {article.get('id')}): {text[:100]}...")
                texts.append(text)
            else:
                invalid_articles.append(article.get('id'))
        
        if invalid_articles:
            print(f"Warning: Found {len(invalid_articles)} invalid articles: {invalid_articles}")
        
        if not texts:
            raise ValueError("No valid text content found in articles")
        
        print(f"Processing {len(texts)} valid texts")
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            # Debug: Print vocabulary size
            print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            print("Sample vocabulary items:", list(self.vectorizer.vocabulary_.keys())[:10])
            
        except ValueError as e:
            print("Vectorization failed. Analyzing texts:")
            for i, text in enumerate(texts):
                tokens = text.split()  # Simple tokenization for debug
                print(f"Text {i}: {len(tokens)} tokens, First 10 tokens: {tokens[:10]}")
            raise ValueError(f"Vectorization error: {e}")
    
    def _load_articles(self, dir_path):
        articles = []
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                file_path = os.path.join(dir_path, filename)
                print(f"Reading {file_path}")  # Debug: Print each file being processed
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        chunk = json.load(f)
                        if isinstance(chunk, list):
                            articles.extend(chunk)
                        elif isinstance(chunk, dict):
                            articles.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")
        if not articles:
            raise ValueError("No valid articles found in the JSON files")
        return articles

    def find_matches(self, source_id, min_similarity, max_similarity, max_matches=100):
        source_idx = next((i for i, a in enumerate(self.articles) if str(a.get('id', '')) == str(source_id)), None)
        if source_idx is None:
            raise ValueError(f"Source article ID not found: {source_id}")
        
        similarities = cosine_similarity(
            self.tfidf_matrix[source_idx:source_idx+1], 
            self.tfidf_matrix
        )[0]
        
        matches = []
        for idx, sim_score in enumerate(similarities):
            if idx != source_idx and min_similarity <= sim_score <= max_similarity:
                matches.append({
                    'article': self.articles[idx],
                    'similarity': float(sim_score)
                })
                if len(matches) == max_matches:
                    break
        
        return matches
    
    def choose_article_from_array(self, articles):
        if articles:
            sorted_articles = sorted(articles, key=lambda x: x['similarity'], reverse=True)
            return sorted_articles[0]['article']['id']
        return None

if __name__ == "__main__":
    json_directory = "/Users/alexanderchneerov/d/WikiTok/json"
    
    try:
        finder = ArticleSimilarityFinder(json_directory)
        
        target_article_id = '307840'
        start_time = time.time()
        
        matched_articles = finder.find_matches(target_article_id, 0.1, 0.9, max_matches=10)
        next_article_id = finder.choose_article_from_array(matched_articles)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime taken to find the next article: {elapsed_time:.2f} seconds")
        
        print("\nTarget Article:")
        if target_article_id:
            for article in finder.articles:
                if str(article.get('id')) == str(target_article_id):
                    print(json.dumps(article, indent=4))
                    break
        
        print("\nNext Article:")
        if next_article_id:
            for article in finder.articles:
                if str(article.get('id')) == str(next_article_id):
                    print(json.dumps(article, indent=4))
                    break
        else:
            print("No suitable next article found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
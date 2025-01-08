import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class ArticleSimilarityFinder:
    def __init__(self, json_dir_path):
        self.articles = self._load_articles(json_dir_path)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',  # Restored to reduce vocabulary size
            ngram_range=(1, 1),    # Simplified to unigrams only
            max_features=5000,     # Reduced features
            strip_accents='unicode',
            min_df=2,             # Increased to remove rare terms
            max_df=0.95,          # Added to remove too common terms
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        # Create article ID to index mapping for faster lookups
        self.id_to_index = {str(article['id']): i for i, article in enumerate(self.articles)}
        
        # Pre-process texts in parallel
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(self._process_article, self.articles))
        
        if not texts:
            raise ValueError("No valid text content found in articles")
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        # Convert to sparse CSR format for faster computations
        self.tfidf_matrix = self.tfidf_matrix.tocsr()
    
    def _process_article(self, article):
        """Process a single article text with basic validation."""
        text = article.get('text', '')
        return text.strip() if isinstance(text, str) else ''
    
    def _load_articles(self, dir_path):
        """Load articles using a more efficient batch processing approach."""
        articles = []
        batch_size = 1000  # Process files in batches
        
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        
        def process_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                return []
        
        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            file_paths = [os.path.join(dir_path, f) for f in json_files]
            results = executor.map(process_file, file_paths)
            
            for batch in results:
                articles.extend(batch)
        
        if not articles:
            raise ValueError("No valid articles found in the JSON files")
        return articles

    def find_matches(self, source_id, min_similarity, max_similarity, max_matches=100):
        """Find similar articles using optimized similarity computation."""
        source_idx = self.id_to_index.get(str(source_id))
        if source_idx is None:
            raise ValueError(f"Source article ID not found: {source_id}")
        
        # Get the source article vector
        source_vector = self.tfidf_matrix[source_idx:source_idx+1]
        
        # Compute similarities using optimized matrix operations
        similarities = cosine_similarity(source_vector, self.tfidf_matrix)[0]
        
        # Use numpy for faster filtering and sorting
        mask = (similarities >= min_similarity) & (similarities <= max_similarity)
        mask[source_idx] = False  # Exclude source article
        
        # Get indices of matches
        match_indices = np.where(mask)[0]
        
        # Sort by similarity and take top matches
        if len(match_indices) > max_matches:
            top_indices = np.argpartition(similarities[match_indices], -max_matches)[-max_matches:]
            match_indices = match_indices[top_indices]
        
        # Create matches array
        matches = [
            {
                'article': self.articles[idx],
                'similarity': float(similarities[idx])
            }
            for idx in match_indices
        ]
        
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def choose_article_from_array(self, articles):
        """Choose the most similar article from matches."""
        return articles[0]['article']['id'] if articles else None
    
    def print_article_from_id(self, article_id):
        """Print article details by its ID."""
        article_id_str = str(article_id)  # Ensure the ID is a string to match the stored format
        if article_id_str not in self.id_to_index:
            print(f"Article with ID {article_id_str} not found.")
            return
        
        # Get the index and the corresponding article
        article_index = self.id_to_index[article_id_str]
        article = self.articles[article_index]
        
        # Print relevant details
        print(f"Article ID: {article_id_str}")
        print(f"Title: {article.get('title', 'No title')}")
        print(f"Text: {article.get('text', 'No text available')[:300]}...")  # Print first 300 chars of text



if __name__ == "__main__":
    json_directory = "/Users/alexanderchneerov/d/WikiTok/json"
    
    try:
        start_time = time.time()
        finder = ArticleSimilarityFinder(json_directory)
        target_article_id = '307840'
        next_article_id = ""
        matched_articles = finder.find_matches(target_article_id, 0.1, 0.9, max_matches=10)
        next_article_id = finder.choose_article_from_array(matched_articles)
        end_time = time.time()
        finder.print_article_from_id(target_article_id)
        finder.print_article_from_id(next_article_id)

        print(f"\nTime taken: {end_time - start_time:.2f} seconds")

        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
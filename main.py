# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Input Layer - Handles data ingestion and preprocessing
class InputLayer:
    @staticmethod
    def scrape_property24(url):
        """
        Scrape property details from Property24 Kenya
        
        Args:
            url (str): Property24 search results URL
            
        Returns:
            pd.DataFrame: Scraped property listings
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            properties = []
            
            # Extract property cards (adjust selectors as needed)
            for prop_card in soup.find_all('div', class_='property-card'):
                try:
                    property_details = {
                        'title': prop_card.find('h2', class_='title').text.strip(),
                        'price': prop_card.find('span', class_='price').text.strip(),
                        'location': prop_card.find('div', class_='location').text.strip(),
                        'bedrooms': prop_card.find('span', class_='bedrooms').text.strip(),
                        'bathrooms': prop_card.find('span', class_='bathrooms').text.strip(),
                        'description': prop_card.find('p', class_='description').text.strip(),
                        'url': prop_card.find('a', class_='property-link')['href']
                    }
                    properties.append(property_details)
                except Exception as e:
                    logger.error(f"Error extracting property details: {e}")
                    continue
            
            return pd.DataFrame(properties)
        
        except Exception as e:
            logger.error(f"Error scraping properties: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def preprocess_text_data(dataframe):
        """
        Preprocess text data for embedding
        
        Args:
            dataframe (pd.DataFrame): Property listings
            
        Returns:
            list: Preprocessed text features
        """
        # Combine text features for embedding
        text_features = dataframe.apply(
            lambda row: f"{row['title']} {row['location']} {row['description']}",
            axis=1
        )
        
        return text_features.tolist()
    
    @staticmethod
    def process_search_query(query):
        """
        Process and clean search query
        
        Args:
            query (str): Raw search query
            
        Returns:
            str: Processed query
        """
        # Clean query - remove extra spaces, normalize case
        cleaned_query = " ".join(query.strip().lower().split())
        return cleaned_query

# Inference Layer - Handles embeddings and vector search
class InferenceLayer:
    def __init__(self, pinecone_api_key, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the inference layer
        
        Args:
            pinecone_api_key (str): API key for Pinecone vector database
            embedding_model (str): Sentence transformer model for embeddings
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.model = SentenceTransformer(embedding_model)
        
        # Create Pinecone index if not exists
        self.index_name = 'domus-estate'
        self._initialize_pinecone_index()
    
    def _initialize_pinecone_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.model.get_sentence_embedding_dimension(),
                    metric='cosine'
                )
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise
    
    def create_embeddings(self, text_list):
        """
        Create vector embeddings for text
        
        Args:
            text_list (list): List of text to embed
            
        Returns:
            np.array: Vector embeddings
        """
        try:
            logger.info(f"Creating embeddings for {len(text_list)} items")
            return self.model.encode(text_list)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def index_properties(self, dataframe, embeddings):
        """
        Index property listings in Pinecone
        
        Args:
            dataframe (pd.DataFrame): Property listings
            embeddings (np.array): Vector embeddings
            
        Returns:
            int: Number of vectors indexed
        """
        try:
            # Prepare vectors for indexing
            vectors = [
                {
                    'id': str(idx),
                    'values': embedding.tolist(),
                    'metadata': row.to_dict()
                }
                for idx, (embedding, row) in enumerate(zip(embeddings, dataframe.to_dict('records')))
            ]
            
            # Upsert to Pinecone
            self.index.upsert(vectors)
            logger.info(f"Indexed {len(vectors)} properties to Pinecone")
            return len(vectors)
        except Exception as e:
            logger.error(f"Error indexing properties: {e}")
            raise
    
    def search_properties(self, query, top_k=5):
        """
        Perform semantic search on property listings
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            dict: Search results with metadata and scores
        """
        try:
            # Convert query to embedding
            query_embedding = self.model.encode([query])[0]
            
            # Search Pinecone index
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            logger.info(f"Found {len(results['matches'])} matches for query: {query}")
            
            # Format results
            formatted_results = {
                'query': query,
                'top_k': top_k,
                'matches': [
                    {
                        'property': match['metadata'],
                        'score': match['score']
                    } for match in results['matches']
                ]
            }
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching properties: {e}")
            raise

# Output Layer - Handles response formatting and evaluation
class OutputLayer:
    @staticmethod
    def format_search_results(search_results):
        """
        Format search results for API response
        
        Args:
            search_results (dict): Raw search results
            
        Returns:
            dict: Formatted results for API response
        """
        query = search_results['query']
        matches = search_results['matches']
        
        response = {
            'query': query,
            'result_count': len(matches),
            'properties': []
        }
        
        for match in matches:
            property_data = match['property']
            response['properties'].append({
                'title': property_data.get('title', 'Unknown'),
                'price': property_data.get('price', 'Unknown'),
                'location': property_data.get('location', 'Unknown'),
                'bedrooms': property_data.get('bedrooms', 'Unknown'),
                'bathrooms': property_data.get('bathrooms', 'Unknown'),
                'description': property_data.get('description', 'No description available'),
                'url': property_data.get('url', '#'),
                'relevance_score': round(match['score'] * 100, 2)
            })
        
        return response
    
    @staticmethod
    def evaluate_search_performance(query, results, ground_truth=None):
        """
        Evaluate search performance metrics
        
        Args:
            query (str): Search query
            results (dict): Search results
            ground_truth (list, optional): Known relevant results
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        response_time = results.get('response_time', 0)
        result_count = len(results['matches'])
        
        metrics = {
            'query': query,
            'result_count': result_count,
            'response_time_ms': response_time,
            'average_relevance': round(
                sum(match['score'] for match in results['matches']) / max(1, result_count), 3
            )
        }
        
        # Add precision/recall if ground truth available
        if ground_truth:
            # Implementation for precision/recall would go here
            pass
        
        return metrics

# Initialize the inference layer 
inference_layer = None

@app.before_first_request
def initialize_app():
    """Initialize app dependencies before first request"""
    global inference_layer
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    inference_layer = InferenceLayer(pinecone_api_key)

# API Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/scrape', methods=['POST'])
def scrape_and_index():
    """Scrape properties and index them"""
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = data['url']
    
    # Scrape properties
    properties_df = InputLayer.scrape_property24(url)
    if properties_df.empty:
        return jsonify({'error': 'No properties found'}), 404
    
    # Preprocess text
    text_features = InputLayer.preprocess_text_data(properties_df)
    
    # Create embeddings
    embeddings = inference_layer.create_embeddings(text_features)
    
    # Index properties
    num_indexed = inference_layer.index_properties(properties_df, embeddings)
    
    return jsonify({
        'success': True,
        'properties_indexed': num_indexed,
        'message': f'Successfully indexed {num_indexed} properties'
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search for properties"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query = InputLayer.process_search_query(data['query'])
    top_k = int(data.get('top_k', 5))
    
    # Search properties
    search_results = inference_layer.search_properties(query, top_k)
    
    # Format results
    formatted_results = OutputLayer.format_search_results(search_results)
    
    # Optional: Evaluate performance
    if data.get('evaluate', False):
        metrics = OutputLayer.evaluate_search_performance(query, search_results)
        formatted_results['metrics'] = metrics
    
    return jsonify(formatted_results)

@app.errorhandler(Exception)
def handle_error(e):
    """Handle application errors"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
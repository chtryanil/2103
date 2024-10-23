import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid  # For generating unique IDs

# Initialize ChromaDB client
client = chromadb.Client()

# Create or get a collection for storing conversation memory
collection = client.get_or_create_collection(name="conversation_memory")

def get_embedding_model():
    # Initialize the model if it doesn't exist
    if not hasattr(get_embedding_model, "model"):
        get_embedding_model.model = SentenceTransformer('all-MiniLM-L6-v2')
    return get_embedding_model.model

def add_to_memory(user_input, assistant_response):
    embedding_model = get_embedding_model()  # Initialize the model within the function
    # Combine user input and assistant response for context
    text = f"User: {user_input}\nAssistant: {assistant_response}"
    # Generate embedding
    embedding = embedding_model.encode(text).tolist()
    # Generate a unique ID for the document
    unique_id = str(uuid.uuid4())
    # Add to ChromaDB collection
    collection.add(
        ids=[unique_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{'user_input': user_input, 'assistant_response': assistant_response}]
    )

def get_memory(query, top_k=5):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query).tolist()
    
    # Get the current size of the index
    index_size = collection.count()
    
    if index_size == 0:
        # Return empty results if the collection is empty
        return {'documents': []}
    
    # Adjust n_results based on the index size
    n_results = min(top_k, index_size)
    
    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

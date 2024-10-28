import time
import uuid
import threading
import nltk
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import spacy
import webview
from flask import Flask, render_template_string, request, jsonify

# Import Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Download NLTK data (only need to run once)
nltk.download('punkt')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not found, download it
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize ChromaDB client
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="qa_collection")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(prompt):
    """
    Generates a response using an Ollama model.
    """
    model_name = 'mistral:latest'  # Use 'mistral:latest'
    try:
        # Call the Ollama CLI using subprocess
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        response_text = result.stdout.decode('utf-8')
        return response_text.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred while calling the Ollama model: {e.stderr.decode('utf-8')}"
        print(error_message)
        return "I'm sorry, but I couldn't generate a response at this time."


def grammar_coherence_check(text):
    """
    Checks the grammar and coherence of the text using spaCy.
    """
    doc = nlp(text)
    # Simple coherence check based on POS tagging
    if len(doc) == 0 or not doc.has_annotation("DEP"):
        return False
    return True

def fact_check(response):
    """
    Simple fact-checking using Wikipedia API.
    """
    import wikipedia
    wikipedia.set_lang("en")
    try:
        # Extract key entities from the response
        doc = nlp(response)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART']]
        for entity in entities:
            # Try to find the entity on Wikipedia
            wikipedia.summary(entity)
        return True  # If no exception, assume facts are verifiable
    except Exception:
        return False  # If any exception occurs, assume fact-check failed

def score_response(response, user_question):
    """
    Scores the response based on predefined criteria using NLP techniques.
    """
    # Tokenize sentences
    response_sentences = sent_tokenize(response)
    question_sentences = sent_tokenize(user_question)

    # Correctness and Relevance via Cosine Similarity
    vectorizer = TfidfVectorizer().fit_transform([user_question, response])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    correctness = relevance = min(cosine_sim * 5, 5)  # Scale to 0-5

    # Completeness: Number of key points covered
    completeness = min(len(response_sentences) / max(len(question_sentences), 1) * 5, 5)

    # Coherence and Grammar Check
    coherence = 5 if grammar_coherence_check(response) else 2.5

    # Fact Checking
    fact_check_result = fact_check(response)
    correctness = correctness if fact_check_result else correctness * 0.5

    scores = {
        'correctness': round(correctness, 2),
        'completeness': round(completeness, 2),
        'coherence': round(coherence, 2),
        'relevance': round(relevance, 2)
    }
    total_score = sum(scores.values()) / len(scores)
    scores['total_score'] = round(total_score, 2)
    return scores

def generate_embedding(text):
    """
    Generates an embedding for the given text using Sentence Transformers.
    """
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def store_interaction(question, response, scores, user_feedback):
    """
    Stores the interaction in ChromaDB.
    """
    # Convert values to regular float for compatibility with ChromaDB
    scores = {key: float(value) for key, value in scores.items()}
    user_feedback = float(user_feedback)

    # Flatten the scores dictionary and add it to the metadata
    metadata = {
        'question': question,
        'user_feedback': user_feedback,
        'timestamp': time.time()
    }
    metadata.update(scores)  # Add each score field to the metadata directly

    embedding = generate_embedding(question + " " + response)
    collection.add(
        documents=[response],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[str(uuid.uuid4())]
    )

def compute_combined_score(metadata, distance):
    """
    Computes a combined score for sorting similar interactions.
    Higher scores are better.
    """
    user_feedback = metadata.get('user_feedback', 0)
    total_score = metadata.get('total_score', 0)
    timestamp = metadata.get('timestamp', 0)
    recency = time.time() - timestamp  # Time since the interaction
    # Normalize recency (e.g., interactions within the last week)
    recency_weight = max(0, (604800 - recency) / 604800)  # 604800 seconds in a week
    # Combine the factors
    combined_score = (
        0.5 * user_feedback +
        0.3 * total_score +
        0.2 * recency_weight * 5  # Scale recency_weight to 0-5
    ) - distance  # Subtract distance to prefer closer matches
    return combined_score

def retrieve_similar_interactions(query):
    """
    Retrieves similar past interactions from ChromaDB, prioritizing those with high user feedback and model scores.
    """
    embedding = generate_embedding(query)
    n_initial_results = 10  # Retrieve more results initially
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_initial_results,
        include=['metadatas', 'documents', 'distances']
    )
    # Check if results are empty
    if not results['documents']:
        return []
    # Flatten the results
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    # Combine into a list of dictionaries
    combined_results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        combined_results.append({
            'document': doc,
            'metadata': meta,
            'distance': dist,
            'combined_score': compute_combined_score(meta, dist)
        })
    # Now, sort the combined_results based on combined_score
    combined_results.sort(key=lambda x: -x['combined_score'])  # Higher combined_score is better
    # Select top N results after sorting
    n_final_results = 5
    top_results = combined_results[:n_final_results]
    return top_results

def chain_of_thought(user_question):
    """
    Implements the Chain-of-Thought prompting.
    """
    prompt = f"Question: {user_question}\n\nPlease think through this problem step-by-step and provide your reasoning leading to the final answer.\n\nAnswer:"
    return generate_response(prompt)

def iterative_querying(user_question):
    """
    Breaks down the question into sub-questions and answers them.
    """
    prompt = f"Given the question: '{user_question}', please break it down into smaller sub-questions, answer each one, and then provide a final conclusion.\n\nAnswer:"
    return generate_response(prompt)

def reasoning_loop(response):
    """
    Implements a reasoning loop where the model critiques and refines its answer.
    """
    for i in range(2):  # Adjust the range for more iterations if needed
        critique_prompt = f"Please review the following answer for accuracy and depth, and improve it if necessary:\n\n{response}\n\nImproved Answer:"
        response = generate_response(critique_prompt)
        time.sleep(0.5)  # Reduced sleep time due to local inference
    return response

def react_framework(user_question):
    """
    Utilizes the ReAct framework for reasoning and action.
    """
    prompt = f"Question: {user_question}\n\nThink through this problem step-by-step. Perform any necessary calculations or information retrieval before providing the final answer.\n\nAnswer:"
    return generate_response(prompt)

def analyze_response(response):
    """
    Custom middleware to analyze the response for depth and completeness.
    """
    if len(response.split()) < 100:  # Threshold for minimum length
        return False  # Response is too short
    return True

def enhanced_response(user_question):
    """
    Combines all methods to generate an enhanced response, score it, and store it.
    """
    # Retrieve similar past interactions
    similar_interactions = retrieve_similar_interactions(user_question)

    # Check if similar interactions were found
    if similar_interactions:
        # Incorporate information from past interactions
        past_info = "\n\n".join([item['document'] for item in similar_interactions])
        context_prompt = f"Previous related information:\n{past_info}\n\nNow, answer the following question by considering the above information:\n{user_question}\n\nAnswer:"
        response = generate_response(context_prompt)
    else:
        # No similar past interactions found, proceed with standard process
        response = chain_of_thought(user_question)
        # Step 2: Iterative Querying and Self-Questioning
        response = iterative_querying(user_question)
        # Step 3: Reasoning Loop
        response = reasoning_loop(response)
        # Step 4: ReAct Framework
        response = react_framework(user_question)

    # Step 5: Custom Middleware Analysis
    if not analyze_response(response):
        follow_up_prompt = f"The previous answer was brief. Please provide a more detailed explanation for the question: {user_question}\n\nAnswer:"
        response = generate_response(follow_up_prompt)

    # Score the final response
    scores = score_response(response, user_question)

    return response, scores

app = Flask(__name__)

# HTML template with Bootstrap
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Assistant</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 50px; }
        .container { max-width: 800px; }
        #response { white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Assistant</h1>
        <div class="mb-3">
            <label for="question" class="form-label">Enter your question:</label>
            <input type="text" id="question" class="form-control">
        </div>
        <button onclick="submitQuestion()" class="btn btn-primary">Submit</button>
        <div id="response" class="mt-4"></div>
    </div>

    <script>
        function submitQuestion() {
            var question = document.getElementById('question').value;
            if (question.trim() === '') {
                alert('Please enter a question.');
                return;
            }
            document.getElementById('response').innerHTML = '<p><em>Generating response...</em></p>';
            fetch('/submit_question', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('response').innerHTML = '<h3>Response:</h3><p>' + data.response + '</p>';
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('response').innerHTML = '<p>An error occurred while generating the response.</p>';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/submit_question', methods=['POST'])
def submit_question():
    data = request.get_json()
    question = data['question']

    # Generate the response
    response, scores = enhanced_response(question)
    result = {'response': response, 'scores': scores}

    # Return the response to the frontend
    return jsonify(result)

def start_server():
    app.run()

if __name__ == '__main__':
    # Start the Flask server in a separate thread
    threading.Thread(target=start_server).start()
    # Create the Pywebview window
    webview.create_window('AI Assistant', 'http://localhost:5000')

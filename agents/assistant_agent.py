import subprocess
import os
import yaml
import threading
import time
from agents.agent import Agent
from memory.memory import add_to_memory, get_memory
from search.search import search_google

class AssistantAgent(Agent):
    def __init__(self, name="Assistant"):
        super().__init__(name)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.load_config()

    def load_config(self):
        with open('models/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.assistant_model = config['models']['assistant']['name']
        self.parameters = config['models']['assistant'].get('parameters', {})

    def show_processing_indicator(self, stop_event):
        indicators = ["Processing", "Processing.", "Processing..", "Processing..."]
        i = 0
        while not stop_event.is_set():
            print(f"\r{indicators[i % len(indicators)]}", end='', flush=True)
            i += 1
            time.sleep(0.5)
        print('\r', end='')  # Clear the line when done

    def generate_response(self, input_data):
        query = input_data['input']

        # Retrieve relevant memory from past conversations
        memory_results = get_memory(query)
        memory_context = ""
        if memory_results['documents']:
            memory_context = "\n".join(memory_results['documents'][0])

        # Perform the internet search using Google Custom Search
        search_results = search_google(query)

        # Prepare the prompt
        prompt = self.create_prompt(query, memory_context, search_results)

        # Set up a threading event to control the processing indicator
        stop_event = threading.Event()

        # Start the processing indicator in a separate thread
        indicator_thread = threading.Thread(target=self.show_processing_indicator, args=(stop_event,))
        indicator_thread.start()

        # Generate the response
        response = self.run_model(prompt)

        # Stop the processing indicator
        stop_event.set()
        indicator_thread.join()

        return response

    def run_model(self, prompt):
        # Executes the model command
        try:
            command = ['ollama', 'run', self.assistant_model]

            # Add parameters to the command if any
            if self.parameters:
                for key, value in self.parameters.items():
                    command.extend([f'--{key}', str(value)])

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=prompt)

            if stdout:
                return stdout.strip()
            elif stderr:
                print(f"Model Error: {stderr}")
                return "I'm sorry, I couldn't generate a response."
            else:
                return "I'm sorry, I couldn't generate a response."
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response."

    def create_prompt(self, query, memory_context, search_results):
        # Prepare a summary of search results
        summary = ""
        if search_results:
            summary = "\n".join([f"{result['Title']}: {result['Text']}" for result in search_results])

        # Build the prompt
        prompt = f"""
You are an AI assistant. Use the following conversation history and search results to answer the user's query.

Conversation History:
{memory_context}

Search Results:
{summary}

Current Query:
{query}

Instructions:
- Provide clear and concise answers.
- If you use information from the search results, reference them appropriately.
- Use the user's conversation history to understand context.
- Be friendly and professional.

Example Conversation:

User: What is the capital of France?
Assistant: The capital of France is Paris.

User: Can you explain quantum computing in simple terms?
Assistant: Quantum computing uses principles of quantum mechanics to perform computations more efficiently for certain tasks...

Answer:
"""
        return prompt
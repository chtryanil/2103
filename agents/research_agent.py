import os
import threading
import time
import yaml
from agents.agent import Agent
from memory.memory import add_to_memory
from processing.pdf_processor import extract_text_from_pdf
from processing.web_scraper import scrape_web_content
import subprocess
import requests

class ResearchAgent(Agent):
    def __init__(self, name="ResearchAgent"):
        super().__init__(name)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.load_config()
        self.last_content = None  # Store the last content processed
        self.last_source = None   # Store the source of the last content

    def load_config(self):
        with open('models/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.research_model = config['models']['research']['name']
        self.parameters = config['models']['research'].get('parameters', {})

    def process_input(self, user_input):
        # Check for follow-up questions
        if self.last_content and not user_input.lower().startswith('http') and not user_input.lower().endswith('.pdf'):
            # Treat as a follow-up question
            return {'input': user_input, 'source': 'follow-up'}
        else:
            # Check if the input is a URL or a PDF file path
            if user_input.lower().startswith('http') and user_input.lower().endswith('.pdf'):
                # It's a direct link to a PDF
                content = self.download_and_extract_pdf(user_input)
                return {'input': content, 'source': 'pdf'}
            elif user_input.lower().startswith('http'):
                content = scrape_web_content(user_input)
                return {'input': content, 'source': 'web'}
            elif user_input.lower().endswith('.pdf'):
                # It's a local PDF file path
                content = extract_text_from_pdf(user_input)
                return {'input': content, 'source': 'pdf'}
            else:
                # If it's a regular query, pass it forward
                return {'input': user_input, 'source': 'text'}

    def download_and_extract_pdf(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open('temp_downloaded.pdf', 'wb') as f:
                f.write(response.content)
            content = extract_text_from_pdf('temp_downloaded.pdf')
            os.remove('temp_downloaded.pdf')
            return content
        except Exception as e:
            print(f"Error processing PDF URL: {e}")
            return "Error processing the PDF file."

    def generate_response(self, input_data):
        content = input_data['input']
        source = input_data['source']

        if source == 'follow-up':
            # Use the last content to answer the follow-up question
            prompt = self.create_followup_prompt(self.last_content, content)
        else:
            # Process new content
            self.last_content = content
            self.last_source = source

            # Split content into chunks
            content_chunks = self.split_content_into_chunks(content)

            # Collect individual summaries
            partial_summaries = []

            for chunk in content_chunks:
                # Prepare the prompt
                prompt = self.create_prompt(chunk, source)

                # Set up a threading event to control the processing indicator
                stop_event = threading.Event()

                # Start the processing indicator in a separate thread
                indicator_thread = threading.Thread(target=self.show_processing_indicator, args=(stop_event,))
                indicator_thread.start()

                # Generate the response using the model
                summary = self.run_model(prompt)

                # Stop the processing indicator
                stop_event.set()
                indicator_thread.join()

                partial_summaries.append(summary)

            # Combine the partial summaries
            combined_summary = "\n".join(partial_summaries)

            # Optionally, generate an overall summary of the summaries
            final_prompt = self.create_final_prompt(combined_summary)
            final_summary = self.run_model(final_prompt)

            # Add the interaction to memory
            add_to_memory(content, final_summary)

            return final_summary

    def show_processing_indicator(self, stop_event):
        indicators = ["Researching", "Researching.", "Researching..", "Researching..."]
        i = 0
        while not stop_event.is_set():
            print(f"\r{indicators[i % len(indicators)]}", end='', flush=True)
            i += 1
            time.sleep(0.5)
        print('\r', end='')  # Clear the line when done

    def run_model(self, prompt):
        try:
            command = ['ollama', 'run', self.research_model]

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

    def create_prompt(self, content, source):
        # Build the prompt based on the source
        if source == 'pdf' or source == 'web':
            prompt = f"""
You are a research assistant. Analyze the following content in depth:

{content}

Instructions:
- Provide a comprehensive summary.
- Highlight key points, methodologies, and findings.
- Discuss the significance of the work.
- Critically evaluate the strengths and weaknesses.
- Use clear and professional language.

Answer:
"""
        else:
            prompt = f"""
You are a research assistant. Provide detailed information on the following topic:

{content}

Instructions:
- Provide comprehensive insights.
- Reference relevant sources if applicable.
- Use clear and professional language.

Answer:
"""
        return prompt

    def split_content_into_chunks(self, content, max_tokens=1024):
        # Approximate token count by splitting the content
        words = content.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1  # Approximate one word as one token for simplicity
            if current_length >= max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_final_prompt(self, summaries):
        prompt = f"""
You are a research assistant. Based on the following summaries of different sections of a document, provide a comprehensive and deep analysis:

{summaries}

Instructions:
- Combine the key insights from each summary.
- Highlight important findings, methodologies, and conclusions.
- Provide critical analysis where appropriate.
- Use clear and professional language.

Answer:
"""
        return prompt

    def create_followup_prompt(self, content, question):
        prompt = f"""
You are a research assistant. Based on the following content:

{content}

Answer the following question:

{question}

Instructions:
- Provide a detailed answer based on the content.
- Use clear and professional language.

Answer:
"""
        return prompt

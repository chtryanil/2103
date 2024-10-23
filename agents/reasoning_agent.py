# agents/reasoning_agent.py

from agents.agent import Agent
from reasoning.text_analyzer import is_ambiguous, analyze_text

class ReasoningAgent(Agent):
    def __init__(self, name="ReasoningAgent"):
        super().__init__(name)
        self.load_config()

    def load_config(self):
        # Load any specific configurations if necessary
        pass

    def process_input(self, user_input):
        # Check if the input is a URL or a PDF file path
        if user_input.lower().startswith('http') or user_input.lower().endswith('.pdf'):
            # Skip processing and pass the input as is
            return {
                'status': 'processed',
                'input': user_input
            }

        # Analyze and preprocess the user input
        if is_ambiguous(user_input):
            # Handle ambiguity
            return {
                'status': 'clarification_needed',
                'message': "Your question seems ambiguous. Could you please clarify?"
            }
        else:
            # Clean the text
            cleaned_input = analyze_text(user_input)
            return {
                'status': 'processed',
                'input': cleaned_input
            }

    def generate_response(self, input_data):
        # This agent doesn't generate a final response but can provide clarification
        if input_data['status'] == 'clarification_needed':
            return input_data['message']
        else:
            # Pass the processed input forward
            return None

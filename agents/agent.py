# agent.py

class Agent:
    def __init__(self, name):
        self.name = name
        self.parameters = {}

    def load_config(self):
        # Implement configuration loading in subclasses if needed
        pass

    def process_input(self, user_input):
        # Preprocess the user input if necessary
        return user_input

    def generate_response(self, input_data):
        # Generate a response based on the input data
        raise NotImplementedError("Subclass must implement generate_response method.")

    def interact(self):
        # Define interaction flow in subclasses
        raise NotImplementedError("Subclass must implement interact method.")
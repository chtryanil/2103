from memory.memory import add_to_memory, get_memory
class AgentManager:
    def __init__(self):
        self.agents = []

    def register_agent(self, agent):
        self.agents.append(agent)

    def interact(self):
        print("Welcome to the AI assistant. Type 'quit' to exit.")
        while True:
            # Get user input
            user_input = input("\nEnter your question: ")

            if user_input.lower() == 'quit':
                print("Thank you for using the AI assistant. Goodbye!")
                break

            input_data = {'input': user_input}
            for agent in self.agents:
                result = agent.process_input(input_data['input'])

                if result is not None:
                    if 'status' in result and result['status'] == 'clarification_needed':
                        print(f"\n{agent.name}: {result['message']}")
                        break  # Skip to next iteration to get new input
                    elif 'input' in result:
                        input_data = result  # Update input data with processed input

                response = agent.generate_response(input_data)

                if response:
                    print(f"\n{agent.name}: {response}")
                    # Add the interaction to memory
                    if agent.name == 'Assistant':
                        add_to_memory(user_input, response)
                    break  # Response generated, move to next user input
            else:
                print("No agent could handle the input.")
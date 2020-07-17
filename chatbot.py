import numpy
import re

from sequence_generation import max_encoder_seq_length, num_encoder_tokens, input_features_dict
from testing_setup import decode_response


class ChatBot:

    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    # method to start the conversation
    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?\n")

        if user_response in self.negative_responses:
            print("Sucks to be you. Bye.")
            return
        self.chat(user_response)

    # method to handle the conversation
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply) + "\n")

    # method to convert user input into a matrix
    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = numpy.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    # method that will create a response using a seq2seq model we built
    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)

        # remove <START> and <END> tokens from chatbot_response
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')

        return chatbot_response

    # method to check for exit commands
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Okay, have a great day.")
                return True
        return False


chatbot = ChatBot()
chatbot.start_chat()

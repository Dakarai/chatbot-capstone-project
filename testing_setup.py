import numpy
from keras.layers import Input
from keras.models import Model, load_model
from training_model import dimensionality, decoder_lstm, decoder_inputs, decoder_dense, num_decoder_tokens
from sequence_generation import target_features_dict, reverse_target_features_dict, max_decoder_seq_length

training_model = load_model("training_model.h5")
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_hidden = Input(shape=(dimensionality,))
decoder_state_input_cell = Input(shape=(dimensionality,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_response(test_input):
    # getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)

    # generating empty target sequence of length 1
    target_seq = numpy.zeros((1, 1, num_decoder_tokens))

    # setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # a variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False

    while not stop_condition:
        # predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

        # choosing the one with the highest probability
        sampled_token_index = numpy.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # stop if hit max length or found the stop token
        if sampled_token == "<END>" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        # update the target sequence
        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # update states
        states_value = [hidden_state, cell_state]

        return decoded_sentence

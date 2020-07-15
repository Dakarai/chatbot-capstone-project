from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sequence_generation import num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data

# dimensionality
dimensionality = 256

# the batch size and number of epochs
batch_size = 50
epochs = 600

# encoder training setup
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# decoder training setup
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# building the training model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# print summary (optional)
print("Model Summary:\n")
training_model.summart()
print("\n\n")

# compile the model
training_model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs = epochs, validation_split=0.2)
training_model.save('training_model.h5')

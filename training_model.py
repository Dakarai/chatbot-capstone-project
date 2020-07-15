from tensorflow import  keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# dimensionality
dimensionality = 256

# the batch size and number of epochs
batch_size = 10
epochs = 600

# encoder


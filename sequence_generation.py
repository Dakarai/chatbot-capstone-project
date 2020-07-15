from preprocessing import pairs
import numpy

# building empty lists to hold sentences
input_docs = []
target_docs = []

# building empty vocabulary sets
input_tokens = set()
target_tokens = set()

for line in pairs:
    input_doc, target_doc = line[0], line[1]

    # appending each input sentence to input_docs
    input_docs.append(input_doc)

    # splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

    # redefine target_doc and append it to target_docs

    target_doc = "<START> " + target_doc + " <END>"
    target_docs.append(target_doc)

    # split up each sentence into words and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add((token))

# sort the token lists then find the count
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# create dictionaries
# dict: (token, index from input_tokens/target_tokens)
input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

# create reverse dictionaries
# dict: (index from input_tokens/target_tokens, token)
reverse_input_features_dict = dict([(i, token) for token, i in input_features_dict.items()])
reverse_target_features_dict = dict([(i, token) for token, i in target_features_dict.items()])


# Maximum length of sentences in input and target documents
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

# create numpy matrices
encoder_input_data = numpy.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = numpy.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = numpy.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

# create one-hot vectors
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]"), input_doc):
        # assign 1. for the current line, timestep, and word in encoder_input_data
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep-1, target_features_dict[token]] = 1.

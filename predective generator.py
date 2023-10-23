# Import required libraries
import numpy as np
import tensorflow as tf
import requests
import string
import random

# Download and preprocess the dataset (e.g., Shakespeare's writings)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
text = text[:10000]  # Using the first 10,000 characters for simplicity

# Create a set of unique characters in the text
chars = sorted(set(text))

# Create a mapping of characters to integers and vice versa
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Create training sequences
sequence_length = 100
sequences = []
next_chars = []

for i in range(0, len(text) - sequence_length, 1):
    seq = text[i:i+sequence_length]
    target = text[i+sequence_length]
    sequences.append([char_to_idx[char] for char in seq])
    next_chars.append(char_to_idx[target])

# Prepare data for training
X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char_idx in enumerate(sequence):
        X[i, t, char_idx] = 1
    y[i, next_chars[i]] = 1

# Build an RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(sequence_length, len(chars))),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=128, epochs=30)

# Generate text
start_index = random.randint(0, len(text) - sequence_length - 1)
seed_text = text[start_index:start_index + sequence_length]

generated_text = seed_text
for _ in range(400):
    sampled = np.zeros((1, sequence_length, len(chars)))
    for t, char in enumerate(seed_text):
        sampled[0, t, char_to_idx[char]] = 1

    preds = model.predict(sampled, verbose=0)[0]
    next_index = np.random.choice(len(chars), p=preds)
    next_char = idx_to_char[next_index]

    generated_text += next_char
    seed_text = seed_text[1:] + next_char

print(generated_text)



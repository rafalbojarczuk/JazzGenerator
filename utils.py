import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import music21
import json



#dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')

def load_dictionaries():
	with open(os.path.join(os.getcwd(), 'lookup_dictionaries.json'), 'r') as file:
		(notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset) = json.load(file)
	return (notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset)

def load_training_data():
	with open(os.path.join(os.getcwd(), 'training_data.json'), 'r') as file:
		(notes, durations, offsets) = json.load(file)
	return (notes, durations, offsets)


def music_to_network_input(notes, durations, offsets, seq_length=32):
	notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset = load_dictionaries()
	notes_num = len(notes_to_int)
	dur_num = len(duration_to_int)
	off_num = len(offset_to_int)


	notes_network_input = []
	notes_network_output = []
	durations_network_input = []
	durations_network_output = []
	offsets_network_input = []
	offsets_network_output = []

	for i in range(len(notes) - seq_length):
		notes_sequence_in = notes[i:i+seq_length]
		notes_out = notes[i+seq_length]
		notes_network_input.append([notes_to_int[chord] if chord in notes_to_int.keys() else notes_to_int['UNKNOWN'] for chord in notes_sequence_in])
		notes_network_output.append([notes_to_int[notes_out] if notes_out in notes_to_int.keys() else notes_to_int['UNKNOWN']])

		durations_sequence_in = durations[i:i+seq_length]
		duration_out = durations[i+seq_length]
		durations_network_input.append([duration_to_int[str(dur)] for dur in durations_sequence_in])
		durations_network_output.append(duration_to_int[str(duration_out)])

		offsets_sequence_in = offsets[i:i+seq_length]
		offset_out = offsets[i+seq_length]
		offsets_network_input.append([offset_to_int[str(off)] for off in offsets_sequence_in])
		offsets_network_output.append(offset_to_int[str(offset_out)])

	n_patterns = len(notes_network_input)

	notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_length))
	durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_length))
	offsets_network_input = np.reshape(offsets_network_input, (n_patterns, seq_length))

	network_input = [notes_network_input, durations_network_input, offsets_network_input]

	notes_network_output = to_categorical(notes_network_output, num_classes=notes_num)
	durations_network_output = to_categorical(durations_network_output, num_classes=dur_num)
	offsets_network_output = to_categorical(offsets_network_output, num_classes=off_num)

	network_output = [notes_network_output, durations_network_output, offsets_network_output]

	return (network_input, network_output)


def create_RNN_model(notes_num, dur_num, off_num, embedding_size, rnn_units=256, add_attention=False):
	#dur == durations, off == offset
	notes_input = Input(shape = (None,))
	dur_input = Input(shape = (None,))
	off_input = Input(shape = (None,))

	notes_embed_layer = Embedding(input_dim = notes_num, output_dim = embedding_size)(notes_input)
	dur_embed_layer =  Embedding(input_dim = dur_num, output_dim = embedding_size//10)(dur_input)
	off_embed_layer = Embedding(input_dim = off_num, output_dim = embedding_size//10)(off_input)

	embedding_layer = Concatenate()([notes_embed_layer, dur_embed_layer, off_embed_layer])

	lstm_1 = LSTM(rnn_units, return_sequences=True)(embedding_layer)
	if add_attention:
		lstm_2 = None
	else:
		lstm_2 = LSTM(rnn_units)(lstm_1)

	notes_out = Dense(notes_num, activation='softmax')(lstm_2)
	dur_out = Dense(dur_num, activation='softmax')(lstm_2)
	off_out = Dense(off_num, activation='softmax')(lstm_2)

	model = Model([notes_input, dur_input, off_input], [notes_out, dur_out, off_out])
	opt = tf.keras.optimizers.Adam(lr = 0.001)
	model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model

def sample(predictions, temp):
	if temp == 0:
		return np.argmax(predictions)
	else:
		predictions =np.exp(np.log(predictions) / temp)
		predictions /= np.sum(predictions)
		return np.random.choice(len(predictions), p=predictions)

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import os
import mido
import py_midicsv
import sys
import music21


def chord_to_one_hot(chord):
	one_hot = np.zeros(shape=(96,1), dtype=np.int32)
	base_note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	for note in chord.pitches:
		base_note_index = base_note.index(note.name[0])
		note_index = base_note_index + 12*note.octave
		if '#' in note.name:
			note_index+=1
		elif '-' in note.name:
			note_index-=1
		one_hot[note_index] = 1

	return one_hot

def midi_to_matrix(midi):
	matrix = np.empty(shape=(98,1), dtype=np.int32)
	prev_offset = 0.0
	for chord in midi.flat.notes:
		one_hot_list = chord_to_one_hot(chord)
		#smallest duration times/offsets are 0.25 and 0.33 so I multiply by 12 to convert them all to integer
		one_hot_list = np.append(one_hot_list, round(chord.duration.quarterLength*12))
		one_hot_list = np.append(one_hot_list, round(12*((chord.offset) - prev_offset)))
		prev_offset = float(chord.offset)
		matrix = np.hstack([matrix, one_hot_list.reshape(98,1)])
	matrix = np.delete(matrix, obj=0, axis=1)
	print(matrix.shape)
	return matrix

dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')

for file in os.listdir(dataset_dir)[:1]:
	if file.endswith(".mid"):
		path = os.path.join(dataset_dir, file)
		midi = music21.converter.parse(path)
		print(midi_to_matrix(midi)[90:,-5:].T)
		
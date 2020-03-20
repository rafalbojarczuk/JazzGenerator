import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import music21
from music21 import note, chord, duration, instrument
from utils import create_RNN_model, load_dictionaries, load_training_data, music_to_network_input, sample


notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset = load_dictionaries()
notes_num = len(notes_to_int)
dur_num = len(duration_to_int)
off_num = len(offset_to_int)

weigths_dir = os.path.join(os.getcwd(), 'weights')
weigths_file = 'weights.h5'
weigths = os.path.join(weigths_dir, weigths_file)

model = create_RNN_model(notes_num, dur_num, off_num, embedding_size=300, rnn_units=256, add_attention=False)
model.load_weights(weigths)


max_extra_notes = 100
max_seq_len = 32
seq_len = 32

#notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
#durations = [0, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24]
#offsets = [0, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24]


#notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
#durations = [0, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24]
#offsets = [0, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24, 9, 3, 12, 12, 12, 24]

notes = ['START', 'C4.E4.G4', 'B3.C4.E4', 'D4']
durations = [0, 6, 3, 6]
offsets = [0, 0, 6, 3]

if seq_len is not None:
    notes = ['START'] * (seq_len - len(notes)) + notes
    durations = [0] * (seq_len - len(durations)) + durations
    offsets = [0] * (seq_len - len(offsets)) + offsets


sequence_length = len(notes)

prediction_output = []
notes_input_sequence = []
durations_input_sequence = []
offsets_input_sequence = []

overall_preds = []

for n,d,o in zip(notes, durations, offsets):
	notes_input_sequence.append(notes_to_int[n])
	durations_input_sequence.append(duration_to_int[str(d)])
	offsets_input_sequence.append(offset_to_int[str(o)])
	prediction_output.append([n,d,o])


for note_index in range(max_extra_notes):

	prediction_input = [np.array([notes_input_sequence])
						, np.array([durations_input_sequence])
						, np.array([offsets_input_sequence])]
	notes_prediction, durations_prediction, offsets_prediction = model.predict(prediction_input)

	new_note = np.zeros(128)

	for idx, n_i in enumerate(notes_prediction[0]):
		try:
			note_name = int_to_notes[str(idx)]
			midi_note = music21.note.Note(note_name)
			new_note[midi_note.pitch.midi] = n_i
		except:
			pass


	overall_preds.append(new_note)
	temperature = 2.5

	note_result = int_to_notes[str(sample(notes_prediction[0], temperature))]
	duration_result = int_to_duration[str(sample(durations_prediction[0], 2.5))]
	offset_result = int_to_offset[str(sample(offsets_prediction[0], 1.5))]

	prediction_output.append([note_result, duration_result, offset_result])

	if len(notes_input_sequence) > max_seq_len:
		notes_input_sequence = notes_input_sequence[1:]
		durations_input_sequence = durations_input_sequence[1:]

	if note_result == 'START':
		break

overall_preds = np.transpose(np.array(overall_preds))

print(f"Generated {len(prediction_output)} notes")

save_folder = os.path.join(os.getcwd(), 'generated music')

midi = music21.stream.Stream()

offset = 0
for notes_result, duration_result, offset_result in prediction_output:
	offset += int(offset_result)
	if('.' in notes_result):
		notes = notes_result.split('.')
		chord = []
		for single_note in notes:
			#dividing by 12 to go back to original duration and offset
			new_note = note.Note(single_note)
			new_note.duration = duration.Duration(int(duration_result)/12)
			new_note.offset = float(offset/12)
			new_note.storedInstrument = instrument.Piano()
			chord.append(new_note)
		midi.append(music21.chord.Chord(chord))
	elif notes_result == 'rest':
		new_note = note.Rest()
		new_note.duration = duration.Duration(int(duration_result)/12)
		new_note.offset = float(offset/12)
		new_note.storedInstrument = instrument.Piano()
		midi.append(new_note)
	elif notes_result != 'START' and notes_result != 'UNKNOWN':
		new_note = note.Note(notes_result)
		new_note.duration = duration.Duration(int(duration_result)/12)
		new_note.offset = float(offset/12)
		new_note.storedInstrument = instrument.Piano()
		midi.append(new_note)
		#print(f"{new_note} - {new_note.duration}")
midi = midi.chordify()
for element in midi.flat.notes:
	print(f"{element} - {element.duration.quarterLength} - {element.offset}")
midi.write('midi', fp=os.path.join(save_folder, 'output-after-one-epoch' + '.mid'))

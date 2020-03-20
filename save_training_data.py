import os
import music21
import json


dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')
notes = []
durations = []
offsets = []
for file in os.listdir(dataset_dir):
	if file.endswith(".mid"):
		print(f"Gathering informations from: {file}")
		path = os.path.join(dataset_dir, file)
		midi = music21.converter.parse(path)
		parts = music21.instrument.partitionByInstrument(midi)
		if parts:
			notes_to_parse = parts.parts[0].recurse().notes
		else:
			notes_to_parse = midi.flat.notes
		prev_offset = 0.0
		for element in notes_to_parse:
			if isinstance(element, music21.note.Note):
				notes.append(element.nameWithOctave)
			elif isinstance(element, music21.chord.Chord):
				chord = '.'.join(sorted([note.nameWithOctave for note in element.pitches]))
				notes.append(chord)
			durations.append(str(element.duration.quarterLength))
			#smallest offsets are 0.25 and 0.33 so I multiply by 12 and round
			#it prevents adding different floating-point approximations to the dictionary
			#i.e. 0.333312 and 0.33335 should be treated as the same value 1/3
			offsets.append(round(12*(element.offset - prev_offset)))
			prev_offset = element.offset
	
with open('training_data.json', 'w') as file:
	json.dump((notes, durations, offsets), file)
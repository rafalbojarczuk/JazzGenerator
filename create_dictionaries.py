import os
import sys
import music21
import json

dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')

notes_to_int = {"rest" : 0}
durations = []
offsets = []
note_ind = 1
for file in os.listdir(dataset_dir):
	if file.endswith(".mid"):
		print(file)
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
				if element.nameWithOctave not in notes_to_int.keys():
					notes_to_int[element.nameWithOctave] = note_ind
					note_ind+=1
			elif isinstance(element, music21.chord.Chord):
				chord = '.'.join(note.nameWithOctave for note in element.pitches)
				if chord not in notes_to_int.keys():
					notes_to_int[chord] = note_ind
					note_ind+=1
			durations.append(element.duration.quarterLength)
			#smallest offsets are 0.25 and 0.33 so I multiply by 12 and round
			#it prevents adding different floating-point approximations to the dictionary
			#i.e. 0.333312 and 0.33335 should be treated as the same value 1/3
			offsets.append(round(12*(element.offset - prev_offset)))
			prev_offset = element.offset
durations = sorted(set(item for item in durations))
offsets = sorted(set(item for item in offsets))
duration_to_int = dict((str(duration), number) for number, duration in enumerate(durations))
offset_to_int = dict((offset, number) for number, offset in enumerate(offsets))

with open('notes_to_int.json', 'w') as file:
	json.dump(notes_to_int, file)
with open('duration_to_int.json', 'w') as file:
	json.dump(duration_to_int, file)
with open('offset_to_int.json', 'w') as file:
	json.dump(offset_to_int, file)
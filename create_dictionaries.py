import os
import sys
import music21
import json

dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')

notes = ["rest", "START"]
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
			durations.append(element.duration.quarterLength)
			#smallest offsets are 0.25 and 0.33 so I multiply by 12 and round
			#it prevents adding different floating-point approximations to the dictionary
			#i.e. 0.333312 and 0.33335 should be treated as the same value 1/3
			offsets.append(round(12*(element.offset - prev_offset)))
			prev_offset = element.offset
notes = sorted(set(item for item in notes))
durations = sorted(set(item for item in durations))
offsets = sorted(set(item for item in offsets))
notes_to_int = dict((note, number) for number, note in enumerate(notes))
int_to_notes = dict((number, note) for number, note in enumerate(notes))
duration_to_int = dict((str(duration), number) for number, duration in enumerate(durations))
int_to_duration = dict((number, str(duration)) for number, duration in enumerate(durations))
offset_to_int = dict((offset, number) for number, offset in enumerate(offsets))
int_to_offset = dict((number, offset) for number, offset in enumerate(offsets))

with open('lookup_dictionaries.json', 'w') as file:
	json.dump((notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset), file)

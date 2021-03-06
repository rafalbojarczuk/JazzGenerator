import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import sys
import music21
from utils import create_RNN_model, load_dictionaries, load_training_data, music_to_network_input


seq_length = 32
notes, durations, offsets = load_training_data()
network_input, network_output = music_to_network_input(notes, durations, offsets, seq_length)

notes_to_int, int_to_notes, duration_to_int, int_to_duration, offset_to_int, int_to_offset = load_dictionaries()
notes_num = len(notes_to_int)
dur_num = len(duration_to_int)
off_num = len(offset_to_int)

model = create_RNN_model(notes_num, dur_num, off_num, embedding_size=300, rnn_units=256, add_attention=False)

model.summary()



weights_folder = os.path.join(os.getcwd(), 'weights')

checkpoint1 = ModelCheckpoint(
    os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

checkpoint2 = ModelCheckpoint(
    os.path.join(weights_folder, "weights.h5"),
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='loss'
    , restore_best_weights=True
    , patience = 10
)


callbacks_list = [
    checkpoint1
    , checkpoint2
    , early_stopping
 ]

model.save_weights(os.path.join(weights_folder, "weights.h5"))
model.fit(network_input, network_output
          , epochs=20, batch_size=32
          , validation_split = 0.08
          , callbacks=callbacks_list
          , shuffle=True
         )
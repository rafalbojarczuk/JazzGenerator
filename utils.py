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
import pickle



dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'jazzdataset')

		
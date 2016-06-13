import json
import h5py
from keras.models import model_from_json
import os

def save_model(model, directory):
    os.makedirs(directory, exist_ok=True)
    json_string = model.to_json()
    with open('{}/architechture.json'.format(directory), 'w') as outfile:
        outfile.write(json_string)
    model.save_weights('{}/weights.h5'.format(directory))

def load_model(directory):
    with open('{}/architechture.json'.format(directory), 'r') as infile:
        model = model_from_json(infile.read())
    model.load_weights('{}/weights.h5'.format(directory))
    return model

import csv
import os
intent_to_output = dict()

def load_intent_to_output(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            intent_to_output[row[0]] = row[1]

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'intent_to_output.csv')
load_intent_to_output(file_path)
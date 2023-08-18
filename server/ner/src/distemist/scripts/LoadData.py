from pymongo import MongoClient
import os

client = MongoClient("mongodb://db:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1")

db = client['ClinicalDB']
collection = db['patientsData']


def load_text_file(file_path):
    id = path.splitext(path.basename(file_path))[0]

    with open(file_path, 'r') as file:
        text = file.read()

    document = {
        "_id": id,
        "text": text,
        "ann": ""
    }

    collection.insert_one(document)



corpus_path = "src/distemist/datasets/distemist/"

directory = corpus_path + "training/text_files/"

path = os.path

files = [f for f in os.listdir(directory) if path.isfile(path.join(directory, f))]


for file in files:
    load_text_file(path.join(directory, file))
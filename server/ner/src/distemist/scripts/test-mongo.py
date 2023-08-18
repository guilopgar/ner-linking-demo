from pymongo import MongoClient
from os import path, listdir

client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1")

db = client['ClinicalDB']
collection = db['patientsData']


all_documents = collection.find()


for doc in all_documents:
    id = doc.get('_id')
    text = doc.get('text')
    
    print("ID:" + id + ", Text:" + text)
import pymongo
from pymongo import MongoClient
client = MongoClient("mongodb+srv://nandish:12345@cluster0.q0vwo.mongodb.net/attention?retryWrites=true&w=majority")
db=cluster["attention"]
collection=db["test"]
post1={"_id":5,"name":"joe"}
post2={"_id":6,"name":"bill"}
collection.insert_one(post1)
print (client)

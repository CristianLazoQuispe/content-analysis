import numpy as np
import requests
import json

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

with open('message.txt', 'r') as file:
    data = file.read().rstrip().replace('\n', '')
# Serialize the data into json and send the request to the model
payload = {'data': data}

# "porn is prohibited"
# where can i buy illegal weapons
# Watch free porn videos at pornrox com with new clips added daily Find sexy pornstars on your desktop or mobile phone without worrying about annoying adds and streaming speeds that will make you flip the table We bring the porn tube experience to your door in a secure connection for free

print("Sending.. : ",payload)
response = requests.post('http://127.0.0.1:5000/predict', json=payload).json()
print("response : ",response)

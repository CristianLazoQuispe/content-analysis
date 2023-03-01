from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class Analyze_sentence(Resource):
    def post(self):
        args = parser.parse_args()
        print(args)
        sentence = np.array(json.loads(args['data']))
        print(sentence)
        preprocessed = tf_model.transform(sentence)
        return clf_model.predict(preprocessed)[0]
    
api.add_resource(Analyze_sentence, '/predict')

if __name__ == '__main__':
    # Load model
    tf_model = pickle.load(open("Results/preprocess.pkl", 'rb'))
    clf_model = pickle.load(open("Results/model.pkl", 'rb'))
    
    app.run(debug=True)
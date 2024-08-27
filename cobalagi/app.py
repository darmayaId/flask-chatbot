from flask import Flask, render_template, request
import nltk
import pickle
import numpy as np
import random
from keras.models import load_model
import json
nltk.download('wordnet')


# Inisialisasi Flask
app = Flask(__name__)
app.static_folder = 'static'

# Download dan inisialisasi NLTK lemmatizer
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Muat model yang telah dilatih
model = load_model('modelas.h5')

# Muat file JSON dan pickle
# Muat file JSON dengan encoding yang benar
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('tek.pkl', 'rb'))
classes = pickle.load(open('lab.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenisasi kalimat - pisahkan kata-kata menjadi array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize setiap kata - buat bentuk singkat untuk kata
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenisasi kalimat
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matriks N kata, matriks kosakata
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # Beri nilai 1 jika kata saat ini ada di posisi kosakata
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    print("Results:", results)  # Tambahkan ini
    return_list = []
    for r in results:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})
    print("Return list:", return_list)  # Tambahkan ini
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()

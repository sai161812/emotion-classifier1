import joblib

emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
model = joblib.load("emotion_classifier.pkl")

while True:
    text = input("Enter text to classify: ")
    print(emotions[model.predict([text])[0]])
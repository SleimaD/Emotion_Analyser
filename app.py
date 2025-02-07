# Import necessary modules from Flask and Transformers.
from flask import Flask, render_template, request
from transformers import pipeline

#! instance of the Flask application.
app = Flask(__name__)


#! Initialize the emotion analysis pipeline.
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base", 
    device="cpu"
) #! "j-hartmann/emotion-english-distilroberta-base" model is pre-trained for emotion classification.

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            results = emotion_analyzer(text)
            return render_template('result.html', text=text, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

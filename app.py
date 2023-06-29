from flask import Flask, request, send_file
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import soundfile as sf
import os
import io

app = Flask(__name__)

# Load your model here
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# hello world route
@app.route("/")
def hello():
    return "Hello World!"

# get the model's name using method GET
@app.route("/model", methods=["GET"])
def get_model():
    return "microsoft/speecht5_tts"


@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = request.json["text"]
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Instead of writing to a file, write to an in-memory bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, speech.numpy(), samplerate=16000, format='WAV')
    buffer.seek(0)

    return send_file(buffer, mimetype='audio/wav')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 8080)))

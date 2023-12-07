from flask import Flask, render_template, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BartForConditionalGeneration, BertTokenizer, BartTokenizer,BertModel , BartModel
import os
import torch
import soundfile as sf
import librosa
from pytube import YouTube
import re
from langdetect import detect
from language_tool_python import LanguageTool
from google.cloud import speech_v1p1beta1 as speech
from googletrans import Translator
import ffmpeg
import subprocess


app = Flask(__name__)

# Set the path to your service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\krish\OneDrive\Desktop\GOOGLE_APPLICATION_CREDENTIALS.json"
client = speech.SpeechClient()

# Configuration variables
MODEL_NAME_WAV2VEC2 = "facebook/wav2vec2-base-960h"
MODEL_NAME_BERT = "bert-base-uncased"
MODEL_NAME_BART = "facebook/bart-large-cnn"

def initialize_wav2vec2_model():
    model_name = "facebook/wav2vec2-base-960h"
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return tokenizer, model

def initialize_bert_model():
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer_bert, model_bert

def initialize_bart_model():
    tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model_bart = BartModel.from_pretrained('facebook/bart-large-cnn')
    return tokenizer_bart, model_bart

# Initialize ASR and summarization models
tokenizer_wav2vec2, model_wav2vec2 = initialize_wav2vec2_model()
tokenizer_bert, model_bert = initialize_bert_model()
tokenizer_bart = BartTokenizer.from_pretrained(MODEL_NAME_BART)
model_bart = BartForConditionalGeneration.from_pretrained(MODEL_NAME_BART)

ffmpeg_path = r"C:\ffmpeg_Path\ffmpeg.exe"
    
def download_youtube_video(youtube_url, output_path):
    try:
        # Download YouTube video
        yt = YouTube(youtube_url)
        ys = yt.streams.filter(only_audio=True).first()
        ys.download(output_path)

        # Use ffmpeg to convert video to audio
        video_path = os.path.join(output_path, ys.default_filename)
        audio_path = os.path.join(output_path, f"{os.path.splitext(ys.default_filename)[0]}.mp3")

        # Print statements for debugging
        print(f"Video downloaded to: {video_path}")
        print(f"Converting video to audio. Output path: {audio_path}")

        subprocess.run([ffmpeg_path, "-i", video_path, audio_path])

        # Delete the downloaded video file
        os.remove(video_path)

        # Print statement for debugging
        print("Audio conversion completed.")

        return audio_path
    except Exception as e:
        print(f"Error downloading and extracting audio from YouTube video: {str(e)}")
        return None

def post_process_transcription(transcription):
    # Remove unreadable words using regular expression
    transcription = re.sub(r'\b\w{1,2}\b', '', transcription)
    # Remove audio noise (repeated characters)
    transcription = re.sub(r'(.)\1{2,}', r'\1', transcription)
    return transcription

def perform_ASR(audio_chunk_path, chunk_size=25):
    try:
        if audio_chunk_path is None:
            print("Error: Audio file path is None")
            return None

        # Initialize the Wav2Vec2 tokenizer and model
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Load the entire audio file using librosa
        audio_input, _ = librosa.load(audio_chunk_path, sr=16000)

        # Determine the number of chunks based on the specified size
        num_chunks = len(audio_input) // (chunk_size * 16000)

        transcriptions = []

        # Process each chunk separately
        for i in range(num_chunks + 1):
            start_idx = i * chunk_size * 16000
            end_idx = (i + 1) * chunk_size * 16000

            # Extract the current chunk of audio
            current_chunk = audio_input[start_idx:end_idx]

            # Perform ASR on the chunk
            input_values = tokenizer(current_chunk, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)

            transcriptions.append(" ".join(transcription))

        # Combine transcriptions from all chunks
        final_transcription = " ".join(transcriptions)

        # Post-process the final transcription
        final_transcription = post_process_transcription(final_transcription)

        return final_transcription
    except Exception as e:
        print(f"Error in perform_ASR: {str(e)}")
        return None
    
def extract_transcript(youtube_url):
    try:
        # Use ASR to transcribe the entire audio
        audio_path = download_youtube_video(youtube_url, '.')

        if audio_path is None:
            return None, 'en'  # Return None if there's an issue with downloading audio

        full_transcript = perform_ASR(audio_path)

        if full_transcript is None:
            return None, 'en'  # Return None if there's an issue with ASR

        # Detect the language of the transcription
        detected_lang = detect(full_transcript)

        return full_transcript.lower(), detected_lang
    except Exception as e:
        print(f"Error extracting transcript using ASR: {str(e)}")
        return None, 'en'
        
def summarize_asr(audio_chunk_paths, target_summary_length=200):
    try:
        combined_transcription = ""
        for audio_chunk_path in audio_chunk_paths:
            asr_transcription = perform_ASR(audio_chunk_path)
            if asr_transcription:
                combined_transcription += asr_transcription + " "

        if not combined_transcription:
            return ""

        # Tokenize and summarize using Bart model
        tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer_bart(combined_transcription, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model_bart.generate(inputs.input_ids, max_length=target_summary_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)

        # Perform grammar check on summary
        summary = grammar_check(summary)

        return summary.lower()  # Convert summary to lowercase
    except Exception as e:
        print(f"Error in summarize_asr: {str(e)}")
        return ""
    
def grammar_check(text, language='en-US'):
    if language.lower() == 'hi':
        language_tool = LanguageTool('hi')
    else:
        language_tool = LanguageTool('en-US')
    matches = language_tool.check(text)
    return language_tool.correct(text)

def summarize_transcript(full_transcript):
    try:
        # Tokenize the transcript into chunks for summarization
        chunks = [full_transcript[i:i+1024] for i in range(0, len(full_transcript), 1024)]

        summary_pieces = []

        for chunk in chunks:
            tokens = tokenizer_bart(chunk, max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model_bart.generate(tokens['input_ids'], max_length=210, min_length=50, length_penalty=2.0, num_beams=3, early_stopping=True)
            summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
            summary_pieces.append(summary)

        summary = " ".join(summary_pieces)
        summary = grammar_check(summary)  
        return summary.lower()  
    except Exception as e:
        print(f"Error in summarize_transcript: {str(e)}")
        return None
 
def detect_language(text):
    translator = Translator()
    detected_lang = translator.detect(text).lang
    if detected_lang == 'en':
        return 'en', 'en'
    else:
        return detected_lang, 'en'

def translate_text(text, source_lang):
    translator = Translator()
    try:
        translation = translator.translate(text, src=source_lang, dest="en-US")
        return translation.text
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return text
   
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process_video', methods=['POST'])
def process_video():
    youtube_url = request.form['youtube_url']

    if youtube_url:
        try:
            # Download audio and get transcript
            full_transcript, detected_lang = extract_transcript(youtube_url)

            if full_transcript:
                # Translate to English if the language is not English
                if detected_lang != 'en':
                    translated_transcript = translate_text(full_transcript, detected_lang)
                else:
                    translated_transcript = full_transcript

                # Convert the translated_transcript to lowercase
                translated_transcript = translated_transcript.lower()

                # Summarize the transcript
                summary = summarize_transcript(translated_transcript)

                # Translate summary to English if the language is not English
                detected_lang_summary, dest_lang_summary = detect_language(summary)
                translated_summary = translate_text(summary, detected_lang_summary)

                # Calculate and print the percentage of summary
                if len(translated_transcript) > 0:
                    percentage_summary = (len(summary) / len(translated_transcript)) * 100
                    print(f"\nPercentage of Summary: {percentage_summary:.2f}%")

                # Print the transcription to the terminal
                print("\nTranscription:")
                print(translated_transcript)
                
                print("\n Summarization:")
                print(translated_summary)

                # Return transcription and summary as part of the JSON response
                return jsonify({'transcription': translated_transcript, 'summary': translated_summary})
            else:
                return jsonify({'error': 'Error retrieving transcript or subtitles'})
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the video. Please try again.'})
    else:
        return jsonify({'error': 'Error extracting video ID'})

if __name__ == '__main__':
    app.run(debug=True)

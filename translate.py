
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from nltk.tokenize import sent_tokenize
import os
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from transformers import MarianMTModel, MarianTokenizer, GPTNeoForCausalLM, GPT2Tokenizer
from gtts import gTTS
from pygame import mixer
import nltk
import urllib3
nltk.download('wordnet')
from threading import Thread
from gingerit.gingerit import GingerIt
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import stanza
import spacy
from PIL import Image, ImageTk
from nltk.corpus import wordnet
from nltk.corpus.reader import NOUN, VERB, ADJ, ADV

nltk.download('punkt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m_name = "C:\\Users\\Mahitab.Ayman\\Documents\\Downloads\\comp\\models\\summarymodel"
tokenizer = AutoTokenizer.from_pretrained(m_name)
model = AutoModelWithLMHead.from_pretrained(m_name).to(device)
def get_summary(text, tokenizer, model, device="cpu", num_beams=2):
    if len(text.strip()) < 50:
        return ["Please provide a longer text."]
    text = "summarize: <paragraph> " + " <paragraph> ".join([s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
    text = text.strip().replace("\n", "")
    tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)
    summary_ids = model.generate(
        tokenized_text,
        max_length=512,
        num_beams=num_beams,
        repetition_penalty=1.5, 
        length_penalty=1.0, 
        early_stopping=True
    )
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return [s.strip() for s in output.split("<hl>") if s.strip() != ""]




# Load SpaCy's English model
nlp = spacy.load('en_core_web_sm')
supported_languages = {
"Arabic": "ar",
"English": "en",
}
# Download and save the models
stanza.download("ar")
arabic_nlp = stanza.Pipeline("ar", processors="tokenize,pos")
model_directory = "models"
arabic_to_english_model = "Helsinki-NLP/opus-mt-ar-en"
english_to_arabic_model = "Helsinki-NLP/opus-mt-en-ar"
gpt_model_name = "EleutherAI/gpt-neo-125M"

for model_path in [arabic_to_english_model, english_to_arabic_model]:
    if not os.path.exists(f"{model_directory}/{model_path}"):
        os.makedirs(f"{model_directory}/{model_path}")
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer.save_pretrained(f"{model_directory}/{model_path}")
        model.save_pretrained(f"{model_directory}/{model_path}")

gpt_model_directory = f"{model_directory}/gpt-neo"
if not os.path.exists(gpt_model_directory):
    os.makedirs(gpt_model_directory)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
    gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_name)
    gpt_tokenizer.save_pretrained(gpt_model_directory)
    gpt_model.save_pretrained(gpt_model_directory)

gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_directory)
gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_directory)
from textblob import TextBlob

def detect_language(text):
    if any("\u0600" <= char <= "\u06FF" for char in text):
        return "arabic"
    else:
        return "english"

def get_derived_forms(word):
    derived_forms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            for form in lemma.derivationally_related_forms():
                derived_forms.append(form.name())
    return set(derived_forms)

from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment
import textstat

def get_text_difficulty(text):
    fk_grade = textstat.flesch_kincaid_grade(text)
    return fk_grade
def get_part_of_speech(word, language):
    if language == "english":
        # Use SpaCy for POS tagging
        doc = nlp(word)
        pos = doc[0].pos_

        if pos == "NOUN":
            return "Noun"
        elif pos == "VERB":
            return "Verb"
        elif pos == "ADJ":
            return "Adjective"
        elif pos == "ADV":
            return "Adverb"
        elif pos == "PRON":
            return "Pronoun"
        else:
            return "Other"
def get_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

def get_named_entities(text):
    doc = nlp(text)
    entities = {ent.text: ent.label_ for ent in doc.ents}
    return entities
# Supported languages

import requests

def check_grammar_and_spelling_gingerit(text, language_code):
    if language_code == "english":  
        ginger_parser = GingerIt()
        try:
            parsed_result = ginger_parser.parse(text)
            corrected_text = parsed_result["result"]
            matches = parsed_result["corrections"]
            return corrected_text, matches
        except (requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError):
            # Retry the connection
            parsed_result = ginger_parser.parse(text)
            corrected_text = parsed_result["result"]
            matches = parsed_result["corrections"]
            return corrected_text, matches
    else:
        return "Grammar and spelling checks are only supported for English.", []
def translate_text(text, source_model, target_model):
    tokenizer = MarianTokenizer.from_pretrained(source_model)
    model = MarianMTModel.from_pretrained(target_model)

    input_tokens = tokenizer(text, return_tensors="pt")
    translation_tokens = model.generate(**input_tokens)
    translated_text = tokenizer.decode(translation_tokens[0], skip_special_tokens=True)

    return translated_text
def interpret_fk_grade(fk_grade):
    if fk_grade < 1:
        return "Kindergarten"
    elif fk_grade < 2:
        return "First/Second Grade"
    elif fk_grade < 3:
        return "Third Grade"
    elif fk_grade < 4:
        return "Fourth Grade"
    elif fk_grade < 5:
        return "Fifth Grade"
    elif fk_grade < 6:
        return "Sixth Grade"
    elif fk_grade < 7:
        return "Seventh Grade"
    elif fk_grade < 8:
        return "Eighth Grade"
    elif fk_grade < 9:
        return "Ninth Grade"
    elif fk_grade < 10:
        return "Tenth Grade"
    elif fk_grade < 11:
        return "Eleventh Grade"
    elif fk_grade < 12:
        return "Twelfth grade"
    elif fk_grade < 14:
        return "College student"
    else:
        return "Professor"
def translate(input_text, src_lang, tgt_lang):
    if src_lang == "arabic" and tgt_lang == "english":
        source_model = f"{model_directory}/{arabic_to_english_model}"
        target_model = f"{model_directory}/{arabic_to_english_model}"
    elif src_lang == "english" and tgt_lang == "arabic":
        source_model = f"{model_directory}/{english_to_arabic_model}"
        target_model = f"{model_directory}/{english_to_arabic_model}"
    else:
        raise ValueError("Unsupported language combination")

    translated_text = translate_text(input_text, source_model, target_model)
    return translated_text

import tempfile
import os

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        temp_path = fp.name
    tts.save(f"{temp_path}.mp3")
    mixer.init()
    mixer.music.load(f"{temp_path}.mp3")
    mixer.music.play()

def generate_example_sentence(word):
    prompt = f"Write a sentence using the word '{word}': "
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt")
    max_length = len(prompt) + 25

    output = gpt_model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.8,
    )

    generated_sentence = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_sentence

def on_example_sentence_button_click():
    def example_sentence_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        language = detect_language(input_text)

        if len(input_text.split()) == 1 and language == "english":
            example_sentence = generate_example_sentence(input_text)
            output_textbox.delete("1.0", tk.END)
            output_textbox.insert("1.0", example_sentence)
        else:
            output_textbox.delete("1.0", tk.END)
            output_textbox.insert("1.0", "Example sentences are available only for single English words.")

    example_sentence_thread = Thread(target=example_sentence_worker)
    example_sentence_thread.start()



def on_ner_button_click():
    def ner_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        entities = get_named_entities(input_text)
        entities_text = "\n".join(f"{entity}: {label}" for entity, label in entities.items())
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", entities_text)

    ner_thread = Thread(target=ner_worker)
    ner_thread.start()
def on_difficulty_button_click():
    def difficulty_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        fk_grade = get_text_difficulty(input_text)
        difficulty_level = interpret_fk_grade(fk_grade)
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", f"Text Difficulty: {difficulty_level} (Flesch–Kincaid Grade: {fk_grade})")

    difficulty_thread = Thread(target=difficulty_worker)
    difficulty_thread.start()
def on_sentiment_button_click():
    def sentiment_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        sentiment = get_sentiment(input_text)
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", f"Sentiment: {sentiment}")

    sentiment_thread = Thread(target=sentiment_worker)
    sentiment_thread.start()
def on_pos_button_click():
    def pos_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        language = detect_language(input_text)

        if len(input_text.split()) == 1:
            pos = get_part_of_speech(input_text, language)
            output_textbox.delete("1.0", tk.END)
            output_textbox.insert("1.0", f"Part of Speech: {pos}")
        else:
            output_textbox.delete("1.0", tk.END)
            output_textbox.insert("1.0", "Please enter a single word.")

    pos_thread = Thread(target=pos_worker)
    pos_thread.start()

def on_synonyms_button_click():
    def synonyms_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        language = detect_language(input_text)

        if len(input_text.split()) == 1:
            if language == "english":
                synonyms = get_synonyms(input_text)
                example_sentence =generate_example_sentence(input_text)
                synonyms_text = f"Synonyms: {', '.join(synonyms)}\nExample Sentence: {example_sentence}"
            else:
                synonyms_text = "Synonyms and example sentences are available only for single English words."
        else:
            synonyms_text = "Synonyms and example sentences are available only for single English words."

        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", synonyms_text)

    synonyms_thread = Thread(target=synonyms_worker)
    synonyms_thread.start()

def on_check_grammar_button_click():
    def check_grammar_worker():
      input_text = input_textbox.get("1.0", "end-1c").strip()
      lang_code = detect_language(input_text)

      if lang_code == "english":
             corrected_text, matches = check_grammar_and_spelling_gingerit(input_text, lang_code)
             output_textbox.delete("1.0", tk.END)
             output_textbox.insert("1.0", corrected_text)

        # You can process 'matches' variable to display specific information about the corrections

      else:
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", "Grammar and spelling checks are only supported for English.")

    check_grammar_thread = Thread(target=check_grammar_worker)
    check_grammar_thread.start()

def on_derived_forms_button_click():
    def derived_forms_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        language = detect_language(input_text)

        if len(input_text.split()) == 1:
            if language == "english":
                derived_forms = get_derived_forms(input_text)
                derived_forms_text = f"Derived Forms: {', '.join(derived_forms)}"
            else:
                derived_forms_text = "Derived forms are available only for single English words."
        else:
            derived_forms_text = "Derived forms are available only for single English words."

        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", derived_forms_text)

    derived_forms_thread = Thread(target=derived_forms_worker)
    derived_forms_thread.start()

def on_translate_button_click():
    def translate_worker():
        input_text = input_textbox.get("1.0", "end-1c").strip()
        src_lang = detect_language(input_text)
        # Set target language based on source language
        tgt_lang = 'english' if src_lang == 'arabic' else 'arabic'

        if src_lang:
            translated_text = translate(input_text, src_lang, tgt_lang)
        else:
            translated_text = "Please enter text to translate."

        output_textbox.delete("1.0", tk.END)
        output_textbox.insert("1.0", translated_text)

    translate_thread = Thread(target=translate_worker)
    translate_thread.start()

def on_listen_button_click():
    def listen_worker():
        output_text = output_textbox.get("1.0", "end-1c").strip()
        language = detect_language(output_text)

        if language == "english":
            text_to_speech(output_text, "en")
        elif language == "arabic":
            text_to_speech(output_text, "ar")
        else:
            output_textbox.delete("1.0", tk.END)
            output_textbox.insert("1.0", "Unsupported language for text-to-speech.")

    listen_thread = Thread(target=listen_worker)
    listen_thread.start()

def on_summarize_button_click():
    input_text = input_textbox.get("1.0", "end-1c")
    summarized_text = get_summary(input_text, tokenizer, model, device)
    output_textbox.delete("1.0", tk.END)
    output_textbox.insert("1.0", "\n".join(summarized_text))

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

root = ThemedTk(theme="clearlooks")
root.title("Language Buddy")
root.geometry("800x600")

# Update colors for elements
bg_color = "#FFFF00"  # A sunny yellow for the background
button_bg = "#FF6B6B"  # Bright red for the button background
button_fg = "#333A44"  # White color for the button text for contrast
label_bg = "#F3B9DF"  # Same as the background color for the label background
label_fg = "#333A44"  # Dark blue color for the label text for readability
text_bg = "#D4BBDD"  # White color for the text field background
text_fg = "#333A44"  # A bright blue color for the text
canvas = tk.Canvas(root)
canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a custom style for the frame
style = ttk.Style()
style.configure("Custom.TFrame", background=bg_color)

frame = ttk.Frame(root, padding="10", style="Custom.TFrame")
frame_id = canvas.create_window(0, 0, window=frame, anchor='nw')

font_style = ('Helvetica', 14)

input_label = ttk.Label(frame, text="Your Words:", font=font_style, background=label_bg, foreground=label_fg)
input_label.grid(row=0, column=0, sticky="w")
input_textbox = tk.Text(frame, wrap="word", width=40, height=10, font=font_style, bg=text_bg, fg=text_fg)
input_textbox.grid(row=1, column=0)

output_label = ttk.Label(frame, text="Language Buddy Says:", font=font_style, background=label_bg, foreground=label_fg)
output_label.grid(row=0, column=1, sticky="w")
output_textbox = tk.Text(frame, wrap="word", width=40, height=10, font=font_style, bg=text_bg, fg=text_fg)
output_textbox.grid(row=1, column=1)

# Update the 'Custom.TButton' style with the new colors
style.configure("Custom.TButton", background=button_bg, foreground=button_fg)

sentiment_button = ttk.Button(frame, text="Analyze Sentiment", command=on_sentiment_button_click, style="Custom.TButton")
sentiment_button.grid(row=6, column=1, pady=5)
synonyms_button = ttk.Button(frame, text="Word Friends", command=on_synonyms_button_click, style="Custom.TButton")
synonyms_button.grid(row=2, column=0, pady=5)
# Create a new button for Text Difficulty
difficulty_button = ttk.Button(frame, text="Text Difficulty", command=on_difficulty_button_click, style="Custom.TButton")
difficulty_button.grid(row=5, column=1, pady=5)
example_sentence_button = ttk.Button(frame, text="Make a Sentence", command=on_example_sentence_button_click, style="Custom.TButton")
example_sentence_button.grid(row=2, column=1, pady=5)

pos_button = ttk.Button(frame, text="Word Type", command=on_pos_button_click, style="Custom.TButton")
pos_button.grid(row=3, column=0, pady=5)

check_grammar_button = ttk.Button(frame, text="Grammar Check", command= on_check_grammar_button_click, style="Custom.TButton")
check_grammar_button.grid(row=3, column=1, pady=5)

translate_button = ttk.Button(frame, text="Translate", command=on_translate_button_click, style="Custom.TButton")
translate_button.grid(row=4, column=0, pady=5)

listen_button = ttk.Button(frame, text="Listen", command=on_listen_button_click, style="Custom.TButton")
listen_button.grid(row=4, column=1, pady=5)
ner_button = ttk.Button(frame, text="Named Entity Recognition", command=on_ner_button_click, style="Custom.TButton")
ner_button.grid(row=7, column=1, pady=5)

# Add a new button for finding derived forms
derived_forms_button = ttk.Button(frame, text="Derived Forms", command=on_derived_forms_button_click, style="Custom.TButton")
derived_forms_button.grid(row=5, column=0, pady=5)

summarize_button = ttk.Button(frame, text="Summarize", command=on_summarize_button_click)
summarize_button.grid(row=6, column=0, pady=5)

def on_canvas_configure(event):
    canvas.itemconfig(frame_id, width=event.width, height=event.height)

canvas.bind('<Configure>', on_canvas_configure)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
canvas.columnconfigure(0, weight=1)
canvas.rowconfigure(0, weight=1)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
frame.rowconfigure(1, weight=1)

root.mainloop()
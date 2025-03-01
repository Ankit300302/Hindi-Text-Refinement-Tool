from flask import Flask, request, render_template
from huggingface_hub import login
from googletrans import Translator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the Flask app
app = Flask(__name__)

# Hugging Face login (you can also manage this through environment variables)
# login(token= "add your token")

# Initialize the models and tokenizers
translator = Translator()

# Load paraphrasing model
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load grammar correction model
corrector_model = "prithivida/grammar_error_correcter_v1"
tokenizer_gc = AutoTokenizer.from_pretrained(corrector_model)
model_gc = AutoModelForSeq2SeqLM.from_pretrained(corrector_model)

# Initialize sentence transformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Light model for embedding

def translate_to_english(text):
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text

def paraphrase(text):
    inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

def grammar_correction(text):
    inputs = tokenizer_gc.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_gc.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer_gc.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def translate_to_hindi(text):
    translation = translator.translate(text, src='en', dest='hi')
    return translation.text

def process_hindi_paragraph(text):
    # Step 1: Translate Hindi to English
    english_text = translate_to_english(text)
    # Step 2: Paraphrase the English text
    paraphrased_text = paraphrase(english_text)
    # Step 3: Correct the Grammar of the Paraphrased text
    corrected_text = grammar_correction(paraphrased_text)
    # Step 4: Translate back to Hindi
    final_hindi_text = translate_to_hindi(corrected_text)
    return final_hindi_text

def validate_output(original_hindi_text, processed_hindi_text):
    # Translate both texts to English for comparison
    original_translation = translate_to_english(original_hindi_text)
    processed_translation = translate_to_english(processed_hindi_text)
    
    # Obtain embeddings for the translations
    original_embedding = sentence_model.encode([original_translation])
    processed_embedding = sentence_model.encode([processed_translation])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(original_embedding, processed_embedding)[0][0]
    
    return original_translation, processed_translation, similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get text from form
    hindi_text = request.form['hindi_text']
    
    # Process the Hindi paragraph
    processed_hindi_text = process_hindi_paragraph(hindi_text)
    
    # Validate output and get similarity score
    original_translation, processed_translation, similarity_score = validate_output(hindi_text, processed_hindi_text)
    
    # Calculate word counts
    original_word_count = len(hindi_text.split())
    processed_word_count = len(processed_hindi_text.split())
    
    # Return the result along with the similarity score and word counts
    return render_template('index.html', 
                           input_text=hindi_text, 
                           output_text=processed_hindi_text, 
                           similarity_score=similarity_score, 
                           original_word_count=original_word_count, 
                           processed_word_count=processed_word_count)


if __name__ == '__main__':
    app.run(debug=True)

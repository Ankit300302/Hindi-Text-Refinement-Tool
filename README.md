# Hindi-Text-Refinement-Tool

Hindi Text Refinement Tool (NLP-based Web Application)
Overview:

The Hindi Text Refinement Tool is a Flask-based web application that enhances Hindi text using state-of-the-art Natural Language Processing (NLP) techniques. The tool performs a multi-step pipeline that translates Hindi input to English, paraphrases and grammatically corrects it, and then translates it back to Hindi — preserving the meaning while improving clarity and correctness. Additionally, it computes a semantic similarity score to ensure the refined output stays faithful to the original meaning.

Key Features:

🔁 Bi-directional Translation: Uses Google Translate to convert text between Hindi and English for NLP model compatibility.

✍️ Paraphrasing Engine: Implements the Vamsi/T5_Paraphrase_Paws model from Hugging Face to generate alternate, fluent sentence structures.

✅ Grammar Correction: Utilizes prithivida/grammar_error_correcter_v1 to correct grammatical issues in the translated English text.

🔄 Back Translation to Hindi: Ensures the final output is returned in Hindi, preserving the refined meaning.

📊 Semantic Similarity Validation: Employs Sentence-BERT (paraphrase-MiniLM-L6-v2) to compute the cosine similarity between the original and final output, providing confidence in content preservation.

🔍 Insightful Metrics: Displays original vs. refined word count and similarity scores to help users understand the extent of changes.

Technologies Used:

Python | Flask – For backend development and web integration

Hugging Face Transformers – For paraphrasing and grammar correction models

Googletrans – For high-quality machine translation between Hindi and English

Sentence Transformers – For calculating semantic similarity

HTML/Jinja2 – For front-end rendering of input/output and results

Use Cases:

Refinement of Hindi text for professional or academic use

Assisting content creators and students with clearer expression

Pre-processing Hindi text data for downstream NLP tasks

Teaching or aiding low-resource language enhancement


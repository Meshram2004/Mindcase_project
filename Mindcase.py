import fitz  
import requests

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def prepare_question_answer_input(question, context, max_context_length=512):
    if len(context) > max_context_length:
        context = context[:max_context_length]  
    input_data = {
        "question": question,
        "context": context
    }
    return input_data

def query_huggingface_api(input_data, api_key, model_name, confidence_threshold=0.5):
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, headers=headers, json=input_data)
    
    if response.status_code == 200:
        output = response.json()
        answer = output.get("answer")
        return answer
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


pdf_file_path = "blade.pdf"
question = input("Enter your question here: ")

pdf_text = extract_text_from_pdf(pdf_file_path)
input_data = prepare_question_answer_input(question, pdf_text)

api_key = "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"
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

answer = query_huggingface_api(input_data, api_key, model_name)

if answer:
    print(f"Answer: {answer}")
else:
    print("Failed to retrieve answer from the Hugging Face API.")
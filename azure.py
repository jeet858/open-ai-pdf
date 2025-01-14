import PyPDF2
import os
from dotenv import load_dotenv
from tqdm import tqdm
import requests

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2023-03-15-preview"  # Replace with your Azure API version if different

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in tqdm(reader.pages, desc="Extracting text"):
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def get_relevant_answer(context, question):
    """Use Azure OpenAI API to get a relevant answer based on the context."""
    try:
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the context:\n{context}\n\nAnswer the following question:\n{question}"}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying Azure OpenAI API: {e}")
        return "Sorry, I couldn't fetch an answer."

def main():
    print("Welcome to the Azure OpenAI-based PDF Document Search Application!")
    pdf_path = './Jeet_Majumder_Resume_python.pdf'

    # Extract text from the PDF
    print("Extracting text from the PDF...\n")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("Failed to extract text from the PDF.")
        return

    print("Text extracted successfully! You can now ask questions.\n")

    # Search functionality
    while True:
        question = input("Ask a question (or type 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            print("Goodbye!")
            break

        print("Searching for the answer...\n")
        answer = get_relevant_answer(pdf_text, question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()

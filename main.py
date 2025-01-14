import PyPDF2
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    """Use OpenAI API to get a relevant answer based on the context."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the context:\n{context}\n\nAnswer the following question:\n{question}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return "Sorry, I couldn't fetch an answer."

def main():
    print("Welcome to the OpenAI-based PDF Document Search Application!")
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

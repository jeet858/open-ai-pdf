import PyPDF2
import os
from dotenv import load_dotenv
from tqdm import tqdm
import requests

#Load environment variables
load_dotenv()

#Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

WORD_LIMIT = 9000
FIRST_TIME_REQUEST = False
def extract_text_from_pdf(pdf_path):
    #Extracting text from pdf
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

def split_text_into_chunks(text, word_limit):
    #Dividing texts with a 29000 words limit
    words = text.split()
    chunks = []
    for i in range(0, len(words), word_limit):
        chunks.append(' '.join(words[i:i + word_limit]))
    return chunks

def get_relevant_answer(context, question):
    #Api call
    try:
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        if FIRST_TIME_REQUEST==False:
            payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the context:\n{context}\n\nAnswer the following question:\n{question} if the there is no answer in the provided context then give this specific reply 'no relevant information'"}
            ]
        
        }
            FIRST_TIME_REQUEST=True
        else:
            payload = {
            "messages": [
                {"role": "system", "content": f"based on the previous contect answer this question:\n{question}?"},
            
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying {e}")
        return "Sorry, I couldn't fetch an answer."

def main():
    print("Welcome to the PDF Document Search Application!")
    pdf_path = './VDSPHelpAgenciesv3.0-1-25.pdf' #replace this path with the pdf you want to upload

    print("Extracting text from the PDF...\n")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("Failed to extract text from the PDF.")
        return

    print("Splitting text into chunks...\n")
    chunks = split_text_into_chunks(pdf_text, WORD_LIMIT)

    print(f"Text split into {len(chunks)} chunks. You can now ask questions.\n")

    while True:
        question = input("Ask a question (or type 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            print("Goodbye!")
            break

        print("Searching for the answer...\n")
        for idx, chunk in enumerate(chunks):
            print(f"Searching in chunk {idx + 1} of {len(chunks)}...\n")
            answer = get_relevant_answer(chunk, question)

            if "no relevant information" not in answer.lower():
                print(f"Answer: {answer}\n")
                break
        else:
            print("The question was irrelevant to the entire PDF.\n")

if __name__ == "__main__":
    main()

import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv


llm = OllamaLLM(
    model="mistral", config={"max_new_tokens": 1000, "temperature": 0.7}
)

def fetch_website_content(urls):
 
    print("Fetching")
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:

            print(f"Error fetching content from {url}: {e}")
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(f"Fetched and split content into {len(split_docs)} chunks.")
    return split_docs

def generate_prompt(question, docs):
    """
    Generate a suitable prompt for the user question and available documents.
    """
    combined_content = "\n\n".join(
        [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )
    
    prompt = f"""
    You are an AI assistant trained to summarize and answer queries based on the latest 
    information from news articles. Analyze the following content and provide an accurate, 
    concise, and context-aware response to the user's question.

    User Question: {question}

    Content:
    {combined_content}

    Please provide the response in a structured format.
    """
    return prompt

def answer_user_question(docs, question):
    prompt = generate_prompt(question, docs)
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    urls = [
        "https://thehimalayantimes.com/",
        "https://kantipurdaily.com/"
    ]

    docs = fetch_website_content(urls)

    print("\nWelcome to the News Query System!")
    print("You can ask questions based on the latest news from Kantipur and The Himalayan Times.")
    print("Type 'exit' to quit the system.\n")

    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() == "exit":
            print("Thank you for using the News Query System. Goodbye!")
            break

        if not docs:
            print("No content available to answer questions. Please try again later.")
            continue

 
        answer = answer_user_question(docs, user_question)
        print(f"\nAnswer:\n{answer}\n")
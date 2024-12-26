import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
import shutil
import os

urls = [
    "https://english.onlinekhabar.com/is-nepal-prepared-for-fire-disasters-a-call-for-action-at-every-level.html",
    "https://english.khabarhub.com/2024/19/420464/",
    "https://myrepublica.nagariknetwork.com/news/dhapasi-fire-brought-under-control-67642f244f2ff.html"
]

def fetch_web_data_with_loader(urls):
    try:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        raw_texts = [doc.page_content for doc in documents]
        print("Fetched and processed web content.")
        return raw_texts
    except Exception as e:
        print(f"Error fetching web data: {e}")
        return None

def split_text(raw_texts, chunk_size=2000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for text in raw_texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def initialize_vectorstore(texts, embedding_model, directory="./chromadb"):
    if os.path.exists(directory):
        shutil.rmtree(directory)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        collection_name="web-documents",
        embedding_function=embeddings,
        persist_directory=directory,
    )

    metadatas = [{"source": f"Document {i+1}"} for i in range(len(texts))]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print("Vectorstore initialized and populated.")
    return vectorstore

def initialize_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
    )

def initialize_qa_chain(retriever, model="mistral", max_tokens=1000, temperature=0.8):
    llm = OllamaLLM(
        model=model, config={"max_new_tokens": max_tokens, "temperature": temperature}
    )

    prompt_template = """
    give me the report of the contents also provide appropriate heading and description of the report. In the next step provide these all in json format.
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )

def ask_question(qa_chain, query, chat_history=None):
    if chat_history is None:
        chat_history = []
    try:
        result = qa_chain({"query": query, "chat_history": chat_history})
        answer = result.get("result", "").strip()
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

# def test_llm():
#     try:
#         llm = OllamaLLM(model="mistral")
#         print("LLM initialized successfully.")

#         prompt_template = """
#         Use the following context to answer the user's question:
#         Context: {context}
#         Question: {question}
#         Answer:
#         """
#         PROMPT = PromptTemplate(
#             template=prompt_template, input_variables=["context", "question"]
#         )

#         context = "This is a test context about Python programming."
#         question = "What is Python?"
#         prompt = PROMPT.format(context=context, question=question)

#         result = llm.invoke(prompt)
#         print("LLM response:", result)
#     except Exception as e:
#         print("Error testing LLM:", e)

def main():
    raw_texts = fetch_web_data_with_loader(urls)
    if not raw_texts:
        print("No data fetched. Exiting.")
        return

    texts = split_text(raw_texts)
    vectorstore = initialize_vectorstore(texts, embedding_model="all-minilm")
    retriever = initialize_retriever(vectorstore)
    qa_chain = initialize_qa_chain(retriever)

    chat_history = []
    while True:
        query = input("Enter your question (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = ask_question(qa_chain, query, chat_history)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

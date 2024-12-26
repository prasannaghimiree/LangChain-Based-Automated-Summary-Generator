import os
import getpass
import json
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_ollama import OllamaLLM

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
)

llm = OllamaLLM(model="mistral", config={"max_new_tokens": 1000, "temperature": 0.9})


def generate_summary(topic, output_file="summary.json"):
    query = f"Fetch detailed and recent information about {topic} with date, time, context, and important features."
    print("1st step")

    search_results = search_tool.invoke({"query": query})
    print("2nd step")

    if not search_results:
        print("No search results found. Please try another topic.")
        return

    combined_content = "\n\n".join(
        [f"Source: {result['url']}\n{result['content']}" for result in search_results]
    )
    print("3rd step")

    prompt = f"""
    Analyze the following content and summarize it. Provide:
    - Headings for each key section.
    - Descriptions with key points, dates, and important features.
    - Ensure the report is structured and concise.

    Content:
    {combined_content}

    Provide the summary in JSON format.
    """

    response = llm.invoke(prompt)
    print("4th step")

    try:
        summary = json.loads(response)
        print("finally")
        print(summary)
    except json.JSONDecodeError:
        print("Error")
        summary = {"topic": topic, "summary": response}


if __name__ == "__main__":
    topic = input("Enter the topic to summarize: ")
    generate_summary(topic)

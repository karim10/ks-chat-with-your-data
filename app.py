from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = OllamaLLM(model="llama3.1")

llm.temperature = 1

messages = [
    SystemMessage(content="You are a helpful assistant in Programming, you will be given a question and you will have to answer it."),
    HumanMessage(content="What is the capital of India?")
]

print(llm.invoke(messages))

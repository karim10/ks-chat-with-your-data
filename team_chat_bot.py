import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

llm = OllamaLLM(model="llama3.1")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm.temperature = 0.8

def load_team_docs():
    pdf_docs = []
    pdf_files = list(Path("profiles").glob("*.pdf"))
    
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()
            team_member = pdf.stem.split('.')[0]
            
            for doc in documents:
                doc.metadata['team_member'] = team_member
                enhanced_content = f"""
                TEAM MEMBER: {team_member}
                
                CONTENT START
                {doc.page_content}
                CONTENT END

                KEY SKILLS AND EXPERIENCE FOR: {team_member}
                """
                doc.page_content = enhanced_content
            
            pdf_docs.extend(documents)
        except Exception as e:
            print(f"Error loading {pdf}: {e}")
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "TEAM MEMBER:",
            "KEY SKILLS AND EXPERIENCE FOR",
        ]
    )
    docs = text_splitter.split_documents(pdf_docs)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vectorstore

vectorstore = load_team_docs()

retriever = vectorstore.as_retriever(
    search_type="mmr",  # MMR tries to find documents that are both relevant to the query AND diverse from each other. This helps avoid redundant information in the results
    search_kwargs={
        'k': 10,  # The number of documents to retrieve
        'fetch_k': 30,  # The number of documents to fetch initially
        'lambda_mult': 0.7  # The diversity factor, Higher values (closer to 1) prioritize relevance over diversity. Lower values prioritize diversity over relevance
    }
)

system_template = """You are an AI assistant specialized in providing information about team members based on their profiles.

IMPORTANT GUIDELINES:
1. Please do not make up information, only use the information provided in the profiles
2. If you cannot find information about a team member, please say so
3. Please do not mention that you are using the profiles to answer the question

Current context: {context}
Chat history: {chat_history}
Human: {question}
"""

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    combine_docs_chain_kwargs={'prompt': ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ])}
)

chat_history = []

def respond(message, history):
    global chat_history

    result = qa_chain({
        "question": message,
        "chat_history": chat_history
    })

    history.append((message, result["answer"]))
    return "", history

def clear_history():
    global chat_history
    chat_history = []
    return None, []


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Frontenders Chat Bot")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Question", placeholder="Ask about your teammates...")
    clear = gr.Button("Clear Conversation")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_history, None, [msg, chatbot])
    
    gr.Examples([
        "What is Karim's current role?",
        "What are Karim's skills?",
        "Which projects has Karim worked on?",
        "What is Karim's educational background?"
    ], inputs=msg)

if __name__ == "__main__":
    demo.launch()
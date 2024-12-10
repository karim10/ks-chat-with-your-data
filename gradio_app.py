import gradio_app as gr
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = OllamaLLM(model="llama3.1")
messages = [
    SystemMessage(content="""You are an expert in Geography...""")
]

def respond(message, chat_history):
    messages.append(HumanMessage(content=message))
    bot_message = llm.invoke(messages)
    messages.append(AIMessage(content=str(bot_message)))
    chat_history.append((message, str(bot_message)))
    return "", chat_history

def clear_history():
    global messages
    messages = [messages[0]]  # Keep only the system message
    return None, []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Geography Expert Bot")
    gr.Markdown("Ask me anything about geography! I specialize in physical geography, climate, and landforms.")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Question", placeholder="Ask about geography...")
    clear = gr.Button("Clear Conversation")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_history, None, [msg, chatbot])
    
    gr.Examples(
        examples=["What are the major oceans?",
                 "Describe the climate zones of Earth",
                 "What is the highest mountain range?"],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch()
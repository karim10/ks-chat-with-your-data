from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def interactive_conversation():
    llm = OllamaLLM(model="llama3.1")

    messages = [
        SystemMessage(content="""You are an expert in Geography.
        You don't answer questions that are not related to Geography.""")
    ]
    
    print("Geography Bot (type 'quit' to exit)")
    print("----------------------------------------")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        
        print("\nGeography Bot:", response)
        messages.append(AIMessage(content=str(response)))
        print('\nMessages:', messages)

if __name__ == "__main__":
    interactive_conversation()

def interactive_conversation_with_context():
    llm = OllamaLLM(model="llama3.1")

    # Define a large context about geography
    geography_context = """
    CONTEXT AND KNOWLEDGE BASE:
    
    1. Physical Geography:
    - Continents: Asia, Africa, North America, South America, Antarctica, Europe, Australia
    - Major mountain ranges: Himalayas, Andes, Rocky Mountains, Alps
    - Oceans: Pacific, Atlantic, Indian, Southern, Arctic
    
    2. Climate Zones:
    - Tropical
    - Temperate
    - Polar
    - Desert
    - Mediterranean
    
    3. Important Geographic Terms:
    - Latitude and Longitude
    - Tectonic Plates
    - Erosion and Weathering
    - River Systems
    - Coastal Features
    
    4. Major Geographic Features:
    - Grand Canyon
    - Great Barrier Reef
    - Amazon Rainforest
    - Sahara Desert
    - Dead Sea
    
    CONSTRAINTS:
    - Only answer questions related to geography
    - Use metric system for measurements
    - Provide specific examples when possible
    - Cite major geographic features when relevant
    """

    messages = [
        SystemMessage(content=f"""You are an expert in Geography.
        Use the following context for your responses:
        {geography_context}
        
        You must:
        1. Only answer geography-related questions
        2. Reference the provided context when possible
        3. Clearly state if a question is outside your geography expertise
        4. Provide factual, context-based responses""")
    ]
    
    print("Geography Bot (type 'quit' to exit)")
    print("----------------------------------------")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        
        print("\nGeography Bot:", response)
        messages.append(AIMessage(content=str(response)))

if __name__ == "__main__":
    interactive_conversation()

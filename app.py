import gradio as gr

from main import qa_chain

# Step 1: Define the ask function to handle user queries
def ask(question, history):
    return qa_chain.run(question)

# Step 2: Create a Gradio chat interface and launch it 
demo = gr.ChatInterface(fn=ask, type="messages", title="Recipe Bot")
demo.launch()

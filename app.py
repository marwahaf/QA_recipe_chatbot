import gradio as gr

from main import qa_chain

def ask(question):
    return qa_chain.run(question)

demo = gr.ChatInterface(fn=ask, title="Recipe Bot")

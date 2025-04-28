import gradio as gr

from main import qa_chain


def ask(question, history):
    return qa_chain.run(question)


demo = gr.ChatInterface(fn=ask, type="messages", title="Recipe Bot")
demo.launch()

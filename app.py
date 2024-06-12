"""app.py"""

import io
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from mistralai.models.chat_completion import ChatMessage

from src.utilities import Conversation # type: ignore

def rendar_ui():
    st.set_page_config(
        page_title="Mistral AI_ (Mistral 7B)",
        page_icon="🛠️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.header("Talk to PDF files 📰 using Mistral AI_", divider="rainbow")
    st.subheader(
        "Enjoy :red[talking] with :green[PDF] files using :sunglasses: Mistral 7B"
    )

    st.sidebar.title("Talk to PDF 📰")
    st.sidebar.markdown(
        "[Checkout the repository](https://github.com/ThivaV/chat_with_pdf_using_mistral)"
    )
    st.sidebar.markdown(
        """
            ### This is a LLM powered chatbot, built using:
                
            * [Streamlit](https://streamlit.io)
            * [Mistral AI_](https://mistral.ai)
            * [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/)
            * [FAISS](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)
            ___
            """
    )

    add_vertical_space(2)

    mistral_api_key = st.sidebar.text_input("Enter your Mistral AI_ API key 👇", type="password")
    if not mistral_api_key:
        st.info("👈 :red[Please enter your Mistral AI api key and press enter ↲] ⛔")
        st.stop()
    else:
        st.session_state.API_KEY = mistral_api_key
        uploaded_pdfs = st.sidebar.file_uploader(
            "Upload a pdf files 📤", type="pdf", accept_multiple_files=True
        )

        if not uploaded_pdfs:
            st.info("👈 :red[Please upload pdf files] ⛔")
            st.stop()
        else:
            for uploaded_pdf in uploaded_pdfs:
                bytes_io = io.BytesIO(uploaded_pdf.getvalue())
                st.session_state.PDF_BYTES.append(bytes_io)
        

if __name__ == "__main__":

    # initialize streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.API_KEY = None
        st.session_state.PDF_BYTES = []

    # rendar ui
    rendar_ui()

    conversation = Conversation(
        st.session_state.API_KEY,
        st.session_state.PDF_BYTES,
    )

    # initialize
    conversation.initialize_retriever()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Say something"):
        # user message
        with st.chat_message("user"):
            st.write(query)
            st.session_state.messages.append({"role": "user", "content": query})

        context = conversation.search(query)

        message = [ChatMessage(role="user", content= query)]
        mistral_response = conversation.talk_to_mistral_ai(message)
        response = mistral_response.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
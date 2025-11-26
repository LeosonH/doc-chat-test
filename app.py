import os
import tempfile
import streamlit as st
from embedchain import App
import json



def embedchain_bot(db_path, api_key):
    return App.from_config(
        config={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o",
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "stream": True,
                    "api_key": api_key,
                    "system_prompt": """You are a helpful assistant that answers questions about documents. Follow these rules strictly:

1. When answering questions about document content, ONLY use information explicitly stated in the provided context from the documents.
2. If the answer to a question is not found in the documents, respond with: "I don't have that information in the uploaded documents."
3. Do not make assumptions, inferences, or add information beyond what is explicitly stated in the documents.
4. If asked a general question unrelated to the documents (like "what is the weather?"), you may answer normally, but make it clear you're not referencing the documents.
5. When referencing information from documents, be specific and accurate. Do not paraphrase in ways that could change the meaning.
6. If you're unsure whether information is in the documents, err on the side of saying you don't have that information rather than guessing.

Your goal is accuracy and trustworthiness, not comprehensiveness.""",
                },
            },
            "vectordb": {
                "provider": "chroma",
                "config": {"collection_name": "chat-pdf", "dir": db_path, "allow_reset": True},
            },
            "embedder": {"provider": "openai", "config": {"api_key": api_key}},
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )
def get_db_path():
    db_path = os.path.join(os.getcwd(), "knowledge_base")
    os.makedirs(db_path, exist_ok=True)
    return db_path

def get_files_list_path():
    return os.path.join(get_db_path(), "added_files.json")

def load_added_files():
    """Load the list of files that have been added to the knowledge base."""
    files_list_path = get_files_list_path()
    if os.path.exists(files_list_path):
        with open(files_list_path, "r") as f:
            return json.load(f)
    return []

def save_added_files(files_list):
    """Save the list of files that have been added to the knowledge base."""
    files_list_path = get_files_list_path()
    with open(files_list_path, "w") as f:
        json.dump(files_list, f)

def get_ec_app(api_key):
    if "app" in st.session_state:
        print("Found app in session state")
        app = st.session_state.app
    else:
        print("Creating app")
        db_path = get_db_path()
        app = embedchain_bot(db_path, api_key)
        st.session_state.app = app
    return app
with st.sidebar:
    openai_access_token = st.text_input("OpenAI API Key", key="api_key", type="password")
    "WE DO NOT STORE YOUR OPENAI KEY."
    "Just paste your OpenAI API key here and we'll use it to power the chatbot. [Get your OpenAI API key](https://platform.openai.com/api-keys)"  # noqa: E501

    if st.session_state.api_key:
        app = get_ec_app(st.session_state.api_key)

    # Show existing files in knowledge base
    st.markdown("---")
    st.markdown("### Knowledge Base")
    added_files_persistent = load_added_files()
    if added_files_persistent:
        st.markdown(f"**Files in knowledge base ({len(added_files_persistent)}):**")
        for file in added_files_persistent:
            st.markdown(f"- {file}")
        st.info("💡 To clear the knowledge base, delete the `knowledge_base` folder and restart the app.")
    else:
        st.markdown("*No files added yet*")

    st.markdown("---")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "docx"])
    add_files = load_added_files()
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name in add_files:
            continue
        try:
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API Key")
                st.stop()
            temp_file_name = None
            file_extension = os.path.splitext(file_name)[1].lower()

            with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name, suffix=file_extension) as f:
                f.write(uploaded_file.getvalue())
                temp_file_name = f.name
            if temp_file_name:
                st.markdown(f"Adding {file_name} to knowledge base...")

                # Determine data type based on file extension
                if file_extension == ".pdf":
                    data_type = "pdf_file"
                elif file_extension == ".txt":
                    data_type = "text_file"
                elif file_extension == ".docx":
                    data_type = "docx"
                else:
                    data_type = "text_file"

                app.add(temp_file_name, data_type=data_type)
                st.markdown("")
                add_files.append(file_name)
                save_added_files(add_files)
                os.remove(temp_file_name)
            st.session_state.messages.append({"role": "assistant", "content": f"Added {file_name} to knowledge base!"})
        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()

st.title("📄docGPT")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm docGPT. I can answer questions about your documents.\n
                Upload your documents (PDF, TXT, or DOCX) here and I'll answer your questions about them!
            """,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    if not st.session_state.api_key:
        st.error("Please enter your OpenAI API Key", icon="🤖")
        st.stop()

    app = get_ec_app(st.session_state.api_key)

    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")
        full_response = ""

        for response in app.chat(prompt):
            msg_placeholder.empty()
            full_response += response

        st.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

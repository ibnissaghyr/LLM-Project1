from langchain_community.llms.openai import OpenAI
import streamlit as st
import pickle  # For serialization
import time
from langchain_community.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-SJSWAgiZbYpd8mFwtg3pk9LqQqzhR_nz1oGdVtBFhZJbYombuFzNhsIN5vfEU4T3BlbkFJkkeHpynCBS1HjfUq3oWmtmm3fP_NL-Ixz_juq0Blc6PJ8vNM2RAnLJ8yuABduUXX2GVluTExwA"


st.title("InfoScout: News Research Assistant 📰")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
def set_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
                body {
                    background-color: #1E1E1E;
                    color: white;
                }
                .css-1v3fvcr {  /* Sidebar background color */
                    background-color: #333333;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                body {
                    background-color: white;
                    color: black;
                }
                .css-1v3fvcr {  /* Sidebar background color */
                    background-color: #f4f4f4;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Apply the selected theme
set_theme(theme)
st.sidebar.title("Input News Article URLs")





# Input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    urls.append(url)

# Process URLs Button
process_url_clicked = st.sidebar.button("Process URLs")
vectorstore_path = "vectorstore_openai"

# Placeholder for main messages
main_placeholder = st.empty()

# Initialize LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading data from URLs... ✅")
        data = loader.load()

        # Split data into manageable chunks
        main_placeholder.text("Splitting text into chunks... ✅")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # Generate embeddings and build the FAISS vector store
        main_placeholder.text("Building embeddings and vector store... ✅")
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index to a file for later use
        vectorstore_openai.save_local(vectorstore_path)
        main_placeholder.text("Process completed! ✅ Ready for queries.")
        time.sleep(2)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# Input for Questions
query = main_placeholder.text_input("Enter your question:")
if query:
    try:
        # Load FAISS index with deserialization enabled
        vectorstore = FAISS.load_local(
            vectorstore_path,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

        # Build a QA chain using the retriever
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        # Get the result from the chain
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    except Exception as e:
        st.error(f"An error occurred while processing the query: {e}")
feedback = st.text_area("Your Feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")


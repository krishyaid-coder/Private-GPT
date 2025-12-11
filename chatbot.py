import streamlit as st
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA 
from langchain.llms import HuggingFaceEndpoint 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import os

HF_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', '')

# defining the llm which i am using
def get_llm():
    llm = HuggingFaceEndpoint(
        endpoint_url="HuggingFaceH4/zephyr-7b-beta",
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# to get the response from the asked question
def get_response():
    # initialising the db connection
    # this is my embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    db = Chroma(persist_directory='db', embedding_function = embeddings)

    # now we need to create a retriever object inorder to get the documents
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    # calling the llm function
    llm = get_llm()

    # creating my own custom template
    custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an Data Science assistant and you only answer to any query realted to Data Science and the 
    documents present in the database. Use the following context to answer the question. If you don't know 
    the answer tell that this is out of my knowledge. Also, whenever the user logs in greet him with a warm welcome
    message!!
    
    Context:
    {context}

    Question:
    {question}
    """
    )

    #get the answer to the asked question
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = retriever,
        return_source_documents=True,
        chain_type_kwargs = {"prompt": custom_prompt}
    )
    return qa

def process_answer(user_instruction):
    qa = get_response()
    print('this is my response ########################\n ',qa)
    generated_text = qa(user_instruction)
    parsed_text = generated_text['result']
    return parsed_text

# to display the conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

    
def main():
    st.title('Chat with Your Data 🦜📄')
    with st.expander("About the Chatbot"):
        st.markdown(
            """
            This is a Generative AI powered Chatbot that interacts with you and you can ask followup questions.
            """
        )
   
    # initialise a new session
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['']
    if 'past' not in st.session_state:
        st.session_state['past'] = ['']
    
    footer = st.container()

    with footer:
        st.markdown("---")
        st.subheader("Chat with us:")
        user_input = st.text_input('Please type your query here...', key = 'input')
        # Search the database for a response based on user input and update session state
        if user_input:
            answer = process_answer({'query': user_input})
            st.session_state["past"].append(user_input)
            response = answer
            st.session_state["generated"].append(response)
        
        # Display conversation history using Streamlit messages
        if st.session_state["generated"]:
            display_conversation(st.session_state)

if __name__ == '__main__':
    main()
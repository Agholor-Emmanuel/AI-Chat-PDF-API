from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub, LlamaCpp
from langchain_community.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_community.llms import CTransformers, LlamaCpp
from transformers import pipeline
from dotenv import load_dotenv
# from llama_cpp import Llama


def load_mistral_pipeline():
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = LlamaCpp(
        model_path= "./Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0,
        top_p=3,
        n_ctx=4096,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        streaming=True,
        verbose=True,
        n_gpu_layers=0
    )
    return llm



def chatbot(vector_database, model): 
    load_dotenv()
    llm_models = {
        'Mistral7B': load_mistral_pipeline()
    }
    llm = llm_models.get(model)
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", 
                                                               retriever=vector_database.as_retriever(), 
                                                               memory=memory
                                                               )
    return conversation_chain





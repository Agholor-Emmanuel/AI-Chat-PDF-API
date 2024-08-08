from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from dotenv import load_dotenv



def get_vector_database(text_chunks, model):
    load_dotenv()
    model_to_embeddings = {
        'OpenAI': OpenAIEmbeddings(),
        'HuggingFace': HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}),
        'LaMini': SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2'),
        'Mixtral23B': HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}),
        'CodeLlama70B': HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}),
        'Mistral7B' : HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    }
    embeddings = model_to_embeddings.get(model)
    vector_database = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_database





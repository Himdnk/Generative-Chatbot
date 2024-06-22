from flask import Flask, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)
docs = []

def process_input(urls, question):
    model_local = Ollama(model="mistral")
    
    urls_list = urls.split("\n")
    for url in urls_list:
        loader = YoutubeLoader.from_youtube_url(url) 
        docs1 = loader.load()
        for doc in docs1:
            docs.append(doc)
    
    if not docs:
        return "No documents loaded from the provided URLs.", 400

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question), 200

@app.route('/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    urls = data.get('urls')
    question = data.get('question')
    
    if not urls or not question:
        return jsonify({"error": "Both 'urls' and 'question' fields are required."}), 400

    answer, status_code = process_input(urls, question)
    return jsonify({"answer": answer}), status_code

if __name__ == '__main__':
    app.run(debug=True)

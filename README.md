# Arcitech Assignment

Generative Chatbot using Retrieval-Augmented Generation (RAG)
This project implements a generative chatbot using Retrieval-Augmented Generation (RAG). The chatbot leverages YouTube links to transcribe videos and uses a large language model (LLM) to perform question-and-answer tasks based on a knowledge base. The project has been run locally using Ollama, which incorporates the Mistral LLM and Nomic-Embed-Text for embeddings. There are two available interfaces: Flask and Streamlit.

This project uses various libraries to create a generative chatbot with Retrieval-Augmented Generation (RAG). It extracts transcripts from YouTube videos using YoutubeLoader, splits these transcripts into manageable chunks with CharacterTextSplitter, and converts the chunks into embeddings via OllamaEmbeddings. These embeddings are stored in a Chroma vector database for retrieval. A prompt template then formats the retrieved context and user question for the Mistral LLM, which generates an answer. The project includes a Streamlit interface where users can input YouTube URLs and questions, triggering the processing function to display the answer. Additionally, the same task can be accomplished using the OpenAI API for generating responses.




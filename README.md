# RAG-over-Audio-Data - Audio Transcript Processing and QA System

## Introduction 
This repository orchestrates a sophisticated pipeline for text processing, leveraging various libraries and modules. It begins by transcribing remote audio files, segmenting the text into manageable chunks, and embedding these chunks into a vector database for efficient retrieval. The transcription is facilitated using the AssemblyAI service, allowing for easy access to the content of the audio files.

The **HuggingFaceEmbeddings** module is employed to generate text embeddings using the HuggingFace models, enabling the conversion of text into numerical representations. These embeddings are then used to construct a vector database via Chroma, optimizing the storage and retrieval of text chunks.

The **ChatOpenAI** model, a heightened iteration of GPT-3, is utilized in a question-answering setup (RetrievalQA). This allows users to input questions, which are processed by the model to retrieve relevant answers from the text database created earlier. The retrieved answers are then presented along with the source documents containing relevant content, aiding transparency and context.

Overall, this code amalgamates audio transcription, text segmentation, text embedding, vector database creation, and advanced question-answering capabilities, providing a robust framework for handling text-based queries and interactions in an automated setup.

![image](https://github.com/wittyicon29/RAG-over-Audio-Data/assets/99320225/005da508-f864-44cf-801d-79346a7cbda0)


## Components

1. **Audio Transcription**
The code utilizes the AssemblyAIAudioTranscriptLoader to transcribe audio files from remote URLs into text.

2. **Text Splitting**
The transcribed texts are split into smaller chunks using RecursiveCharacterTextSplitter to facilitate processing and analysis.

3. **Text Embedding**
HuggingFace embeddings (HuggingFaceEmbeddings) are used to convert text chunks into embeddings. This allows for better semantic understanding and similarity calculations.

4. **Vector Database Creation**
The embedded text chunks are stored in a vector database (Chroma) for efficient retrieval and comparison.

5. **Question-Answering (QA) System**
The code implements a QA system using an advanced version of the GPT-3 model (ChatOpenAI). It retrieves relevant information from the vector database based on user queries.

## Installation

**Clone the repo**
```cd
git clone https://github.com/wittyicon29/RAG-over-Audio-Data.git
```

**Switch to the directory**
```cd
cd RAG-over-Audio-Data
```

**Install the dependencies**
```cd
pip install -r requirements.txt
```

## Usage

Set up your environment variables in a **.env** file.

**Transcribing Files**: Provide remote audio file URLs in the URLs list. The provided URLs of the audio files must be publicily accessible so that AssemmbyAI API can access those audio files.

**Run the script**
```python
python main.py
```

## Configuration

Adjust the chunk size and overlap in RecursiveCharacterTextSplitter for text splitting customization.

Modify model names and parameters in the make_embedder() and make_qa_chain() functions to experiment with different language models and settings.

## References 

[Retrieval Augumented Generation for Audio using Langchain](https://www.assemblyai.com/blog/retrieval-augmented-generation-audio-langchain)

## LICENSE

This project is licensed under the MIT License - see the LICENSE file for details

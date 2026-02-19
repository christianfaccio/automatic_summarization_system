from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma 
import chromadb

import json 
import os

docs_path = '../docs/'
output_path = '../output/'

loader = PyPDFDirectoryLoader(docs_path)
docs = loader.load() # returns a list of `document` objects, which refer to the pages of the docs

# -- SPLITTING -- 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 200, add_start_index = True 
)

chunks = splitter.split_documents(docs) # list of documents (just smaller size -> chunks)
chunks = [c for c in chunks if "Código seguro de Verificación" not in c.page_content]

# -- VECTOR STORE --
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
client = chromadb.EphemeralClient()
try:
    client.delete_collection('langchain') # fresh restart
except Exception:
    pass
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client)

# -- OBTAINING INFO AND SAVING TO JSON --
queries = [
    "beneficiarios y destinatarios de las ayudas",
    "requisitos y condiciones para solicitar la ayuda",
    "cuantía e importe de las ayudas",
    "plazo y periodo de solicitud",
    "documentación necesaria para la solicitud",
    "criterios de selección y baremación",
    "modalidades y tipos de ayuda",
    "obligaciones de los beneficiarios",
    "incompatibilidades con otras ayudas",
    "procedimiento de resolución y concesión",
    "forma de pago y abono de las ayudas",
    "causas de denegación o revocación",
]

sources = os.listdir(docs_path)

output = {}
for query in queries:
    output[query] = {}
    for source in sources:
        results = vector_store.similarity_search(
            query=query,
            k=3,
            filter={'source': f'../docs/{source}'}
        )
        output[query][source] = [
            {
                'content': doc.page_content,
                'page': doc.metadata['page']
            } for doc in results
        ]

with open(output_path + 'info.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

import os
import re
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from model import tokenizer, model


# -----------------------------
# Load PDF Documents
# -----------------------------

documents = []

pdf_files = [
    "data/DesFlyer_Chatbot_QA.pdf",
    "data/Research & Development.pdf",
    "data/Chatbot dataset.pdf"
]

for pdf in pdf_files:

    if os.path.exists(pdf):
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    else:
        print(f"{pdf} not found")

print(f"Loaded {len(documents)} documents.")


# -----------------------------
# Clean Text
# -----------------------------

for doc in documents:
    text = doc.page_content
    text = re.sub(r"\s+", " ", text)
    doc.page_content = text.strip()

print("Documents cleaned successfully.")


# -----------------------------
# Split Documents
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")


# -----------------------------
# Create Embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# -----------------------------
# Create / Load Vector Database
# -----------------------------

VECTOR_DB_PATH = "vector_db"


if not os.path.exists(VECTOR_DB_PATH):

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )

    print("Vector database created successfully!")

else:

    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model
    )

    print("Existing vector database loaded.")


# -----------------------------
# Create Retriever
# -----------------------------

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Retriever created successfully.")


# -----------------------------
# Prompt Template
# -----------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""

You are the official DesFlyer chatbot.

Answer ONLY using the given context.

Rules:
1. Answer only DesFlyer-related questions.
2. Do not create information.
3. Do not use outside knowledge.
4. If answer is not available, reply:
"I'm sorry, I can only answer questions related to DesFlyer."

Context:
{context}

Question:
{question}

Answer:

"""
)


print("Prompt created successfully.")



# -----------------------------
# Greetings
# -----------------------------

greetings = {

    "hi": "Hello! Welcome to DesFlyer. How can I assist you today?",

    "hello": "Hello! Welcome to DesFlyer. How can I assist you today?",

    "hey": "Hi! Welcome to DesFlyer. Feel free to ask me about DesFlyer.",

    "good morning": "Good Morning! Welcome to DesFlyer.",

    "good afternoon": "Good Afternoon! Welcome to DesFlyer.",

    "good evening": "Good Evening! Welcome to DesFlyer.",

    "good night": "Good Night! Thank you for visiting DesFlyer."
}



# -----------------------------
# Keywords Restriction
# -----------------------------

keywords = [

    "desflyer",
    "des flyer",
    "company",
    "software",
    "website",
    "mobile",
    "application",
    "service",
    "services",
    "contact",
    "career",
    "office",
    "project",
    "client",
    "job",
    "internship",
    "location",
    "app"

]


print("Keywords loaded successfully.")



# -----------------------------
# Chatbot Function
# -----------------------------

def ask_chatbot(question):

    original_question = question.strip()

    lower_question = original_question.lower()


    # Greeting

    if lower_question in greetings:

        return greetings[lower_question]



    # Keyword restriction

    if not any(keyword in lower_question for keyword in keywords):

        return "I'm sorry, I can only answer questions related to DesFlyer."



    # Retrieve Documents

    retrieved_docs = retriever.invoke(original_question)
    print("Retrieved documents count:", len(retrieved_docs))

    for i, doc in enumerate(retrieved_docs):
     print("\nDOC", i+1)
    print(doc.page_content)


    print(
        "Retrieved documents count:",
        len(retrieved_docs)
    )



    if not retrieved_docs:

        return "I'm sorry, I could not find this information in DesFlyer documents."



    print("\n===== Retrieved Documents =====")


    for i, doc in enumerate(retrieved_docs, start=1):

        print(f"\nDocument {i}")

        print(doc.page_content[:300])



    # Create Context

    context = "\n".join(

        doc.page_content for doc in retrieved_docs

    )



    # Create Prompt

    final_prompt = prompt.format(

        context=context,

        question=original_question

    )



    # Tokenize

    inputs = tokenizer(

        final_prompt,

        return_tensors="pt",

        truncation=True,

        max_length=2048

    ).to(model.device)



    # Generate Answer

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=150,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id

        )



    # Decode

    answer = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True
)

    print("\n===== Generated Answer =====")
    print(answer)

    return answer.strip()
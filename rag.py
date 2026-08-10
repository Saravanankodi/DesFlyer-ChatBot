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
# Vector Database Path
# -----------------------------

VECTOR_DB_PATH = "vector_db"


# -----------------------------
# Create / Load Vector Database
# -----------------------------

if not os.path.exists(VECTOR_DB_PATH):

    print("Vector database not found.")
    print("Creating vector database for the first time...")

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
        chunk_size=300,
        chunk_overlap=50
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
    # Create Vector Database
    # -----------------------------

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )

    print("Vector database created successfully!")


else:

    print("Existing vector database found.")
    print("Loading existing vector database...")


    # -----------------------------
    # Load Embedding Model
    # -----------------------------

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded successfully.")


    # -----------------------------
    # Load Existing Vector Database
    # -----------------------------

    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model
    )

    print("Stored documents:", vector_db._collection.count())

    print("Existing vector database loaded successfully!")


# -----------------------------
# Create Retriever
# -----------------------------

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

print("Retriever created successfully.")


# -----------------------------
# Retrieval Test
# -----------------------------

test_questions = [
    "What services does DesFlyer offer?",
    "Does DesFlyer develop websites?",
    "Does DesFlyer develop mobile applications?"
]

for question in test_questions:

    print("\n====================================")
    print("QUESTION:", question)
    print("====================================")

    docs = retriever.invoke(question)

    print("Number of documents retrieved:", len(docs))

    for i, doc in enumerate(docs):

        print(f"\n--- Document {i + 1} ---")

        print(doc.page_content[:1000])


# -----------------------------
# Prompt Template
# -----------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a DesFlyer FAQ chatbot.

Answer the question using ONLY the context below.

Rules:
- Give ONE short, complete answer.
- Summarize the relevant information from the context.
- Combine related information into one natural answer.
- Do NOT copy sentences directly from the context.
- Do NOT repeat the same information.
- Do NOT include question numbers.
- Do NOT mention the context.
- Do NOT add information that is not in the context.
- Do NOT ask a question.
- End with a normal sentence, not a question mark.
- Keep the answer within 1 or 2 sentences.

Context:
{context}

Question:
{question}

Write only the final answer:

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


    # -----------------------------
    # Greeting
    # -----------------------------

    if lower_question in greetings:

        return greetings[lower_question]


    # -----------------------------
    # Keyword Restriction
    # -----------------------------

    if not any(
        keyword in lower_question
        for keyword in keywords
    ):

        return "I'm sorry, I can only answer questions related to DesFlyer."


    # -----------------------------
    # Retrieve Documents
    # -----------------------------

    retrieved_docs = retriever.invoke(
        original_question
    )

    print("\nQUESTION:", original_question)

    print(
        "RETRIEVED DOCUMENTS:",
        len(retrieved_docs)
    )


    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        print(f"\n--- DOCUMENT {i} ---")

        print(
            doc.page_content[:1000]
        )


    if not retrieved_docs:

        return "I'm sorry, I could not find this information in DesFlyer documents."


    # -----------------------------
    # Create Context
    # -----------------------------

    context = "\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # -----------------------------
    # Create Prompt
    # -----------------------------

    final_prompt = prompt.format(
        context=context,
        question=original_question
    )


    # -----------------------------
    # Tokenize
    # -----------------------------

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)


    # -----------------------------
    # Generate Answer
    # -----------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )


    # -----------------------------
    # Decode Answer
    # -----------------------------

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    answer = answer.strip()

    if answer.endswith("?"):
       answer = answer[:-1] + "."


    # -----------------------------
    # Print Generated Answer
    # -----------------------------

    print("\n===== Generated Answer =====")

    print(answer)
    return answer
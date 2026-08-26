import os
import re
import time
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from model import tokenizer, model


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = "vector_db"

# Keep only recent conversation turns
MAX_HISTORY_TURNS = 4

# Retrieve fewer documents to reduce prompt size
RETRIEVER_K = 2

# ============================================================
# GENERATION PERFORMANCE SETTINGS
# ============================================================

# Final answer should be short.
# Lower value = less generation time.
ANSWER_MAX_NEW_TOKENS = 24

# Rewritten question only needs a short sentence.
REWRITE_MAX_NEW_TOKENS = 16

# Maximum amount of retrieved context sent to Gemma.
MAX_CONTEXT_CHARS = 2500

# Maximum history characters sent to rewrite model.
MAX_HISTORY_CHARS = 1500

# Safety limit for generation.
# This prevents very long generation.
ANSWER_MAX_TIME = 25

# Rewrite should be very quick.
REWRITE_MAX_TIME = 10


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

torch.set_grad_enabled(False)

print("\n====================================")
print("Gemma Device Information")
print("====================================")

print(
    "Model device:",
    model.device
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = []


def clear_conversation_history():

    global conversation_history

    conversation_history = []

    print("\n🧹 Conversation history cleared.")


def get_conversation_history():

    return conversation_history.copy()


def format_conversation_history():

    if not conversation_history:

        return "No previous conversation."

    history_text = []

    for turn in conversation_history:

        history_text.append(
            f"User: {turn['user']}"
        )

        history_text.append(
            f"Assistant: {turn['assistant']}"
        )

    history = "\n".join(history_text)

    # Limit history size
    return history[-MAX_HISTORY_CHARS:]


def add_to_conversation_history(
    user_question,
    assistant_answer
):

    global conversation_history

    conversation_history.append({

        "user": user_question,

        "assistant": assistant_answer

    })

    if len(conversation_history) > MAX_HISTORY_TURNS:

        conversation_history = (
            conversation_history[
                -MAX_HISTORY_TURNS:
            ]
        )


# ============================================================
# LOAD / CREATE VECTOR DATABASE
# ============================================================

if not os.path.exists(VECTOR_DB_PATH):

    print("\n====================================")
    print("Creating Vector Database")
    print("====================================")

    documents = []

    pdf_files = [

        "data/DesFlyer_Chatbot_QA.pdf",

        "data/Research & Development.pdf",

        "data/Chatbot dataset.pdf"

    ]

    for pdf in pdf_files:

        if os.path.exists(pdf):

            print(
                f"Loading: {pdf}"
            )

            loader = PyPDFLoader(pdf)

            documents.extend(
                loader.load()
            )

        else:

            print(
                f"⚠️ File not found: {pdf}"
            )

    print(
        f"Loaded {len(documents)} documents."
    )

    # ========================================================
    # CLEAN DOCUMENTS
    # ========================================================

    for doc in documents:

        text = doc.page_content

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        doc.page_content = text.strip()

    print(
        "Documents cleaned successfully."
    )

    # ========================================================
    # SPLIT DOCUMENTS
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=300,

        chunk_overlap=50

    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Total Chunks: {len(chunks)}"
    )

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_model = HuggingFaceEmbeddings(

        model_name=
            "sentence-transformers/all-MiniLM-L6-v2"

    )

    print(
        "Embedding model loaded successfully."
    )

    # ========================================================
    # CREATE VECTOR DATABASE
    # ========================================================

    vector_db = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory=
            VECTOR_DB_PATH

    )

    print(
        "Vector database created successfully!"
    )


else:

    print("\n====================================")
    print("Existing Vector Database Found")
    print("====================================")

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_model = HuggingFaceEmbeddings(

        model_name=
            "sentence-transformers/all-MiniLM-L6-v2"

    )

    print(
        "Embedding model loaded successfully."
    )

    # ========================================================
    # LOAD EXISTING DATABASE
    # ========================================================

    vector_db = Chroma(

        persist_directory=
            VECTOR_DB_PATH,

        embedding_function=
            embedding_model

    )

    print(
        "Stored documents:",
        vector_db._collection.count()
    )

    print(
        "Existing vector database loaded successfully!"
    )


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_db.as_retriever(

    search_type="similarity",

    search_kwargs={

        "k": RETRIEVER_K

    }

)

print(
    f"Retriever created successfully. k={RETRIEVER_K}"
)


# ============================================================
# QUESTION REWRITING PROMPT
# ============================================================

rewrite_prompt = PromptTemplate(

    input_variables=[
        "history",
        "question"
    ],

    template="""
Rewrite the latest question as ONE standalone question.

Use the history only if needed.

Rules:
- Resolve pronouns such as it, they, them, their, this, that.
- Keep the original meaning.
- Do not invent information.
- Do not answer.
- Output ONLY the rewritten question.
- Keep it short.

Conversation History:
{history}

Latest User Question:
{question}

Rewritten Question:
"""
)


# ============================================================
# ANSWER PROMPT
# ============================================================

answer_prompt = PromptTemplate(

    input_variables=[
        "context",
        "question"
    ],

    template="""
You are the official DesFlyer FAQ assistant.

Answer the question ONLY using the context.

Rules:
- Give one direct answer.
- Use only the provided context.
- Do not invent information.
- Do not mention documents or context.
- Maximum 2 short sentences.
- Keep the answer concise.
- Do not ask a question.

Context:
{context}

Question:
{question}

Answer:
"""
)


print(
    "Prompts created successfully."
)


# ============================================================
# GREETINGS
# ============================================================

greetings = {

    "hi":
        "Hello! Welcome to DesFlyer. How can I assist you today?",

    "hello":
        "Hello! Welcome to DesFlyer. How can I assist you today?",

    "hey":
        "Hi! Welcome to DesFlyer. Feel free to ask me about DesFlyer.",

    "good morning":
        "Good morning! Welcome to DesFlyer.",

    "good afternoon":
        "Good afternoon! Welcome to DesFlyer.",

    "good evening":
        "Good evening! Welcome to DesFlyer.",

    "good night":
        "Good night! Thank you for visiting DesFlyer."

}


# ============================================================
# DESFLYER KEYWORDS
# ============================================================

keywords = [

    "desflyer",
    "des flyer",

    "software",

    "website",
    "websites",
    "web",
    "web development",

    "mobile",
    "application",
    "applications",
    "app",
    "apps",

    "service",
    "services",

    "ui",
    "ux",
    "ui ux",

    "startup",
    "startups",

    "business",
    "businesses",

    "client",
    "clients",

    "project",
    "projects",

    "contact",

    "career",
    "job",
    "internship",

    "android",
    "ios",

    "platform",
    "platforms",

    "office",
    "location",

    "development",
    "developer",
    "develop",

    "custom",
    "customized",

    "solution",
    "solutions",

    "database",
    "databases",

    "responsive",
    "redesign",
    "design"

]


print(
    "Keywords loaded successfully."
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = re.sub(
        r"[^\w\s]",
        " ",
        text.lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GREETING CHECK
# ============================================================

def check_greeting(question):

    clean_question = normalize_text(
        question
    )

    if clean_question in greetings:

        return greetings[
            clean_question
        ]

    greeting_list = [

        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "hello",
        "hey",
        "hi"

    ]

    for greeting in greeting_list:

        if clean_question.startswith(
            greeting + " "
        ):

            return greetings[
                greeting
            ]

    return None


# ============================================================
# FOLLOW-UP QUESTION CHECK
# ============================================================

def is_follow_up_question(question):

    clean_question = normalize_text(
        question
    )

    if not conversation_history:

        return False

    follow_up_patterns = [

        "which",
        "what about",
        "how about",

        "are they",
        "is it",
        "does it",
        "do they",

        "can they",
        "can it",
        "will they",

        "their",
        "they",
        "them",
        "it",

        "this",
        "that",
        "these",
        "those",

        "also",

        "which platform",
        "which platforms",

        "what platform",
        "what platforms",

        "responsive"

    ]

    for pattern in follow_up_patterns:

        if pattern in clean_question:

            return True

    return False


# ============================================================
# DESFLYER QUESTION CHECK
# ============================================================

def is_desflyer_question(question):

    clean_question = normalize_text(
        question
    )

    # Direct keyword
    if any(
        keyword in clean_question
        for keyword in keywords
    ):

        return True

    # Follow-up
    if is_follow_up_question(
        clean_question
    ):

        return True

    return False


# ============================================================
# GEMMA GENERATION
# ============================================================

def generate_text(
    final_prompt,
    max_new_tokens=24,
    max_time=25
):

    total_start = time.time()

    # ========================================================
    # TOKENIZATION
    # ========================================================

    try:

        token_start = time.time()

        inputs = tokenizer(

            final_prompt,

            return_tensors="pt",

            truncation=True,

            max_length=1024

        )

        tokenization_time = (
            time.time() - token_start
        )

        # ----------------------------------------------------
        # Move tensors to model device
        # ----------------------------------------------------

        inputs = {

            key: value.to(
                model.device
            )

            for key, value
            in inputs.items()

        }

        input_tokens = (
            inputs["input_ids"].shape[-1]
        )

        print(
            f"📥 Input tokens: {input_tokens}"
        )

        print(
            f"⏱️ Tokenization: "
            f"{tokenization_time:.2f} seconds"
        )

    except Exception as error:

        print(
            "❌ Tokenization error:",
            error
        )

        return ""


    # ========================================================
    # GENERATION
    # ========================================================

    try:

        generation_start = time.time()

        with torch.inference_mode():

            outputs = model.generate(

                **inputs,

                max_new_tokens=max_new_tokens,

                do_sample=False,

                use_cache=True,

                max_time=max_time,

                pad_token_id=
                    tokenizer.eos_token_id

            )

        generation_time = (
            time.time() - generation_start
        )

    except Exception as error:

        print(
            "❌ Generation error:",
            error
        )

        return ""


    # ========================================================
    # DECODE ONLY NEW TOKENS
    # ========================================================

    input_length = (
        inputs["input_ids"].shape[-1]
    )

    generated_tokens = outputs[

        0

    ][

        input_length:

    ]


    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


    total_generation_time = (
        time.time() - total_start
    )


    output_tokens = len(
        generated_tokens
    )


    print(
        f"📤 Output tokens: "
        f"{output_tokens}"
    )

    print(
        f"⏱️ Gemma generation: "
        f"{generation_time:.2f} seconds"
    )

    print(
        f"⏱️ Total generation process: "
        f"{total_generation_time:.2f} seconds"
    )


    return answer


# ============================================================
# QUESTION REWRITING
# ============================================================

def rewrite_question(question):

    # --------------------------------------------------------
    # Do NOT use Gemma unless this is a follow-up.
    # --------------------------------------------------------

    if not is_follow_up_question(
        question
    ):

        print(
            "⚡ Question rewriting skipped."
        )

        return question.strip()


    if not conversation_history:

        return question.strip()


    history_text = (
        format_conversation_history()
    )


    final_prompt = rewrite_prompt.format(

        history=history_text,

        question=question.strip()

    )


    print(
        "\n🔄 Rewriting follow-up question..."
    )

    print(
        "Original question:",
        question
    )


    rewrite_start = time.time()


    rewritten = generate_text(

        final_prompt,

        max_new_tokens=
            REWRITE_MAX_NEW_TOKENS,

        max_time=
            REWRITE_MAX_TIME

    )


    rewrite_time = (
        time.time() - rewrite_start
    )


    print(
        f"⏱️ Question rewrite time: "
        f"{rewrite_time:.2f} seconds"
    )


    if not rewritten:

        print(
            "⚠️ Rewriting failed. "
            "Using original question."
        )

        return question.strip()


    # ========================================================
    # CLEAN REWRITE
    # ========================================================

    rewritten = re.sub(

        r"^(rewritten question\s*:?)",

        "",

        rewritten,

        flags=re.IGNORECASE

    ).strip()


    rewritten = rewritten.split(
        "\n"
    )[0].strip()


    rewritten = rewritten.strip(
        "\"'"
    )


    if len(rewritten) < 3:

        return question.strip()


    print(
        "Rewritten question:",
        rewritten
    )


    return rewritten


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(retrieved_docs):

    context_parts = []

    current_length = 0


    for doc in retrieved_docs:

        content = (
            doc.page_content.strip()
        )

        if not content:

            continue


        remaining = (
            MAX_CONTEXT_CHARS -
            current_length
        )


        if remaining <= 0:

            break


        content = content[
            :remaining
        ]


        context_parts.append(
            content
        )


        current_length += len(
            content
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# ASK CHATBOT
# ============================================================

def ask_chatbot(question):

    total_start = time.time()


    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question:

        return (
            "I'm sorry, I could not understand your question."
        )


    original_question = (
        question.strip()
    )


    # ========================================================
    # EXIT
    # ========================================================

    if normalize_text(
        original_question
    ) in {

        "exit",
        "quit",
        "bye",
        "goodbye"

    }:

        return (
            "Goodbye! Thank you for visiting DesFlyer."
        )


    # ========================================================
    # GREETING
    # ========================================================

    greeting_response = check_greeting(

        original_question

    )


    if greeting_response:

        return greeting_response


    # ========================================================
    # QUESTION REWRITING
    # ========================================================

    search_question = rewrite_question(

        original_question

    )


    # ========================================================
    # KEYWORD CHECK
    # ========================================================

    if not (

        is_desflyer_question(
            original_question
        )

        or

        is_desflyer_question(
            search_question
        )

    ):

        return (
            "I'm sorry, I can only answer questions "
            "related to DesFlyer."
        )


    # ========================================================
    # PRINT QUESTION
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        "🎤 ORIGINAL USER INPUT:",
        original_question
    )

    print(
        "===================================="
    )


    if search_question != original_question:

        print(
            "\n🔎 SEARCH QUESTION:",
            search_question
        )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = time.time()


    try:

        retrieved_docs = retriever.invoke(

            search_question

        )

    except Exception as error:

        print(
            "❌ Retrieval error:",
            error
        )

        return (
            "Sorry, I could not retrieve "
            "the information right now."
        )


    retrieval_time = (
        time.time() - retrieval_start
    )


    print(
        f"⏱️ Retrieval time: "
        f"{retrieval_time:.2f} seconds"
    )


    print(
        "\n📚 RETRIEVED DOCUMENTS:",
        len(retrieved_docs)
    )


    # ========================================================
    # NO DOCUMENTS
    # ========================================================

    if not retrieved_docs:

        answer = (
            "I'm sorry, I could not find this information "
            "in the DesFlyer documents."
        )

        add_to_conversation_history(

            original_question,

            answer

        )

        return answer


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context_start = time.time()


    context = create_context(
        retrieved_docs
    )


    context_time = (
        time.time() - context_start
    )


    print(
        f"⏱️ Context preparation: "
        f"{context_time:.2f} seconds"
    )

    print(
        f"📄 Context characters: "
        f"{len(context)}"
    )


    # ========================================================
    # CREATE ANSWER PROMPT
    # ========================================================

    final_prompt = answer_prompt.format(

        context=context,

        question=search_question

    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print(
        "\n🤖 Generating answer..."
    )


    answer = generate_text(

        final_prompt,

        max_new_tokens=
            ANSWER_MAX_NEW_TOKENS,

        max_time=
            ANSWER_MAX_TIME

    )


    # ========================================================
    # CLEAN ANSWER
    # ========================================================

    answer = re.sub(

        r"^(final answer\s*:?)",

        "",

        answer,

        flags=re.IGNORECASE

    ).strip()


    answer = re.sub(

        r"\s+",

        " ",

        answer

    ).strip()


    answer = re.sub(

        r"^(answer\s*:?)",

        "",

        answer,

        flags=re.IGNORECASE

    ).strip()


    # Remove accidental question ending
    if answer.endswith("?"):

        answer = (

            answer[:-1].strip()

            + "."

        )


    # ========================================================
    # FALLBACK
    # ========================================================

    if not answer:

        answer = (

            "I'm sorry, I could not generate "
            "a suitable answer."
        )


    # ========================================================
    # SAVE CONVERSATION HISTORY
    # ========================================================

    add_to_conversation_history(

        original_question,

        answer

    )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.time() - total_start
    )


    # ========================================================
    # PRINT FINAL ANSWER
    # ========================================================

    print(
        "\n===== Generated Answer ====="
    )

    print(
        answer
    )

    print(
        "\n===================================="
    )

    print(
        f"⏱️ TOTAL CHATBOT TIME: "
        f"{total_time:.2f} seconds"
    )

    print(
        "===================================="
    )


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    print(
        "\n🧠 Conversation history:"
    )


    for i, turn in enumerate(

        conversation_history,

        start=1

    ):

        print(
            f"{i}. User: "
            f"{turn['user']}"
        )

        print(
            f"   Assistant: "
            f"{turn['assistant']}"
        )


    return answer


# ============================================================
# TEST CONVERSATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "DesFlyer RAG Conversation Test"
    )

    print(
        "===================================="
    )

    print(
        "Type 'clear' to clear history."
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        user_question = input(
            "\n🎤 You: "
        ).strip()


        if not user_question:

            continue


        if normalize_text(
            user_question
        ) == "clear":

            clear_conversation_history()

            continue


        if normalize_text(
            user_question
        ) in {

            "exit",
            "quit"

        }:

            print(
                "\n👋 Goodbye!"
            )

            break


        response = ask_chatbot(
            user_question
        )


        print(
            "\n🤖 DesFlyer:",
            response
        )
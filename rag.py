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

RETRIEVER_K = 2

MAX_HISTORY_TURNS = 4

MAX_CONTEXT_CHARS = 1600

MAX_HISTORY_CHARS = 700

ANSWER_MAX_NEW_TOKENS = 32


# ============================================================
# CPU SETTINGS
# ============================================================

torch.set_grad_enabled(False)

try:
    torch.set_num_threads(4)
except Exception:
    pass


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n====================================")
print("DesFlyer Qwen2.5-1.5B RAG Configuration")
print("====================================")

print("Model device:", model.device)
print("CUDA available:", torch.cuda.is_available())
print("CPU threads:", torch.get_num_threads())

print("====================================")


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = []


def clear_conversation_history():

    global conversation_history

    conversation_history = []

    print("\nConversation history cleared.")


def get_conversation_history():

    return conversation_history.copy()


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
            conversation_history[-MAX_HISTORY_TURNS:]
        )


def format_conversation_history():

    if not conversation_history:

        return "No previous conversation."

    history = []

    for turn in conversation_history:

        history.append(
            f"User: {turn['user']}"
        )

        history.append(
            f"Assistant: {turn['assistant']}"
        )

    text = "\n".join(history)

    return text[-MAX_HISTORY_CHARS:]


# ============================================================
# DOCUMENT FILES
# ============================================================

PDF_FILES = [

    "data/DesFlyer_Chatbot_QA.pdf",

    "data/Research & Development.pdf",

    "data/Chatbot dataset.pdf"

]


# ============================================================
# LOAD / CREATE VECTOR DATABASE
# ============================================================

if not os.path.exists(VECTOR_DB_PATH):

    print("\n====================================")
    print("Creating Vector Database")
    print("====================================")

    documents = []

    for pdf in PDF_FILES:

        if not os.path.exists(pdf):

            print(
                f"WARNING: File not found: {pdf}"
            )

            continue

        print(
            f"Loading: {pdf}"
        )

        try:

            loader = PyPDFLoader(pdf)

            loaded_documents = loader.load()

            documents.extend(
                loaded_documents
            )

            print(
                f"Loaded {len(loaded_documents)} pages from {pdf}"
            )

        except Exception as error:

            print(
                f"ERROR loading {pdf}: {error}"
            )


    print(
        f"\nTotal documents loaded: {len(documents)}"
    )


    # ========================================================
    # CLEAN DOCUMENTS
    # ========================================================

    for document in documents:

        text = document.page_content

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        document.page_content = text.strip()


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
        f"Total chunks created: {len(chunks)}"
    )


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    print(
        "\nLoading embedding model..."
    )


    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"

    )


    print(
        "Embedding model loaded successfully."
    )


    # ========================================================
    # CREATE CHROMA VECTOR DATABASE
    # ========================================================

    print(
        "\nCreating ChromaDB..."
    )


    vector_db = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory=
        VECTOR_DB_PATH

    )


    print(
        "ChromaDB created successfully."
    )


else:

    print("\n====================================")
    print("Existing Vector Database Found")
    print("====================================")


    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    print(
        "Loading embedding model..."
    )


    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"

    )


    print(
        "Embedding model loaded successfully."
    )


    # ========================================================
    # LOAD EXISTING CHROMADB
    # ========================================================

    vector_db = Chroma(

        persist_directory=
        VECTOR_DB_PATH,

        embedding_function=
        embedding_model

    )


    try:

        count = vector_db._collection.count()

    except Exception:

        count = "unknown"


    print(
        "Stored chunks:",
        count
    )

    print(
        "Existing ChromaDB loaded successfully."
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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:

        return ""

    text = text.lower()

    # Normalize common STT variation
    text = text.replace(
        "des flyer",
        "desflyer"
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GREETINGS
# ============================================================

GREETINGS = {

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


def check_greeting(question):

    clean_question = normalize_text(
        question
    )

    if clean_question in GREETINGS:

        return GREETINGS[
            clean_question
        ]

    for greeting in GREETINGS:

        if clean_question.startswith(
            greeting + " "
        ):

            return GREETINGS[
                greeting
            ]

    return None


# ============================================================
# DESFLYER KEYWORDS
# ============================================================

DESFLYER_KEYWORDS = [

    "desflyer",

    "software",
    "website",
    "websites",
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


# ============================================================
# DESFLYER QUESTION CHECK
# ============================================================

def is_desflyer_question(question):

    clean_question = normalize_text(
        question
    )


    # --------------------------------------------------------
    # Direct DesFlyer mention
    # --------------------------------------------------------

    if "desflyer" in clean_question:

        return True


    # --------------------------------------------------------
    # Strong combinations
    # --------------------------------------------------------

    strong_patterns = [

        ("software", "development"),

        ("website", "develop"),

        ("website", "build"),

        ("website", "create"),

        ("website", "database"),

        ("mobile", "application"),

        ("mobile", "app"),

        ("android", "application"),

        ("android", "app"),

        ("ios", "application"),

        ("ios", "app"),

        ("service", "provide"),

        ("services", "provide"),

        ("redesign", "website")

    ]


    for word1, word2 in strong_patterns:

        if (
            word1 in clean_question
            and
            word2 in clean_question
        ):

            return True


    # --------------------------------------------------------
    # Tanglish patterns
    # --------------------------------------------------------

    tanglish_patterns = [

        "enna service",

        "enna services",

        "service provide",

        "services provide",

        "pannanga",

        "pannuvanga",

        "irukka",

        "develop pannuvanga",

        "develop pannanga",

        "website develop",

        "website build",

        "website create",

        "mobile develop",

        "mobile app",

        "android support"

    ]


    for pattern in tanglish_patterns:

        if pattern in clean_question:

            return True


    return False


# ============================================================
# FAST FAQ ANSWERS
# ============================================================

FAST_FAQ = {

    "offer":
        "DesFlyer offers customized, high-quality, secure, and scalable software solutions.",

    "services":
        "DesFlyer provides software development services, including website and mobile application development.",

    "website":
        "Yes, DesFlyer develops websites.",

    "website_database":
        "Yes, DesFlyer can connect websites with databases.",

    "redesign":
        "Yes, DesFlyer can redesign and improve existing websites.",

    "mobile":
        "Yes, DesFlyer develops mobile applications.",

    "android":
        "Yes, DesFlyer develops mobile applications for Android.",

    "ios":
        "Yes, DesFlyer develops mobile applications for iOS."

}


# ============================================================
# FAST FAQ MATCHER
# ============================================================

def get_fast_faq_answer(question):

    q = normalize_text(
        question
    )


    # ========================================================
    # OFFER / SERVICES
    # ========================================================

    offer_patterns = [

        "what does desflyer offer",

        "what does desflyer provide",

        "what does desflyer do",

        "what kind of software solutions",

        "what software solutions",

        "desflyer offer",

        "desflyer provide",

        "desflyer services",

        "what services does desflyer provide",

        "which services does desflyer provide",

        "what services does desflyer offer",

        "which services does desflyer offer",

        "services does desflyer provide",

        "enna service provide",

        "enna services provide",

        "enna service",

        "enna services",

        "service provide pannanga",

        "services provide pannanga",

        "desflyer enna service"

    ]


    for pattern in offer_patterns:

        if pattern in q:

            return FAST_FAQ["services"]


    # ========================================================
    # GENERAL OFFER
    # ========================================================

    if (
        "what does desflyer offer" in q
        or
        "what kind of software" in q
    ):

        return FAST_FAQ["offer"]


    # ========================================================
    # WEBSITE DEVELOPMENT
    # ========================================================

    website_patterns = [

        "does desflyer develop websites",

        "does desflyer develop website",

        "can desflyer develop websites",

        "can desflyer develop website",

        "does desflyer build websites",

        "can desflyer build websites",

        "does desflyer create websites",

        "can desflyer create websites",

        "desflyer develop website",

        "desflyer develop websites",

        "desflyer website development",

        "website develop pannuvanga",

        "website develop pannanga",

        "website create pannuvanga",

        "website build pannuvanga",

        "website develop pannuvangala",

        "website build pannuvangala"

    ]


    for pattern in website_patterns:

        if pattern in q:

            return FAST_FAQ["website"]


    # ========================================================
    # WEBSITE + DATABASE
    # ========================================================

    has_website = (

        "website" in q

        or

        "websites" in q

        or

        "web" in q

    )


    has_database = (

        "database" in q

        or

        "databases" in q

    )


    if has_website and has_database:

        return FAST_FAQ[
            "website_database"
        ]


    # ========================================================
    # WEBSITE REDESIGN
    # ========================================================

    if (

        "redesign" in q

        and

        (
            "website" in q
            or
            "websites" in q
            or
            "web" in q
        )

    ):

        return FAST_FAQ[
            "redesign"
        ]


    # ========================================================
    # MOBILE APPLICATION
    # ========================================================

    has_mobile = (

        "mobile application" in q

        or

        "mobile applications" in q

        or

        "mobile app" in q

        or

        "mobile apps" in q

    )


    has_development_word = (

        "develop" in q

        or

        "development" in q

        or

        "build" in q

        or

        "create" in q

        or

        "make" in q

    )


    if has_mobile and has_development_word:

        return FAST_FAQ[
            "mobile"
        ]


    # ========================================================
    # MOBILE TAMIL / TANGLISH
    # ========================================================

    if (

        "mobile app develop pannuvanga" in q

        or

        "mobile app develop pannuvangala" in q

        or

        "mobile develop pannuvanga" in q

        or

        "mobile application develop" in q

    ):

        return FAST_FAQ[
            "mobile"
        ]


    # ========================================================
    # ANDROID
    # ========================================================

    if "android" in q:

        if (

            "app" in q

            or

            "application" in q

            or

            "develop" in q

            or

            "support" in q

            or

            "mobile" in q

            or

            "build" in q

        ):

            return FAST_FAQ[
                "android"
            ]


    # ========================================================
    # IOS
    # ========================================================

    if "ios" in q:

        if (

            "app" in q

            or

            "application" in q

            or

            "develop" in q

            or

            "support" in q

            or

            "mobile" in q

            or

            "build" in q

            or

            "desflyer" in q

        ):

            return FAST_FAQ[
                "ios"
            ]


    return None


# ============================================================
# FOLLOW-UP QUESTION DETECTION
# ============================================================

def is_follow_up_question(question):

    if not conversation_history:

        return False

    q = normalize_text(
        question
    )


    follow_up_patterns = [

        "they",

        "their",

        "them",

        "it",

        "this",

        "that",

        "these",

        "those",

        "what about",

        "how about",

        "can they",

        "can it",

        "does it",

        "do they",

        "will they",

        "are they",

        "is it",

        "and",

        "or"

    ]


    for pattern in follow_up_patterns:

        if pattern in q:

            return True


    return False


# ============================================================
# FOLLOW-UP NORMALIZATION
# ============================================================

def normalize_follow_up(question):

    q = question.strip()

    if not conversation_history:

        return q


    clean = normalize_text(
        q
    )


    # ========================================================
    # THEY
    # ========================================================

    q = re.sub(

        r"\bthey\b",

        "DesFlyer",

        q,

        flags=re.IGNORECASE

    )


    # ========================================================
    # THEIR
    # ========================================================

    q = re.sub(

        r"\btheir\b",

        "DesFlyer's",

        q,

        flags=re.IGNORECASE

    )


    # ========================================================
    # THEM
    # ========================================================

    q = re.sub(

        r"\bthem\b",

        "DesFlyer",

        q,

        flags=re.IGNORECASE

    )


    # ========================================================
    # IT
    # ========================================================

    q = re.sub(

        r"\bit\b",

        "DesFlyer",

        q,

        flags=re.IGNORECASE

    )


    # ========================================================
    # WHAT ABOUT / HOW ABOUT
    #
    # Example:
    #
    # "What about iOS?"
    #
    # becomes:
    #
    # "What about DesFlyer iOS?"
    # ========================================================

    clean_after_pronouns = normalize_text(
        q
    )


    if (
        clean_after_pronouns.startswith("what about ")
        or
        clean_after_pronouns.startswith("how about ")
    ):

        if "desflyer" not in clean_after_pronouns:

            q = (
                "What about DesFlyer "
                + q[
                    q.lower().find("about") + 5:
                ].strip()
            )


    # ========================================================
    # SHORT FOLLOW-UP
    #
    # Example:
    #
    # Previous:
    # "Does DesFlyer develop mobile applications?"
    #
    # User:
    # "Android."
    #
    # becomes:
    #
    # "DesFlyer Android."
    # ========================================================

    clean_after = normalize_text(
        q
    )

    short_follow_up_words = {

        "android",
        "ios",
        "iphone",
        "mobile",
        "website",
        "websites",
        "database",
        "databases",
        "responsive websites",
        "redesign"

    }


    if clean_after in short_follow_up_words:

        if "desflyer" not in clean_after:

            q = (
                "DesFlyer "
                + q
            )


    # ========================================================
    # STARTING WITH "AND"
    #
    # Example:
    #
    # "And iOS?"
    # ========================================================

    clean_after = normalize_text(
        q
    )


    if clean_after.startswith("and "):

        if "desflyer" not in clean_after:

            q = (
                "DesFlyer "
                + q[4:].strip()
            )


    # ========================================================
    # STARTING WITH "OR"
    #
    # Example:
    #
    # "or their responsive websites"
    # ========================================================

    clean_after = normalize_text(
        q
    )


    if clean_after.startswith("or "):

        if "desflyer" not in clean_after:

            q = (
                "DesFlyer "
                + q[3:].strip()
            )


    return q.strip()


# ============================================================
# CONTEXT CREATION
# ============================================================

def create_context(retrieved_docs):

    context_parts = []

    current_length = 0


    for document in retrieved_docs:

        content = (
            document.page_content.strip()
        )


        if not content:

            continue


        remaining = (

            MAX_CONTEXT_CHARS
            -
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
# RAG PROMPT
# ============================================================

answer_prompt = PromptTemplate(

    input_variables=[
        "context",
        "history",
        "question"
    ],

    template="""
You are the DesFlyer FAQ assistant.

Answer ONLY using the information provided in the context.

Context:
{context}

Previous conversation:
{history}

Current question:
{question}

Rules:
- Answer directly.
- Use simple English.
- Give one short complete answer.
- Do not repeat the question.
- Do not invent information.
- Do not ask a question.
- Do not output "Answer:".
- If the information is not available in the context, say:
"I could not find that information in the DesFlyer documents."

Answer:
"""

)


# ============================================================
# QWEN TEXT GENERATION
# ============================================================

def generate_text(
    final_prompt,
    max_new_tokens=ANSWER_MAX_NEW_TOKENS
):

    total_start = time.time()


    # ========================================================
    # TOKENIZATION
    # ========================================================

    try:

        token_start = time.time()


        if hasattr(
            tokenizer,
            "apply_chat_template"
        ):

            messages = [

                {
                    "role": "user",

                    "content":
                    final_prompt

                }

            ]


            inputs = tokenizer.apply_chat_template(

                messages,

                add_generation_prompt=True,

                return_tensors="pt",

                return_dict=True

            )


        else:

            inputs = tokenizer(

                final_prompt,

                return_tensors="pt",

                truncation=True,

                max_length=768

            )


        tokenization_time = (
            time.time()
            -
            token_start
        )


        # ====================================================
        # MOVE INPUT TO MODEL DEVICE
        # ====================================================

        inputs = {

            key:
            value.to(model.device)

            for key, value in inputs.items()

        }


        input_tokens = (
            inputs[
                "input_ids"
            ].shape[-1]
        )


        print(
            f"Input tokens: {input_tokens}"
        )


        print(
            f"Tokenization time: "
            f"{tokenization_time:.2f} seconds"
        )


    except Exception as error:

        print(
            f"Tokenization error: {error}"
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

                max_new_tokens=
                max_new_tokens,

                do_sample=False,

                num_beams=1,

                use_cache=True,

                pad_token_id=
                tokenizer.eos_token_id

            )


        generation_time = (
            time.time()
            -
            generation_start
        )


    except Exception as error:

        print(
            f"Generation error: {error}"
        )

        return ""


    # ========================================================
    # DECODE ONLY GENERATED TOKENS
    # ========================================================

    input_length = (
        inputs[
            "input_ids"
        ].shape[-1]
    )


    generated_tokens = (

        outputs[0][
            input_length:
        ]

    )


    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


    output_tokens = len(
        generated_tokens
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    if generation_time > 0:

        tokens_per_second = (

            output_tokens
            /
            generation_time

        )

    else:

        tokens_per_second = 0


    total_generation_time = (
        time.time()
        -
        total_start
    )


    print(
        f"Output tokens: {output_tokens}"
    )


    print(
        f"Qwen generation time: "
        f"{generation_time:.2f} seconds"
    )


    print(
        f"Generation speed: "
        f"{tokens_per_second:.2f} tokens/sec"
    )


    print(
        f"Total generation process: "
        f"{total_generation_time:.2f} seconds"
    )


    return answer


# ============================================================
# CLEAN MODEL ANSWER
# ============================================================

def clean_answer(answer):

    if not answer:

        return ""


    answer = answer.strip()


    # Remove labels

    answer = re.sub(

        r"^(answer|final answer)\s*:\s*",

        "",

        answer,

        flags=re.IGNORECASE

    ).strip()


    # Remove extra whitespace

    answer = re.sub(

        r"\s+",

        " ",

        answer

    ).strip()


    # Do not return another question

    if answer.endswith("?"):

        return ""


    # Remove numbered question-like output

    if re.match(

        r"^\d+\s*[\.\)]",

        answer

    ):

        return ""


    # Bad / incomplete answers

    bad_outputs = {

        "our",

        "yes",

        "no",

        "software",

        "development",

        "services",

        "software development",

        "software development services"

    }


    if normalize_text(answer) in bad_outputs:

        return ""


    if len(answer) < 5:

        return ""


    return answer


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(
    question,
    context
):

    history = format_conversation_history()


    final_prompt = answer_prompt.format(

        context=context,

        history=history,

        question=question

    )


    print(
        "\nGenerating answer with Qwen2.5..."
    )


    answer = generate_text(

        final_prompt,

        max_new_tokens=
        ANSWER_MAX_NEW_TOKENS

    )


    answer = clean_answer(
        answer
    )


    if answer:

        return answer


    print(
        "Model produced an incomplete answer."
    )


    return (
        "I could not find that information "
        "in the DesFlyer documents."
    )


# ============================================================
# MAIN CHATBOT FUNCTION
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
    # FAST FAQ
    #
    # First check original question.
    # ========================================================

    fast_answer = get_fast_faq_answer(

        original_question

    )


    if fast_answer:

        print(
            "\nFAST FAQ PATH"
        )

        print(
            "No LLM generation required."
        )


        add_to_conversation_history(

            original_question,

            fast_answer

        )


        total_time = (
            time.time()
            -
            total_start
        )


        print(
            f"Total chatbot time: "
            f"{total_time:.2f} seconds"
        )


        return fast_answer


    # ========================================================
    # FOLLOW-UP DETECTION
    # ========================================================

    follow_up = is_follow_up_question(

        original_question

    )


    # ========================================================
    # FOLLOW-UP NORMALIZATION
    # ========================================================

    search_question = normalize_follow_up(

        original_question

    )


    if search_question != original_question:

        print(
            "\nFollow-up normalization:"
        )

        print(
            "Original:",
            original_question
        )

        print(
            "Search:",
            search_question
        )


    # ========================================================
    # FAST FAQ AGAIN
    #
    # Important for:
    #
    # "What about iOS?"
    #
    # after normalization:
    #
    # "What about DesFlyer iOS?"
    # ========================================================

    fast_answer = get_fast_faq_answer(

        search_question

    )


    if fast_answer:

        print(
            "\nFAST FOLLOW-UP FAQ PATH"
        )

        print(
            "No LLM generation required."
        )


        add_to_conversation_history(

            original_question,

            fast_answer

        )


        total_time = (
            time.time()
            -
            total_start
        )


        print(
            f"Total chatbot time: "
            f"{total_time:.2f} seconds"
        )


        return fast_answer


    # ========================================================
    # DESFLYER QUESTION CHECK
    # ========================================================

    if not is_desflyer_question(

        original_question

    ):

        if not is_desflyer_question(

            search_question

        ):

            print(
                "\nNon-DesFlyer question."
            )


            return (
                "I'm sorry, I can only answer questions "
                "related to DesFlyer."
            )


    # ========================================================
    # LOG
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        "USER QUESTION:",
        original_question
    )

    print(
        "===================================="
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
            f"Retrieval error: {error}"
        )


        return (
            "Sorry, I could not retrieve "
            "the information right now."
        )


    retrieval_time = (
        time.time()
        -
        retrieval_start
    )


    print(
        f"Retrieval time: "
        f"{retrieval_time:.2f} seconds"
    )


    print(
        f"Retrieved documents: "
        f"{len(retrieved_docs)}"
    )


    # ========================================================
    # NO DOCUMENTS
    # ========================================================

    if not retrieved_docs:

        answer = (

            "I could not find this information "
            "in the DesFlyer documents."

        )


        add_to_conversation_history(

            original_question,

            answer

        )


        return answer


    # ========================================================
    # DISPLAY RETRIEVED CONTENT
    # ========================================================

    print(
        "\nRetrieved context:"
    )


    for index, document in enumerate(

        retrieved_docs,

        start=1

    ):

        preview = (
            document.page_content[:200]
            .replace("\n", " ")
        )


        print(
            f"{index}. {preview}..."
        )


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context_start = time.time()


    context = create_context(

        retrieved_docs

    )


    context_time = (
        time.time()
        -
        context_start
    )


    print(
        f"\nContext preparation time: "
        f"{context_time:.2f} seconds"
    )


    print(
        f"Context characters: "
        f"{len(context)}"
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    generation_start = time.time()


    answer = generate_rag_answer(

        search_question,

        context

    )


    generation_total_time = (
        time.time()
        -
        generation_start
    )


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    add_to_conversation_history(

        original_question,

        answer

    )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.time()
        -
        total_start
    )


    print(
        "\n===================================="
    )

    print(
        "GENERATED ANSWER:"
    )

    print(
        answer
    )


    print(
        "\nRetrieval time:",
        f"{retrieval_time:.2f} seconds"
    )


    print(
        "Generation time:",
        f"{generation_total_time:.2f} seconds"
    )


    print(
        "Total chatbot time:",
        f"{total_time:.2f} seconds"
    )


    print(
        "===================================="
    )


    return answer


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    "What does DesFlyer offer?",

    "Which services does DesFlyer provide?",

    "Does DesFlyer develop websites?",

    "Can they connect websites to databases?",

    "Can they redesign websites?",

    "What kind of software solutions does DesFlyer provide?",

    "Does DesFlyer develop mobile applications?",

    "Can DesFlyer build Android applications?",

    "Does DesFlyer support iOS applications?",

    "DesFlyer enna services provide pannanga?",

    "Website develop pannuvangala?",

    "Mobile app develop pannuvangala?",

    "Android support irukka?",

    "What about iOS?",

    "What about their responsive websites?",

    "And Android?",

    "How about iOS?"

]


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "DesFlyer Qwen2.5 RAG Chatbot"
    )

    print(
        "===================================="
    )

    print(
        "Type 'clear' to clear conversation history."
    )

    print(
        "Type 'test' to run test questions."
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        try:

            user_question = input(
                "\nYou: "
            ).strip()


        except KeyboardInterrupt:

            print(
                "\n\nProgram stopped."
            )

            break


        except EOFError:

            print(
                "\n\nProgram stopped."
            )

            break


        if not user_question:

            continue


        clean_command = normalize_text(

            user_question

        )


        # ====================================================
        # CLEAR
        # ====================================================

        if clean_command == "clear":

            clear_conversation_history()

            continue


        # ====================================================
        # TEST
        # ====================================================

        if clean_command == "test":

            print(
                "\n===================================="
            )

            print(
                "Running DesFlyer Test Questions"
            )

            print(
                "===================================="
            )


            clear_conversation_history()


            for question in TEST_QUESTIONS:

                print(
                    f"\nTEST QUESTION: {question}"
                )


                response = ask_chatbot(

                    question

                )


                print(
                    f"DesFlyer: {response}"
                )


            continue


        # ====================================================
        # EXIT
        # ====================================================

        if clean_command in {

            "exit",
            "quit"

        }:

            print(
                "\nGoodbye!"
            )

            break


        # ====================================================
        # NORMAL QUESTION
        # ====================================================

        response = ask_chatbot(

            user_question

        )


        print(
            f"\nDesFlyer: {response}"
        )
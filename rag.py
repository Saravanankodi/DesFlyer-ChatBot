import os
import re
import unicodedata
import psycopg

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from model import tokenizer, model


# ============================================================
# CONFIG
# ============================================================
VECTOR_DB_PATH = "vector_db"

PDF_FILES = [
    "data/DesFlyer_Chatbot_QA.pdf",
    "data/Research & Development.pdf",
    "data/Chatbot dataset.pdf",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 2

MAX_HISTORY_TURNS = 4

# Number of previous questions used to understand follow-ups
FOLLOW_UP_HISTORY_TURNS = 2

# Maximum history characters sent to the LLM
MAX_HISTORY_CHARS = 1000


# ============================================================
# POSTGRESQL CONFIG
# ============================================================
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = "divya"
POSTGRES_DATABASE = os.getenv(
    "POSTGRES_DATABASE",
    "conversation_db"
)


# ============================================================
# POSTGRESQL CONNECTION
# ============================================================
def get_postgres_connection():

    try:

        connection = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DATABASE
        )

        return connection

    except psycopg.Error as e:

        print(
            f"❌ PostgreSQL connection error: {e}"
        )

    return None


# ============================================================
# POSTGRESQL DATABASE / TABLE INITIALIZATION
# ============================================================
def initialize_postgres_database():

    print("\n====================================")
    print("Initializing PostgreSQL")
    print("====================================")

    connection = get_postgres_connection()

    if connection is None:

        print("❌ PostgreSQL is NOT connected.")
        print("⚠️ Conversation storage will not work.")
        print("⚠️ Check POSTGRES_PASSWORD and PostgreSQL server.")

        return False

    cursor = None

    try:

        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_question TEXT NOT NULL,
            resolved_question TEXT,
            assistant_answer TEXT NOT NULL,
            closing_question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        cursor.execute(
            create_table_query
        )

        connection.commit()

        print(
            "✅ PostgreSQL connected successfully."
        )

        print(
            "✅ PostgreSQL conversation table ready."
        )

        return True

    except psycopg.Error as e:

        print(
            f"❌ PostgreSQL table creation error: {e}"
        )

        connection.rollback()

        return False

    finally:

        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# CONVERSATION HISTORY
# ============================================================
conversation_history = []


# ============================================================
# LOAD POSTGRESQL CONVERSATION HISTORY
# ============================================================
def load_postgres_history():

    global conversation_history

    connection = get_postgres_connection()

    if connection is None:

        print(
            "⚠️ Could not load conversation history from PostgreSQL."
        )

        return

    cursor = None

    try:

        cursor = connection.cursor()

        query = """
        SELECT
            user_question,
            resolved_question,
            assistant_answer,
            closing_question
        FROM conversations
        ORDER BY id DESC
        LIMIT %s
        """

        cursor.execute(
            query,
            (MAX_HISTORY_TURNS,)
        )

        rows = cursor.fetchall()

        # Oldest conversation first
        rows.reverse()

        conversation_history = []

        for row in rows:

            conversation_history.append(
                {
                    "question": row[0] or "",
                    "answer": row[2] or "",
                    "closing": row[3] or "",
                    "resolved_question": row[1] or "",
                }
            )

        print(
            f"✅ Loaded {len(conversation_history)} "
            f"conversation(s) from PostgreSQL."
        )

    except psycopg.Error as e:

        print(
            f"❌ PostgreSQL history load error: {e}"
        )

    finally:

        if cursor is not None:
            cursor.close()

        connection.close()


# ============================================================
# FALLBACK
# ============================================================
DESFLYER_FALLBACK = (
    "I'm sorry, I can only answer questions related to DesFlyer."
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# CLEAN ANSWER
# ============================================================
def clean_answer(answer: str) -> str:

    if not answer:
        return ""

    answer = answer.strip()

    # --------------------------------------------------------
    # Remove prompt labels
    # --------------------------------------------------------
    answer = re.sub(
        r"^(assistant|answer|final answer)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    # --------------------------------------------------------
    # If the model returned the prompt and answer together,
    # keep only the text after the final "Answer:"
    # --------------------------------------------------------
    if "Answer:" in answer:

        parts = re.split(
            r"Answer:\s*",
            answer,
            flags=re.IGNORECASE
        )

        if len(parts) > 1:

            answer = parts[-1].strip()

    # --------------------------------------------------------
    # Remove common LLM meta commentary
    # --------------------------------------------------------
    meta_patterns = [
        r"\s*This response directly addresses.*$",
        r"\s*The response directly addresses.*$",
        r"\s*The answer directly addresses.*$",
        r"\s*The tone remains.*$",
        r"\s*This answer remains.*$",
        r"\s*Let me know if.*$",
        r"\s*I'm here to help.*$",
        r"\s*Feel free to ask.*$",
        r"\s*If you need further assistance.*$",
        r"\s*If you have any additional questions.*$",
        r"\s*Please let me know if.*$",
        r"\s*I hope this helps.*$",
    ]

    for pattern in meta_patterns:

        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE
        ).strip()

    # --------------------------------------------------------
    # Remove generated closing questions
    # Our system adds the closing question separately.
    # --------------------------------------------------------
    closing_patterns = [
        r"\s*Would you like to know more.*$",
        r"\s*Would you like details.*$",
        r"\s*Would you like more information.*$",
        r"\s*Would you like to explore.*$",
        r"\s*Would you like.*$",
        r"\s*Can I help you.*$",
        r"\s*Is there anything else.*$",
        r"\s*Do you have any other questions.*$",
        r"\s*Any other questions.*$",
    ]

    for pattern in closing_patterns:

        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE
        ).strip()

    # --------------------------------------------------------
    # Remove obvious retrieved Q&A leakage
    # --------------------------------------------------------
    leakage_patterns = [
        r"\s+\d+\.\s*(How|What|Can|Does|Do|Is|Are|Please|Would)\b.*$",
        r"\s+\d+\s+(How|What|Can|Does|Do|Is|Are|Please|Would)\b.*$",
        r"\s+\d+\.\s*How does\b.*$",
        r"\s+\d+\.\s*What does\b.*$",
        r"\s+\d+\.\s*Can you\b.*$",
        r"\s+\d+\.\s*Does DesFlyer\b.*$",
    ]

    for pattern in leakage_patterns:

        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE
        ).strip()

    # --------------------------------------------------------
    # Remove duplicate sentences
    # --------------------------------------------------------
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    unique_sentences = []

    seen = set()

    for sentence in sentences:

        sentence = sentence.strip()

        normalized_sentence = normalize_text(
            sentence
        )

        if (
            normalized_sentence
            and normalized_sentence not in seen
        ):

            seen.add(
                normalized_sentence
            )

            unique_sentences.append(
                sentence
            )

    answer = " ".join(
        unique_sentences
    )

    # --------------------------------------------------------
    # Remove extra whitespace
    # --------------------------------------------------------
    answer = re.sub(
        r"\s+",
        " ",
        answer
    )

    return answer.strip()


# ============================================================
# GREETINGS
# ============================================================
GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hai",
    "good morning",
    "good afternoon",
    "good evening",
}


def is_greeting(question: str) -> bool:

    normalized = normalize_text(
        question
    )

    return normalized in GREETINGS


def get_greeting_response(question: str) -> str:

    normalized = normalize_text(
        question
    )

    if "good morning" in normalized:

        return (
            "Good morning! "
            "How can I help you with DesFlyer?"
        )

    if "good afternoon" in normalized:

        return (
            "Good afternoon! "
            "How can I help you with DesFlyer?"
        )

    if "good evening" in normalized:

        return (
            "Good evening! "
            "How can I help you with DesFlyer?"
        )

    return (
        "Hello! "
        "How can I help you with DesFlyer?"
    )


# ============================================================
# FAST FAQ
# ============================================================
# These are deterministic DesFlyer FAQ answers.
# Matching these questions skips ChromaDB retrieval and Qwen
# generation, so common voice questions return much faster.
# ============================================================

FAST_FAQ = {

    # SERVICES
    "what service does desflyer provide":
        "DesFlyer provides digital solutions such as website development, mobile application development, and other software solutions.",

    "what services does desflyer provide":
        "DesFlyer provides digital solutions such as website development, mobile application development, and other software solutions.",

    "what services does desflyer offer":
        "DesFlyer provides digital solutions such as website development, mobile application development, and other software solutions.",

    "what does desflyer offer":
        "DesFlyer provides digital solutions such as website development, mobile application development, and other software solutions.",

    "which services does desflyer provide":
        "DesFlyer provides digital solutions such as website development, mobile application development, and other software solutions.",

    # WEBSITE
    "does desflyer develop websites":
        "Yes, DesFlyer develops websites.",

    "can desflyer build websites":
        "Yes, DesFlyer develops websites.",

    "does desflyer create websites":
        "Yes, DesFlyer develops websites.",

    # DATABASE
    "does desflyer provide database services":
        "Yes, DesFlyer can connect websites to databases.",

    "can desflyer connect websites to databases":
        "Yes, DesFlyer can connect websites to databases.",

    "can they connect websites to databases":
        "Yes, DesFlyer can connect websites to databases.",

    # REDESIGN
    "can desflyer redesign an existing website":
        "Yes, DesFlyer can redesign and improve existing websites.",

    "can desflyer redesign existing websites":
        "Yes, DesFlyer can redesign and improve existing websites.",

    "can desflyer redesign websites":
        "Yes, DesFlyer can redesign and improve existing websites.",

    # MOBILE
    "does desflyer develop mobile applications":
        "Yes, DesFlyer develops mobile applications.",

    "can desflyer build mobile applications":
        "Yes, DesFlyer develops mobile applications.",

    "can they build mobile applications":
        "Yes, DesFlyer develops mobile applications.",

    # ANDROID
    "does desflyer develop android applications":
        "Yes, DesFlyer develops mobile applications for Android.",

    "does desflyer support android":
        "Yes, DesFlyer develops mobile applications for Android.",

    "can desflyer build android applications":
        "Yes, DesFlyer develops mobile applications for Android.",

    # IOS
    "does desflyer develop ios applications":
        "Yes, DesFlyer develops mobile applications for iOS.",

    "does desflyer support ios":
        "Yes, DesFlyer develops mobile applications for iOS.",

    "can desflyer build ios applications":
        "Yes, DesFlyer develops mobile applications for iOS.",

    # PLATFORMS
    "what platforms does desflyer support":
        "DesFlyer supports Android and iOS platforms for mobile application development.",

    "which platforms does desflyer support":
        "DesFlyer supports Android and iOS platforms for mobile application development.",

    # UI / UX
    "does desflyer provide ui ux design":
        "Yes, DesFlyer provides UI/UX design services.",

    "does desflyer provide ui ux design services":
        "Yes, DesFlyer provides UI/UX design services.",

    "does desflyer provide ui and ux design":
        "Yes, DesFlyer provides UI/UX design services.",

    # CUSTOM SOFTWARE
    "does desflyer develop custom software":
        "Yes, DesFlyer develops custom software solutions.",

    "can desflyer develop custom software":
        "Yes, DesFlyer develops custom software solutions.",

    # RESPONSIVE
    "does desflyer build responsive applications":
        "Yes, DesFlyer can develop responsive applications that work across different screen sizes.",

    "are desflyer websites responsive":
        "Yes, DesFlyer can develop responsive websites and applications that work across different screen sizes.",

    # GENERAL
    "what does desflyer do":
        "DesFlyer provides digital solutions including website development, mobile application development, and software solutions.",

    "what is desflyer":
        "DesFlyer is a digital solutions company that provides services such as website and mobile application development.",
}


# ============================================================
# FAQ ALIASES / NATURAL VOICE VARIATIONS
# ============================================================
FAQ_ALIASES = {

    # Services
    "what service does desflyer provide": "what service does desflyer provide",
    "what services does desflyer provide": "what services does desflyer provide",
    "what services does desflyer offer": "what services does desflyer offer",
    "what does desflyer offer": "what does desflyer offer",
    "which services does desflyer provide": "which services does desflyer provide",

    # Website
    "does desflyer develop website": "does desflyer develop websites",
    "does desflyer develop websites": "does desflyer develop websites",
    "can desflyer develop websites": "can desflyer build websites",
    "can desflyer build websites": "can desflyer build websites",
    "does desflyer create websites": "does desflyer create websites",

    # Database
    "does desflyer provide database services": "does desflyer provide database services",
    "can desflyer connect website to database": "can desflyer connect websites to databases",
    "can desflyer connect websites to databases": "can desflyer connect websites to databases",
    "can they connect websites to databases": "can they connect websites to databases",

    # Redesign
    "can desflyer redesign an existing website": "can desflyer redesign an existing website",
    "can desflyer redesign existing websites": "can desflyer redesign existing websites",
    "can desflyer redesign websites": "can desflyer redesign websites",

    # Mobile
    "does desflyer develop mobile applications": "does desflyer develop mobile applications",
    "does desflyer develop mobile application": "does desflyer develop mobile applications",
    "can desflyer build mobile applications": "can desflyer build mobile applications",
    "can desflyer build mobile apps": "can desflyer build mobile applications",

    # Android
    "does desflyer develop android applications": "does desflyer develop android applications",
    "does desflyer develop android apps": "does desflyer develop android applications",
    "does desflyer support android": "does desflyer support android",
    "can desflyer build android applications": "can desflyer build android applications",

    # iOS
    "does desflyer develop ios applications": "does desflyer develop ios applications",
    "does desflyer develop ios apps": "does desflyer develop ios applications",
    "does desflyer support ios": "does desflyer support ios",
    "can desflyer build ios applications": "can desflyer build ios applications",

    # Platforms
    "what platform does desflyer support": "what platforms does desflyer support",
    "which platform does desflyer support": "which platforms does desflyer support",
    "what platforms does desflyer support": "what platforms does desflyer support",
    "which platforms does desflyer support": "which platforms does desflyer support",

    # UI/UX
    "does desflyer provide ui ux design": "does desflyer provide ui ux design",
    "does desflyer provide ui ux design services": "does desflyer provide ui ux design services",
    "does desflyer provide ui and ux design": "does desflyer provide ui and ux design",

    # Custom software
    "does desflyer develop custom software": "does desflyer develop custom software",
    "can desflyer develop custom software": "can desflyer develop custom software",

    # Responsive
    "does desflyer build responsive applications": "does desflyer build responsive applications",
    "are desflyer websites responsive": "are desflyer websites responsive",
}


# ============================================================
# FAQ INTENT
# ============================================================
def get_faq_intent(question: str) -> str:

    q = normalize_text(
        question
    )

    if (
        "service" in q
        or "services" in q
        or "provide" in q
        or "provides" in q
        or "offer" in q
        or "offers" in q
    ):

        return "services"

    if (
        "website" in q
        or "web application" in q
        or "web development" in q
        or "redesign" in q
        or "responsive" in q
    ):

        return "website"

    if (
        "android" in q
        or "ios" in q
        or "mobile application" in q
        or "mobile app" in q
    ):

        return "mobile"

    if (
        "custom software" in q
        or "software development" in q
        or "software solution" in q
    ):

        return "software"

    if (
        "ui" in q
        or "ux" in q
        or "design" in q
    ):

        return "design"

    if "database" in q:

        return "database"

    if "platform" in q:

        return "platform"

    if "what is desflyer" in q:

        return "about"

    return "general"


# ============================================================
# CLOSING QUESTIONS
# ============================================================
def get_closing_question(intent: str = None) -> str:

    if intent == "services":

        return (
            "Would you like details about a specific DesFlyer service?"
        )

    if intent == "website":

        return (
            "Would you like to know more about DesFlyer's website development?"
        )

    if intent == "mobile":

        return (
            "Would you like to know more about DesFlyer's mobile application development?"
        )

    if intent == "software":

        return (
            "Would you like to know more about DesFlyer's custom software solutions?"
        )

    if intent == "design":

        return (
            "Would you like to know more about DesFlyer's UI and UX design services?"
        )

    if intent == "database":

        return (
            "Would you like to know more about DesFlyer's database-related solutions?"
        )

    if intent == "platform":

        return (
            "Would you like to know which platforms DesFlyer supports?"
        )

    if intent == "about":

        return (
            "Would you like to know more about DesFlyer?"
        )

    return (
        "Would you like to know more about DesFlyer?"
    )


# ============================================================
# DESFLYER QUESTION VALIDATION
# ============================================================
def is_valid_desflyer_question(question: str) -> bool:

    normalized = normalize_text(
        question
    )

    if not normalized:

        return False

    # --------------------------------------------------------
    # DesFlyer + valid topic
    # --------------------------------------------------------
    if "desflyer" in normalized:

        valid_topics = [

            "service",
            "services",
            "provide",
            "provides",
            "offer",
            "offers",
            "website",
            "web",
            "development",
            "developer",
            "develop",
            "mobile",
            "application",
            "app",
            "android",
            "ios",
            "platform",
            "software",
            "solution",
            "solutions",
            "database",
            "redesign",
            "responsive",
            "design",
            "ui",
            "ux",
            "support",
            "contact",
            "location",
            "career",
            "internship",
            "custom"
        ]

        for topic in valid_topics:

            if topic in normalized:

                return True

        return False

    # --------------------------------------------------------
    # Questions without DesFlyer
    # --------------------------------------------------------
    valid_combinations = [

        ("website", "develop"),
        ("website", "development"),
        ("website", "build"),
        ("website", "redesign"),

        ("mobile", "application"),
        ("mobile", "app"),

        ("android", "application"),
        ("android", "app"),

        ("ios", "application"),
        ("ios", "app"),

        ("ui", "ux"),

        ("custom", "software"),
        ("software", "development"),

        ("platform", "android"),
        ("platform", "ios"),
    ]

    for first_word, second_word in valid_combinations:

        if (
            first_word in normalized
            and second_word in normalized
        ):

            return True

    return False


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================
FOLLOW_UP_WORDS = {

    "it",
    "they",
    "them",
    "this",
    "that",
    "their",
    "its",

}


def is_follow_up(question: str) -> bool:

    words = normalize_text(
        question
    ).split()

    if not words:

        return False

    if len(words) <= 8:

        for word in words:

            if word in FOLLOW_UP_WORDS:

                return True

    return False


# ============================================================
# RESOLVE FOLLOW-UP
# ============================================================
def resolve_follow_up(question: str) -> str:

    """
    Resolve the current question using only recent
    conversation QUESTIONS.

    IMPORTANT:
    Previous assistant answers are NOT used to create
    the ChromaDB retrieval query.

    PostgreSQL provides conversation memory.
    ChromaDB provides DesFlyer knowledge.
    """

    if not conversation_history:

        return question.strip()

    normalized = normalize_text(
        question
    )

    # --------------------------------------------------------
    # Use only the latest 2 conversation turns
    # --------------------------------------------------------
    recent_history = conversation_history[
        -FOLLOW_UP_HISTORY_TURNS:
    ]

    previous_questions = []

    for item in recent_history:

        resolved = item.get(
            "resolved_question",
            ""
        ).strip()

        original = item.get(
            "question",
            ""
        ).strip()

        if resolved:

            previous_questions.append(
                resolved
            )

        elif original:

            previous_questions.append(
                original
            )

    if not previous_questions:

        return question.strip()

    # Most recent meaningful question
    previous_question = previous_questions[-1]

    # ========================================================
    # PRONOUN FOLLOW-UP
    # ========================================================
    #
    # Example:
    #
    # User:
    # What services does DesFlyer offer?
    #
    # User:
    # Can they connect websites to databases?
    #
    # Result:
    # Can DesFlyer connect websites to databases?
    #
    # ========================================================

    if is_follow_up(question):

        resolved = re.sub(
            r"\b(it|they|them|their|its)\b",
            "DesFlyer",
            question,
            flags=re.IGNORECASE
        )

        resolved_normalized = normalize_text(
            resolved
        )

        # If DesFlyer was added but the question is
        # still very short, attach the previous topic.
        if len(resolved_normalized.split()) <= 4:

            return (
                f"{previous_question} "
                f"{resolved.strip()}"
            ).strip()

        return resolved.strip()

    # ========================================================
    # WHAT / WHICH PLATFORM
    # ========================================================
    if "what platform" in normalized:

        return (
            "What platforms does DesFlyer support?"
        )

    if "which platform" in normalized:

        return (
            "What platforms does DesFlyer support?"
        )

    # ========================================================
    # WHAT ABOUT ANDROID
    # ========================================================
    if "what about android" in normalized:

        return (
            "Does DesFlyer develop Android applications?"
        )

    # ========================================================
    # WHAT ABOUT IOS
    # ========================================================
    if "what about ios" in normalized:

        return (
            "Does DesFlyer develop iOS applications?"
        )

    # ========================================================
    # WHAT ABOUT X
    # ========================================================
    if normalized.startswith("what about "):

        topic = re.sub(
            r"^what about\s+",
            "",
            question.strip(),
            flags=re.IGNORECASE
        ).strip()

        if topic:

            return (
                f"What about DesFlyer's {topic}?"
            )

    # ========================================================
    # HOW ABOUT X
    # ========================================================
    if normalized.startswith("how about "):

        topic = re.sub(
            r"^how about\s+",
            "",
            question.strip(),
            flags=re.IGNORECASE
        ).strip()

        if topic:

            return (
                f"What about DesFlyer's {topic}?"
            )

    # ========================================================
    # WHICH ONE / WHAT ABOUT IT
    # ========================================================
    if normalized in {

        "which one",
        "which one?",
        "what about it",
        "what about that",
        "how about it",
        "how about that",

    }:

        return previous_question

    # ========================================================
    # INCOMPLETE SUPPORT QUESTION
    # ========================================================
    if (
        "does support" in normalized
        or "do support" in normalized
        or "can support" in normalized
    ):

        topic = re.sub(
            r"\b(does|do|can)\s+support\b",
            "",
            question,
            flags=re.IGNORECASE
        ).strip()

        topic = topic.rstrip(
            "?"
        ).strip()

        if topic:

            return (
                f"Does DesFlyer support {topic}?"
            )

    # ========================================================
    # WHAT TYPE
    # ========================================================
    if normalized.startswith("what type"):

        return (
            f"What type of solution does DesFlyer provide "
            f"regarding {question.strip()}?"
        )

    # ========================================================
    # SHORT CONTEXTUAL QUESTION
    # ========================================================
    #
    # Example:
    #
    # Previous:
    # What database solutions does DesFlyer provide?
    #
    # Current:
    # And performance?
    #
    # Result:
    # What database solutions does DesFlyer provide?
    # Regarding And performance?
    #
    # ========================================================

    words = normalized.split()

    if len(words) <= 4:

        context_words = {

            "performance",
            "pricing",
            "cost",
            "support",
            "development",
            "database",
            "website",
            "mobile",
            "android",
            "ios",
            "design",
            "software",
            "platform",
            "features",
            "process",
            "connection",
            "integration",
            "security",

        }

        if any(
            word in context_words
            for word in words
        ):

            return (
                f"{previous_question} "
                f"Regarding {question.strip()}"
            ).strip()

    # ========================================================
    # NORMAL QUESTION
    # ========================================================
    return question.strip()


# ============================================================
# FAQ SEARCH
# ============================================================
def find_fast_faq(question: str):

    normalized = normalize_text(
        question
    )

    # --------------------------------------------------------
    # 1. Exact / alias match
    # --------------------------------------------------------
    canonical_question = FAQ_ALIASES.get(
        normalized
    )

    if canonical_question:
        return FAST_FAQ.get(
            canonical_question
        )

    if normalized in FAST_FAQ:
        return FAST_FAQ[normalized]

    # --------------------------------------------------------
    # 2. Small natural-language variations
    # --------------------------------------------------------

    # SERVICES
    if (
        "desflyer" in normalized
        and (
            "service" in normalized
            or "services" in normalized
            or "provide" in normalized
            or "provides" in normalized
            or "offer" in normalized
            or "offers" in normalized
        )
    ):
        return FAST_FAQ[
            "what services does desflyer provide"
        ]

    # WEBSITE
    if (
        "desflyer" in normalized
        and (
            "website" in normalized
            or "websites" in normalized
        )
        and (
            "develop" in normalized
            or "build" in normalized
            or "create" in normalized
        )
        and "redesign" not in normalized
    ):
        return FAST_FAQ[
            "does desflyer develop websites"
        ]

    # WEBSITE + DATABASE
    if (
        (
            "website" in normalized
            or "websites" in normalized
        )
        and (
            "database" in normalized
            or "databases" in normalized
        )
        and (
            "connect" in normalized
            or "integrate" in normalized
            or "provide" in normalized
            or "services" in normalized
        )
    ):
        return FAST_FAQ[
            "can desflyer connect websites to databases"
        ]

    # REDESIGN
    if (
        "redesign" in normalized
        and (
            "website" in normalized
            or "websites" in normalized
            or "web" in normalized
        )
    ):
        return FAST_FAQ[
            "can desflyer redesign websites"
        ]

    # MOBILE
    if (
        "mobile" in normalized
        and (
            "application" in normalized
            or "applications" in normalized
            or "app" in normalized
            or "apps" in normalized
        )
        and (
            "develop" in normalized
            or "build" in normalized
            or "create" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer develop mobile applications"
        ]

    # ANDROID
    if (
        "android" in normalized
        and (
            "app" in normalized
            or "application" in normalized
            or "develop" in normalized
            or "build" in normalized
            or "support" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer develop android applications"
        ]

    # IOS
    if (
        "ios" in normalized
        and (
            "app" in normalized
            or "application" in normalized
            or "develop" in normalized
            or "build" in normalized
            or "support" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer develop ios applications"
        ]

    # PLATFORMS
    if (
        "platform" in normalized
        and (
            "desflyer" in normalized
            or "mobile" in normalized
            or "android" in normalized
            or "ios" in normalized
        )
    ):
        return FAST_FAQ[
            "what platforms does desflyer support"
        ]

    # UI / UX
    if (
        ("ui" in normalized or "ux" in normalized)
        and (
            "desflyer" in normalized
            or "design" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer provide ui ux design"
        ]

    # CUSTOM SOFTWARE
    if (
        "software" in normalized
        and (
            "custom" in normalized
            or "customized" in normalized
        )
        and (
            "desflyer" in normalized
            or "develop" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer develop custom software"
        ]

    # RESPONSIVE
    if (
        "responsive" in normalized
        and (
            "desflyer" in normalized
            or "website" in normalized
            or "websites" in normalized
            or "application" in normalized
        )
    ):
        return FAST_FAQ[
            "does desflyer build responsive applications"
        ]

    return None


# ============================================================
# LOAD DOCUMENTS
# ============================================================
def load_documents():

    documents = []

    for pdf_file in PDF_FILES:

        if not os.path.exists(pdf_file):

            print(
                f"⚠️ PDF not found: {pdf_file}"
            )

            continue

        try:

            loader = PyPDFLoader(
                pdf_file
            )

            docs = loader.load()

            documents.extend(
                docs
            )

            print(
                f"Loaded {len(docs)} documents "
                f"from {pdf_file}"
            )

        except Exception as e:

            print(
                f"⚠️ Error loading {pdf_file}: {e}"
            )

    return documents


# ============================================================
# CLEAN DOCUMENTS
# ============================================================
def clean_documents(documents):

    cleaned_documents = []

    for document in documents:

        text = document.page_content

        text = unicodedata.normalize(
            "NFKC",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip()

        if text:

            document.page_content = text

            cleaned_documents.append(
                document
            )

    return cleaned_documents


# ============================================================
# CREATE / LOAD VECTOR DATABASE
# ============================================================
def create_vector_db():

    # --------------------------------------------------------
    # LOAD EXISTING VECTOR DB
    # --------------------------------------------------------
    if os.path.exists(
        VECTOR_DB_PATH
    ):

        try:

            print(
                "\n📂 Existing ChromaDB found."
            )

            print(
                "Loading existing vector database..."
            )

            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )

            vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )

            print(
                "✅ Existing ChromaDB loaded."
            )

            return vector_db

        except Exception as e:

            print(
                f"⚠️ Existing ChromaDB load failed: {e}"
            )

    # --------------------------------------------------------
    # CREATE NEW VECTOR DB
    # --------------------------------------------------------
    print(
        "\n📄 Creating new ChromaDB..."
    )

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    documents = clean_documents(
        documents
    )

    print(
        "✅ Documents cleaned."
    )

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Total Chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "✅ Embedding model loaded."
    )

    # --------------------------------------------------------
    # CREATE CHROMA
    # --------------------------------------------------------
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print(
        "✅ Vector DB created."
    )

    return vector_db


# ============================================================
# INITIALIZE POSTGRESQL FIRST
# ============================================================
postgres_ready = initialize_postgres_database()

load_postgres_history()


# ============================================================
# INITIALIZE VECTOR DB
# ============================================================
vector_db = create_vector_db()


# ============================================================
# RETRIEVER
# ============================================================
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)

print(
    "✅ Retriever created."
)


# ============================================================
# CREATE CONTEXT
# ============================================================
def create_context(documents) -> str:

    if not documents:

        return ""

    context_parts = []

    for document in documents:

        text = document.page_content.strip()

        if text:

            context_parts.append(
                text
            )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# CONVERSATION HISTORY TEXT
# ============================================================
def get_conversation_history_text() -> str:

    """
    IMPORTANT:

    Only previous USER QUESTIONS and RESOLVED QUESTIONS
    are given to the LLM.

    Previous assistant answers are intentionally NOT included.

    This prevents the LLM from copying old answers,
    repeating responses, or generating unrelated content.
    """

    if not conversation_history:

        return ""

    recent_history = conversation_history[
        -FOLLOW_UP_HISTORY_TURNS:
    ]

    history_parts = []

    for item in recent_history:

        question = item.get(
            "question",
            ""
        ).strip()

        resolved_question = item.get(
            "resolved_question",
            ""
        ).strip()

        if not question:

            continue

        history_parts.append(
            f"Previous user question: {question}\n"
            f"Resolved question: {resolved_question}"
        )

    if not history_parts:

        return ""

    history_text = "\n\n".join(
        history_parts
    )

    # Keep history compact
    if len(history_text) > MAX_HISTORY_CHARS:

        history_text = history_text[
            -MAX_HISTORY_CHARS:
        ]

    return history_text


# ============================================================
# PROMPT
# ============================================================
def create_prompt(
    question: str,
    context: str,
    history: str
) -> str:

    prompt = f"""
You are the DesFlyer company FAQ assistant.

Your task is to answer the CURRENT USER QUESTION.

Use ONLY the information available in the
CURRENT DESFLYER CONTEXT.

CONVERSATION HISTORY is only for understanding
what the user means. It is NOT an answer source.

STRICT RULES:

1. Answer ONLY the current question.
2. Use the current DesFlyer context as the source of truth.
3. Do NOT copy previous answers.
4. Do NOT repeat previous conversation.
5. Do NOT talk about your response.
6. Do NOT explain how you generated the answer.
7. Do NOT say "This response directly addresses..."
8. Do NOT say "The tone remains..."
9. Do NOT say "Let me know..."
10. Do NOT say "I'm here to help..."
11. Do NOT generate a closing question.
12. Do NOT generate multiple questions.
13. Do NOT add information that is not in the context.
14. Keep the answer short and natural for voice conversation.
15. Give only the factual answer to the current question.

PREVIOUS CONVERSATION QUESTIONS:
{history}

CURRENT DESFLYER CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

ANSWER:
"""

    return prompt


# ============================================================
# GENERATE ANSWER
# ============================================================
def generate_answer(
    question: str,
    context: str,
    history: str
) -> str:

    prompt = create_prompt(
        question=question,
        context=context,
        history=history
    )

    try:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # ----------------------------------------------------
        # Extract only the generated answer
        # ----------------------------------------------------
        if "ANSWER:" in generated_text:

            generated_text = generated_text.split(
                "ANSWER:",
                1
            )[1]

        elif "Answer:" in generated_text:

            generated_text = generated_text.split(
                "Answer:",
                1
            )[1]

        # ----------------------------------------------------
        # Remove prompt if it was repeated
        # ----------------------------------------------------
        elif generated_text.startswith(prompt):

            generated_text = generated_text[
                len(prompt):
            ]

        return clean_answer(
            generated_text
        )

    except Exception as e:

        print(
            f"⚠️ LLM generation error: {e}"
        )

        return ""


# ============================================================
# SAVE CONVERSATION TO POSTGRESQL
# ============================================================
def save_conversation(
    question: str,
    answer: str,
    closing: str = "",
    resolved_question: str = ""
):

    global conversation_history

    clean_history_answer = clean_answer(
        answer
    )

    # --------------------------------------------------------
    # POSTGRESQL INSERT
    # --------------------------------------------------------
    connection = get_postgres_connection()

    if connection is None:

        print(
            "❌ Conversation NOT saved to PostgreSQL "
            "because PostgreSQL connection failed."
        )

    else:

        cursor = None

        try:

            cursor = connection.cursor()

            insert_query = """
            INSERT INTO conversations (
                user_question,
                resolved_question,
                assistant_answer,
                closing_question
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """

            values = (
                question,
                resolved_question or None,
                clean_history_answer,
                closing or None
            )

            cursor.execute(
                insert_query,
                values
            )

            row_id = cursor.fetchone()[0]

            connection.commit()

            print(
                "💾 Conversation saved to PostgreSQL."
            )

            print(
                f"   Question: {question}"
            )

            print(
                f"   PostgreSQL Row ID: {row_id}"
            )

        except psycopg.Error as e:

            connection.rollback()

            print(
                f"❌ PostgreSQL conversation save error: {e}"
            )

        finally:

            if cursor is not None:
                cursor.close()

            connection.close()

    # --------------------------------------------------------
    # IN-MEMORY HISTORY
    # --------------------------------------------------------
    conversation_history.append(
        {
            "question": question,
            "answer": clean_history_answer,
            "closing": closing,
            "resolved_question": resolved_question,
        }
    )

    # --------------------------------------------------------
    # KEEP ONLY RECENT HISTORY
    # --------------------------------------------------------
    if len(conversation_history) > MAX_HISTORY_TURNS:

        conversation_history = (
            conversation_history[
                -MAX_HISTORY_TURNS:
            ]
        )


# ============================================================
# FINALIZE ANSWER
# ============================================================
def finalize_answer(
    answer: str,
    question: str,
    intent: str = None,
    resolved_question: str = ""
):

    # --------------------------------------------------------
    # CLEAN ANSWER
    # --------------------------------------------------------
    answer = clean_answer(
        answer
    )

    if not answer:

        answer = (
            "Sorry, I couldn't find a clear "
            "answer to that question."
        )

    # --------------------------------------------------------
    # ONE CLOSING QUESTION ONLY
    # --------------------------------------------------------
    closing = get_closing_question(
        intent
    )

    final_answer = (
        f"{answer} {closing}"
    )

    # --------------------------------------------------------
    # SAVE CONVERSATION
    # --------------------------------------------------------
    save_conversation(
        question=question,
        answer=answer,
        closing=closing,
        resolved_question=resolved_question
    )

    return final_answer


# ============================================================
# ASK CHATBOT
# ============================================================
def ask_chatbot(question: str) -> str:

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------
    if not question or not question.strip():

        return (
            "Please ask me a question "
            "about DesFlyer."
        )

    # --------------------------------------------------------
    # ORIGINAL QUESTION
    # --------------------------------------------------------
    original_question = question.strip()

    normalized_question = normalize_text(
        original_question
    )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------
    if is_greeting(
        normalized_question
    ):

        return get_greeting_response(
            normalized_question
        )

    # --------------------------------------------------------
    # RESOLVE FOLLOW-UP
    # --------------------------------------------------------
    resolved_question = resolve_follow_up(
        original_question
    )

    resolved_normalized = normalize_text(
        resolved_question
    )

    print(
        f"\n🧠 Resolved Question: {resolved_question}"
    )

    # --------------------------------------------------------
    # VALIDATE DESFLYER QUESTION
    # --------------------------------------------------------
    is_contextual_follow_up = (

        is_follow_up(
            original_question
        )

        or (

            conversation_history

            and len(
                normalized_question.split()
            ) <= 10

            and any(

                phrase in normalized_question

                for phrase in [

                    "what platform",
                    "which platform",
                    "what about",
                    "which one",
                    "what type",
                    "how about",
                    "does support",
                    "do support",
                    "can support",

                ]
            )
        )
    )

    if (
        not is_contextual_follow_up
        and not is_valid_desflyer_question(
            original_question
        )
    ):

        print(
            "🚫 UNCLEAR OR INVALID DESFLYER QUESTION"
        )

        return (
            "I'm sorry, I couldn't understand your question. "
            "Please ask me about DesFlyer's services."
        )

    # --------------------------------------------------------
    # FAST FAQ
    # --------------------------------------------------------
    faq_answer = find_fast_faq(
        resolved_normalized
    )

    if faq_answer:

        faq_intent = get_faq_intent(
            resolved_normalized
        )

        print(
            "⚡ FAST FAQ ANSWER"
        )

        final_answer = finalize_answer(
            answer=faq_answer,
            question=original_question,
            intent=faq_intent,
            resolved_question=resolved_question
        )

        return final_answer

    # --------------------------------------------------------
    # STRICT DESFLYER RELEVANCE
    # --------------------------------------------------------
    desflyer_keywords = [

        "desflyer",
        "company",
        "service",
        "services",
        "website",
        "web development",
        "mobile",
        "application",
        "app",
        "software",
        "developer",
        "development",
        "research",
        "development team",
        "chatbot",

    ]

    is_relevant = is_valid_desflyer_question(
        resolved_question
    )

    # Follow-up questions are allowed
    if (
        is_follow_up(
            original_question
        )
        and conversation_history
    ):

        is_relevant = True

    # --------------------------------------------------------
    # IRRELEVANT QUESTION
    # --------------------------------------------------------
    if not is_relevant:

        print(
            "🚫 IRRELEVANT QUESTION"
        )

        return DESFLYER_FALLBACK

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------
    try:

        # IMPORTANT:
        # Only the resolved CURRENT question goes to ChromaDB.
        #
        # PostgreSQL history is NOT added to this query.
        #
        retrieval_query = (
            resolved_question.strip()
        )

        print(
            f"\n🔎 Retrieval Query: {retrieval_query}"
        )

        retrieved_documents = retriever.invoke(
            retrieval_query
        )

    except Exception as e:

        print(
            f"⚠️ Retrieval error: {e}"
        )

        return (
            "Sorry, I couldn't retrieve "
            "the information right now."
        )

    if not retrieved_documents:

        return (
            "Sorry, I couldn't find "
            "relevant information about DesFlyer."
        )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------
    context = create_context(
        retrieved_documents
    )

    if not context:

        return (
            "Sorry, I couldn't find "
            "relevant information about DesFlyer."
        )

    # --------------------------------------------------------
    # SHOW RETRIEVED DOCUMENTS
    # --------------------------------------------------------
    print(
        "\n📚 Retrieved Documents:"
    )

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        print(
            f"\n--- Document {index} ---"
        )

        print(
            document.page_content[:500]
        )

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------
    history = get_conversation_history_text()

    # --------------------------------------------------------
    # LLM GENERATION
    # --------------------------------------------------------
    answer = generate_answer(
        question=resolved_question,
        context=context,
        history=history
    )

    # --------------------------------------------------------
    # CHECK ANSWER
    # --------------------------------------------------------
    if not answer:

        return (
            "Sorry, I couldn't generate "
            "a clear answer right now."
        )

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------
    retrieved_intent = get_faq_intent(
        resolved_question
    )

    # --------------------------------------------------------
    # FINAL ANSWER + POSTGRESQL SAVE
    # --------------------------------------------------------
    final_answer = finalize_answer(
        answer=answer,
        question=original_question,
        intent=retrieved_intent,
        resolved_question=resolved_question
    )

    return final_answer


# ============================================================
# CLI TEST
# ============================================================
if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "DesFlyer RAG Chatbot"
    )

    print(
        "====================================\n"
    )

    while True:

        try:

            question = input(
                "You: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
                "bye"
            }:

                print(
                    "Goodbye!"
                )

                break

            answer = ask_chatbot(
                question
            )

            print(
                f"\nAssistant: {answer}\n"
            )

        except KeyboardInterrupt:

            print(
                "\nGoodbye!"
            )

            break

        except Exception as e:

            print(
                f"⚠️ Error: {e}"
            )
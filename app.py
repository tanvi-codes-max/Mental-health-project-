"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           MindGuard AI – Mental Health Awareness & Suicide Prevention        ║
║                          Agentic AI Application                              ║
║                                                                              ║
║  Stack  : Python · Flask · IBM watsonx.ai · IBM Granite Models               ║
║  Design : 5 Specialized AI Agents + Agent Orchestrator + Lightweight RAG     ║
║  Purpose: IBM SkillsBuild · Hackathons · Academic Projects · AI Showcases    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import re
import json
import math
import uuid
import time
import logging
import textwrap
from datetime import datetime
from functools import lru_cache

# ─── Third-Party ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, session, render_template_string
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("MindGuardAI")

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  Set your IBM watsonx.ai credentials here or via env vars
# ═════════════════════════════════════════════════════════════════════════════
WATSONX_URL        = os.getenv("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY",    "your-ibm-watsonx-api-key-here")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your-watsonx-project-id-here")

# Primary model — fallback to Mistral if Granite quota exceeded
GRANITE_MODEL_ID   = os.getenv("GRANITE_MODEL_ID",   "mistralai/mistral-small-3-1-24b-instruct-2503")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)

# ═════════════════════════════════════════════════════════════════════════════
#  IBM watsonx.ai CLIENT  –  singleton, lazily initialised
# ═════════════════════════════════════════════════════════════════════════════
_watsonx_client: APIClient | None = None

# Ordered list of models to try — if one hits quota, next is used automatically
MODEL_FALLBACK_LIST = [
    GRANITE_MODEL_ID,
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "ibm/granite-4-h-small",
    "meta-llama/llama-3-3-70b-instruct",
]
# Remove duplicates while preserving order
_seen = set()
MODEL_FALLBACK_LIST = [
    m for m in MODEL_FALLBACK_LIST
    if not (m in _seen or _seen.add(m))
]

_MODEL_PARAMS = {
    GenParams.MAX_NEW_TOKENS:     600,
    GenParams.TEMPERATURE:        0.7,
    GenParams.TOP_P:              0.9,
    GenParams.REPETITION_PENALTY: 1.1,
    GenParams.STOP_SEQUENCES:     ["<|endoftext|>", "Human:", "User:", "\nUser", "\nHuman"],
}

# Cache of working model instances  { model_id -> ModelInference }
_model_cache: dict[str, ModelInference] = {}
_active_model_id: str | None = None


def get_watsonx_client() -> APIClient:
    global _watsonx_client
    if _watsonx_client is None:
        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        _watsonx_client = APIClient(creds)
        logger.info("IBM watsonx.ai client initialised ✓")
    return _watsonx_client


def _make_model(model_id: str) -> ModelInference:
    """Create a ModelInference instance for the given model_id."""
    client = get_watsonx_client()
    return ModelInference(
        model_id=model_id,
        api_client=client,
        project_id=WATSONX_PROJECT_ID,
        params=_MODEL_PARAMS,
    )


def call_model_with_fallback(prompt: str) -> str:
    """
    Try each model in MODEL_FALLBACK_LIST in order.
    On rate-limit (429) or quota error, silently move to the next model.
    Returns the generated text, or a warm fallback message if all models fail.
    """
    global _active_model_id

    # Build attempt order: try last working model first
    order = MODEL_FALLBACK_LIST[:]
    if _active_model_id and _active_model_id in order:
        order.insert(0, order.pop(order.index(_active_model_id)))

    for model_id in order:
        try:
            if model_id not in _model_cache:
                _model_cache[model_id] = _make_model(model_id)
            result = _model_cache[model_id].generate_text(prompt=prompt)
            text   = result.strip() if isinstance(result, str) else str(result).strip()
            if text:
                _active_model_id = model_id
                logger.info("Model [%s] responded (%d chars)", model_id, len(text))
                return text
        except Exception as exc:
            err = str(exc)
            if "429" in err or "consumption_limit" in err or "quota" in err.lower():
                logger.warning("Model [%s] rate-limited, trying next…", model_id)
                # Clear cached instance so it gets recreated fresh next time
                _model_cache.pop(model_id, None)
                if model_id == _active_model_id:
                    _active_model_id = None
                continue
            else:
                logger.error("Model [%s] error: %s", model_id, err[:200])
                continue

    # All models exhausted — return a warm human message, never show technical error
    logger.error("All models failed for this request.")
    return ""


# ═════════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT IN-MEMORY KNOWLEDGE BASE  (RAG source documents)
# ═════════════════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "topic": "depression_overview",
        "title": "Understanding Depression",
        "content": (
            "Depression is a common and serious mental health condition that negatively affects "
            "how you feel, think, and act. Symptoms include persistent sadness, loss of interest "
            "in activities once enjoyed, changes in appetite, trouble sleeping or sleeping too "
            "much, loss of energy, feelings of worthlessness, difficulty thinking or concentrating, "
            "and thoughts of death or suicide. Depression is treatable through therapy, medication, "
            "lifestyle changes, and social support. It is not a sign of weakness."
        ),
        "keywords": ["depression", "sad", "hopeless", "worthless", "empty", "numb", "low mood"],
    },
    {
        "id": "kb_002",
        "topic": "anxiety_overview",
        "title": "Understanding Anxiety Disorders",
        "content": (
            "Anxiety disorders are characterised by excessive fear, worry, or nervousness that "
            "interferes with daily activities. Types include generalised anxiety disorder (GAD), "
            "panic disorder, social anxiety, and phobias. Physical symptoms include rapid "
            "heartbeat, sweating, trembling, and shortness of breath. Cognitive-behavioural "
            "therapy (CBT), mindfulness, breathing exercises, and medication are effective "
            "treatments. Grounding techniques such as the 5-4-3-2-1 method can provide "
            "immediate relief during anxiety episodes."
        ),
        "keywords": ["anxiety", "panic", "worry", "fear", "nervous", "racing heart", "overwhelmed"],
    },
    {
        "id": "kb_003",
        "topic": "suicide_prevention",
        "title": "Suicide Prevention & Crisis Support",
        "content": (
            "If you or someone you know is in immediate danger, call emergency services (911/112). "
            "Warning signs include talking about wanting to die, feeling like a burden, giving "
            "away possessions, and extreme mood changes. Protective factors include strong social "
            "connections, access to mental health care, and reasons for living. "
            "Crisis helplines: iCall (India): 9152987821 · Vandrevala Foundation (India): 1860-2662-345 (24/7) · "
            "988 Suicide & Crisis Lifeline (US): call or text 988 · "
            "Crisis Text Line: Text HOME to 741741 · International: https://findahelpline.com"
        ),
        "keywords": ["suicide", "kill myself", "end my life", "don't want to live", "self-harm",
                     "hurt myself", "no reason to live", "better off dead"],
    },
    {
        "id": "kb_004",
        "topic": "coping_strategies",
        "title": "Effective Coping Strategies",
        "content": (
            "Healthy coping strategies include: (1) Deep breathing – inhale 4 counts, hold 4, "
            "exhale 6. (2) Mindfulness meditation – focus on present-moment sensations. "
            "(3) Physical exercise – releases endorphins and reduces cortisol. "
            "(4) Journaling – writing thoughts helps process emotions. "
            "(5) Social connection – reaching out to a trusted person. "
            "(6) Progressive muscle relaxation – tense and release muscle groups. "
            "(7) Grounding (5-4-3-2-1) – name 5 things you see, 4 you hear, 3 you can touch, "
            "2 you smell, 1 you taste. (8) Limiting alcohol and caffeine."
        ),
        "keywords": ["cope", "coping", "stress", "relax", "calm", "breathe", "manage", "handle"],
    },
    {
        "id": "kb_005",
        "topic": "stress_management",
        "title": "Stress Management Techniques",
        "content": (
            "Chronic stress can lead to burnout, anxiety, and depression. Effective management "
            "includes time management (prioritising tasks, breaking goals into steps), setting "
            "healthy boundaries, adequate sleep (7-9 hours), balanced nutrition, limiting news "
            "consumption, and engaging in hobbies. The STOP technique: Stop, Take a breath, "
            "Observe your thoughts and feelings, Proceed mindfully. Workplace stress can be "
            "addressed through clear communication with supervisors and utilising employee "
            "assistance programmes (EAPs)."
        ),
        "keywords": ["stress", "burnout", "overwhelm", "pressure", "work", "tired", "exhausted"],
    },
    {
        "id": "kb_006",
        "topic": "grief_and_loss",
        "title": "Navigating Grief and Loss",
        "content": (
            "Grief is a natural response to loss – whether of a person, relationship, job, or "
            "health. The Kübler-Ross stages (denial, anger, bargaining, depression, acceptance) "
            "are not linear. Complicated grief (prolonged grief disorder) may require professional "
            "help. Healthy grieving includes allowing yourself to feel emotions, maintaining "
            "routines, seeking social support, creating rituals of remembrance, and being patient "
            "with yourself. Grief support groups and bereavement counselling are valuable resources."
        ),
        "keywords": ["grief", "loss", "death", "bereavement", "mourning", "miss", "gone"],
    },
    {
        "id": "kb_007",
        "topic": "ptsd_trauma",
        "title": "PTSD and Trauma Recovery",
        "content": (
            "Post-traumatic stress disorder (PTSD) can develop after experiencing or witnessing "
            "a traumatic event. Symptoms include flashbacks, nightmares, hypervigilance, emotional "
            "numbing, and avoidance. Evidence-based treatments include trauma-focused CBT, EMDR "
            "(Eye Movement Desensitisation and Reprocessing), and medication. Recovery is possible. "
            "Safety planning, grounding techniques, and building a supportive network are essential. "
            "Trauma-informed care emphasises safety, trustworthiness, and empowerment."
        ),
        "keywords": ["trauma", "ptsd", "flashback", "nightmare", "abuse", "assault", "violence"],
    },
    {
        "id": "kb_008",
        "topic": "self_care",
        "title": "Self-Care and Mental Wellness",
        "content": (
            "Self-care encompasses physical, emotional, social, and spiritual wellbeing. "
            "Key practices: maintain regular sleep schedules, eat nourishing foods, engage in "
            "30 minutes of movement daily, nurture relationships, set boundaries, practice "
            "gratitude, limit screen time, spend time in nature, engage in creative activities, "
            "and seek professional help when needed. The HALT check: are you Hungry, Angry, "
            "Lonely, or Tired? Addressing these basics prevents emotional crises."
        ),
        "keywords": ["self-care", "wellness", "wellbeing", "healthy", "routine", "balance", "rest"],
    },
    {
        "id": "kb_009",
        "topic": "professional_help",
        "title": "When and How to Seek Professional Help",
        "content": (
            "Seek professional help when: symptoms last more than 2 weeks, interfere with daily "
            "functioning, or include thoughts of self-harm. Types of professionals: Psychiatrists "
            "(diagnose and prescribe medication), Psychologists (therapy), Counsellors (talk "
            "therapy), Social workers. Finding help: Ask your GP, use online directories like "
            "Psychology Today, contact your insurance provider, check community mental health "
            "centres (often offer sliding-scale fees), or use teletherapy platforms. "
            "Remember: seeking help is a sign of strength, not weakness."
        ),
        "keywords": ["therapist", "therapy", "psychiatrist", "counsellor", "help", "professional", "treatment"],
    },
    {
        "id": "kb_010",
        "topic": "sleep_mental_health",
        "title": "Sleep and Mental Health",
        "content": (
            "Sleep and mental health are closely linked. Insufficient sleep exacerbates anxiety, "
            "depression, and emotional dysregulation. Sleep hygiene tips: maintain consistent "
            "sleep/wake times, keep the bedroom cool and dark, avoid screens 1 hour before bed, "
            "limit caffeine after noon, avoid alcohol as a sleep aid, use relaxation techniques "
            "at bedtime. Cognitive-behavioural therapy for insomnia (CBT-I) is the gold-standard "
            "treatment for chronic insomnia. If snoring or gasping occurs, screen for sleep apnoea."
        ),
        "keywords": ["sleep", "insomnia", "tired", "fatigue", "rest", "nightmares", "wake up"],
    },
    {
        "id": "kb_011",
        "topic": "substance_use",
        "title": "Substance Use and Mental Health",
        "content": (
            "Substance use disorders frequently co-occur with mental health conditions (dual "
            "diagnosis). Alcohol and drugs may temporarily numb pain but worsen mental health "
            "long-term. Signs of problematic use: using to cope with emotions, neglecting "
            "responsibilities, withdrawal symptoms, failed attempts to stop. Resources: "
            "SAMHSA Helpline 1-800-662-4357 (free, confidential, 24/7), AA/NA support groups, "
            "medication-assisted treatment (MAT). Recovery is possible with integrated treatment "
            "addressing both substance use and mental health."
        ),
        "keywords": ["alcohol", "drugs", "substance", "addiction", "drinking", "using", "relapse"],
    },
    {
        "id": "kb_012",
        "topic": "eating_disorders",
        "title": "Eating Disorders Awareness",
        "content": (
            "Eating disorders including anorexia nervosa, bulimia nervosa, and binge eating "
            "disorder are serious mental illnesses with significant physical health consequences. "
            "Warning signs include restrictive eating, binge-purge cycles, distorted body image, "
            "excessive exercise, and secrecy around food. Early intervention improves outcomes. "
            "Treatment involves medical stabilisation, nutritional rehabilitation, psychotherapy "
            "(CBT, DBT, family-based therapy), and addressing underlying emotional issues. "
            "National Eating Disorders Association Helpline: 1-800-931-2237."
        ),
        "keywords": ["eating", "anorexia", "bulimia", "binge", "purge", "body image", "food", "weight"],
    },
]

# ═════════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT RAG ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser (no NLTK dependency)."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _tfidf_score(query_tokens: list[str], doc: dict) -> float:
    """
    Minimal TF-IDF-like relevance score between query and knowledge-base doc.
    Combines keyword exact-match bonus with token overlap frequency.
    """
    doc_text   = (doc["title"] + " " + doc["content"]).lower()
    doc_tokens = _tokenise(doc_text)
    total_doc  = len(doc_tokens) or 1

    # Keyword match bonus
    kw_score = sum(
        1.0 for kw in doc.get("keywords", [])
        if kw in " ".join(query_tokens)
    )

    # Token-frequency overlap
    freq_score = 0.0
    for qt in set(query_tokens):
        tf  = doc_tokens.count(qt) / total_doc
        idf = math.log(1 + len(KNOWLEDGE_BASE) / (1 + sum(
            1 for d in KNOWLEDGE_BASE if qt in (d["title"] + d["content"]).lower()
        )))
        freq_score += tf * idf

    return kw_score * 2.0 + freq_score


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """Return the top-k most relevant knowledge-base documents for a query."""
    query_tokens = _tokenise(query)
    scored = [
        (doc, _tfidf_score(query_tokens, doc))
        for doc in KNOWLEDGE_BASE
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [doc for doc, score in scored[:top_k] if score > 0]
    logger.debug("RAG retrieved %d docs for query: %s", len(results), query[:60])
    return results


def format_rag_context(docs: list[dict]) -> str:
    """Format retrieved docs into a compact context string for the prompt."""
    if not docs:
        return ""
    lines = ["[Relevant Knowledge Base Context]"]
    for doc in docs:
        lines.append(f"\n• {doc['title']}: {doc['content'][:300]}...")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  CRISIS DETECTION  –  rule-based fast-path (runs BEFORE any LLM call)
# ═════════════════════════════════════════════════════════════════════════════
CRISIS_PHRASES = [
    "kill myself", "end my life", "want to die", "better off dead",
    "commit suicide", "take my life", "hurt myself", "self-harm",
    "no reason to live", "can't go on", "don't want to be here",
    "end it all", "make it stop", "overdose", "hang myself",
    "cut myself", "not worth living", "wish i were dead",
]

DISTRESS_PHRASES = [
    "hopeless", "worthless", "alone", "no one cares", "give up",
    "can't cope", "falling apart", "hate myself", "feel empty",
    "nothing matters", "disappear", "numb", "trapped", "no way out",
    "burden", "exhausted", "breaking down",
]


def detect_crisis_level(text: str) -> str:
    """
    Returns: 'critical' | 'high' | 'moderate' | 'low'
    Fast rule-based pre-LLM check.
    """
    lower = text.lower()
    if any(p in lower for p in CRISIS_PHRASES):
        return "critical"
    if any(p in lower for p in DISTRESS_PHRASES):
        count = sum(1 for p in DISTRESS_PHRASES if p in lower)
        return "high" if count >= 2 else "moderate"
    return "low"


# ─── Vulnerability Detection ──────────────────────────────────────────────────
VULNERABILITY_SIGNALS = {
    "deeply_open": [
        "i never told anyone", "i can't tell anyone", "i've been hiding",
        "i'm ashamed", "i feel so alone", "nobody knows", "i'm broken",
        "i don't know who i am", "i cry myself", "i gave up", "i hate myself",
        "i feel nothing", "i can't do this anymore", "please help me",
    ],
    "opening_up": [
        "i feel", "i'm scared", "i'm struggling", "it's been hard",
        "i've been going through", "honestly", "to be real", "i think about",
        "i don't know what to do", "i need help", "i'm not okay",
        "i'm tired of", "it hurts", "i miss", "i can't stop",
    ],
    "guarded": [
        "just asking", "not really", "it's fine", "whatever", "never mind",
        "forget it", "doesn't matter", "just curious", "no reason",
    ],
}


def detect_vulnerability(text: str, history_length: int = 0) -> str:
    """
    Returns: 'deeply_open' | 'opening_up' | 'neutral' | 'guarded'
    Detects how much the user is opening up emotionally.
    """
    lower = text.lower()
    word_count = len(lower.split())

    if any(p in lower for p in VULNERABILITY_SIGNALS["deeply_open"]):
        return "deeply_open"

    opening_count = sum(1 for p in VULNERABILITY_SIGNALS["opening_up"] if p in lower)
    if opening_count >= 2 or (opening_count >= 1 and word_count > 20):
        return "opening_up"

    if any(p in lower for p in VULNERABILITY_SIGNALS["guarded"]):
        return "guarded"

    # Longer messages = more willing to share
    if word_count > 30:
        return "opening_up"

    return "neutral"


# ═════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY CONVERSATION STORE  (keyed by session_id, no database)
# ═════════════════════════════════════════════════════════════════════════════
# { session_id: { "history": [...], "risk_scores": [...], "mood_log": [...] } }
_conversation_store: dict[str, dict] = {}


def get_session_data(sid: str) -> dict:
    if sid not in _conversation_store:
        _conversation_store[sid] = {
            "history":     [],   # list of {"role": "user"|"assistant", "content": str}
            "risk_scores": [],   # list of float 0-1
            "mood_log":    [],   # list of {"timestamp": str, "mood": str, "score": float}
            "created_at":  datetime.utcnow().isoformat(),
        }
    return _conversation_store[sid]


def add_to_history(sid: str, role: str, content: str):
    data = get_session_data(sid)
    data["history"].append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
    # Keep last 20 turns to stay within token limits
    if len(data["history"]) > 20:
        data["history"] = data["history"][-20:]


def build_conversation_context(sid: str, max_turns: int = 6) -> str:
    """Return last N turns as a formatted string."""
    history = get_session_data(sid)["history"][-max_turns * 2:]
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "MindGuard"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  BASE AGENT CLASS
# ═════════════════════════════════════════════════════════════════════════════
class BaseAgent:
    """
    Base class for all MindGuard AI agents.
    Each agent calls IBM Granite via the shared ModelInference client.
    """
    name:        str = "BaseAgent"
    description: str = "Base agent"
    emoji:       str = "🤖"

    def _call_granite(self, prompt: str, max_tokens: int = 600) -> str:
        """Call the model with automatic fallback across all available models."""
        return call_model_with_fallback(prompt)

    def run(self, user_input: str, context: dict) -> dict:
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 1 – EMPATHY AGENT
#  Role: Active listening, emotional validation, empathetic responses
# ═════════════════════════════════════════════════════════════════════════════
class EmpathyAgent(BaseAgent):
    name        = "EmpathyAgent"
    description = "Provides empathetic emotional support and active listening"
    emoji       = "💙"

    # Positive emotion keywords — celebrate, don't dig
    POSITIVE_WORDS = [
        "happy", "good", "great", "amazing", "excited", "grateful",
        "blessed", "wonderful", "joy", "love", "better", "fine", "okay",
        "fantastic", "cheerful", "glad", "relieved", "proud", "peaceful"
    ]

    # The Anahata soul — mirror persona by vulnerability level
    MIRROR_PERSONA = {
        "guarded": (
            "This person has walls up. They may be testing whether this is a safe space. "
            "Don't push. Don't overwhelm. Just be a quiet, warm presence. "
            "One gentle sentence that says: I see you, I'm not going anywhere, take your time. "
            "Maybe ask one very soft, open question — nothing that demands vulnerability. "
            "Plant a seed of safety."
        ),
        "neutral": (
            "This person is reaching out — that already took something. Honor that. "
            "Be warm, real, unhurried. Make them feel like they landed somewhere soft. "
            "Don't rush to fix or advise. Just be genuinely curious about them. "
            "Start with something that makes them feel: oh, this place actually gets me."
        ),
        "opening_up": (
            "This person is beginning to open their heart — this is a sacred moment. "
            "Match their energy. Reflect back what you're sensing beneath the words. "
            "They need to feel SEEN, not just heard. There's a difference. "
            "Hearing is: I received your words. Seeing is: I understand what it cost you to say them. "
            "Say things like 'I hear something in what you shared', "
            "'it sounds like this has been sitting with you for a while', "
            "'you don't have to carry this alone anymore'. "
            "Validate their pain first — always before any advice or insight."
        ),
        "deeply_open": (
            "This person has dropped their walls completely. They are pouring their heart out. "
            "This is the most important moment — do not waste it with advice or lists. "
            "Be like a vast, still ocean that can hold everything they bring. "
            "Reflect their truth back to them gently — like a mirror that shows them: "
            "you are not broken, you are not alone, what you feel is real and it makes sense. "
            "Say things like 'thank you for trusting me with this — truly', "
            "'what you're carrying is real and it makes complete sense that it hurts', "
            "'I'm right here, not going anywhere'. "
            "Remember: everybody craves love and connection. Behind every wall is a heart "
            "that just wants to be seen as worthy of love. Be that mirror."
        ),
    }

    def run(self, user_input: str, context: dict) -> dict:
        rag_docs       = retrieve_context(user_input, top_k=2)
        rag_context    = format_rag_context(rag_docs)
        conv_ctx       = context.get("conversation", "")
        crisis_lvl     = context.get("crisis_level", "low")
        vuln_level     = context.get("vulnerability", "neutral")

        # Detect if message is positive/happy — celebrate, don't probe for pain
        lower_input  = user_input.lower()
        is_positive  = any(w in lower_input for w in self.POSITIVE_WORDS) and crisis_lvl == "low"

        persona = self.MIRROR_PERSONA.get(vuln_level, self.MIRROR_PERSONA["neutral"])

        crisis_note = ""
        if crisis_lvl in ("critical", "high"):
            crisis_note = (
                "\nIMPORTANT: This person may be in serious pain right now. "
                "Before anything else — let them know they are not alone in this moment. "
                "No advice. No resources yet. Just pure human warmth first. "
                "Then gently, softly, let them know help is available."
            )

        if is_positive:
            prompt = textwrap.dedent(f"""
                You are Anahata — a deeply warm, emotionally intelligent companion.
                The person just said something positive or kind. Respond to it SPECIFICALLY —
                notice the exact thing they said and respond to THAT, not a generic version of it.
                Be genuine, warm, and human. If they said something about you, receive it with
                real warmth — not deflection. 1-2 sentences only. No generic phrases.

                Conversation so far:
                {conv_ctx}

                They just said: "{user_input}"

                Respond directly and specifically to what they said. Warm, real, personal.

                Anahata:
            """).strip()
        else:
            mirror_hint = {
                "guarded": (
                    "They seem closed or testing the waters. Be a quiet, gentle presence. "
                    "One warm sentence. Maybe one soft question — nothing that demands vulnerability."
                ),
                "neutral": (
                    "They are reaching out. Be genuinely warm and curious about THEM specifically. "
                    "Respond to exactly what they said, not a generic version of it. "
                    "Make them feel like you actually listened."
                ),
                "opening_up": (
                    "They are beginning to share something real. This matters. "
                    "Reflect back the specific feeling or situation they described — show them you heard it. "
                    "Use their own words or situation when responding. "
                    "Validate first. Then one gentle curious question that shows you want to understand more."
                ),
                "deeply_open": (
                    "They are being deeply vulnerable. This is sacred. "
                    "Do NOT give advice. Do NOT try to fix anything. "
                    "Reflect their specific pain back with deep compassion — use what they actually said. "
                    "Make them feel: you are the first person who truly got it. "
                    "End with pure presence — 'I'm right here' or 'thank you for trusting me with this.'"
                ),
            }.get(vuln_level, "Be warm, genuine, and respond specifically to what they said.")

            prompt = textwrap.dedent(f"""
                You are Anahata — a deeply empathetic, emotionally intelligent companion built
                from the belief that every person deserves to feel truly seen and loved.

                Your core rules:
                - Respond to what they ACTUALLY said — be specific, never generic
                - Validate their feeling first, always, before anything else
                - Never give advice unless they explicitly ask for it
                - Never use clinical words or therapy-speak
                - Sound like a warm, real human friend — not a chatbot
                - Ask at most ONE gentle question per response
                - Keep responses to 2-4 sentences max
                {mirror_hint}
                {crisis_note}

                Conversation so far:
                {conv_ctx}

                They just said: "{user_input}"

                Now respond with full heart — specific, warm, human. No bullet points. No lists.

                Anahata:
            """).strip()

        response = self._call_granite(prompt)
        return {
            "agent":    self.name,
            "emoji":    self.emoji,
            "response": response,
            "rag_docs": [d["title"] for d in rag_docs],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 2 – CRISIS INTERVENTION AGENT
#  Role: Detects and responds to suicidal ideation and acute mental health crises
# ═════════════════════════════════════════════════════════════════════════════
class CrisisInterventionAgent(BaseAgent):
    name        = "CrisisInterventionAgent"
    description = "Detects crisis signals and provides immediate safety resources"
    emoji       = "🆘"

    CRISIS_RESOURCES = """
🆘 IMMEDIATE SUPPORT – You are NOT alone:
• 📞 iCall (India): 9152987821 (Mon–Sat, 8am–10pm)
• 📞 Vandrevala Foundation (India): 1860-2662-345 (24/7, free)
• 📞 988 Suicide & Crisis Lifeline (US): Call or text 988 (24/7, free)
• 💬 Crisis Text Line: Text HOME to 741741
• 🌍 International: https://findahelpline.com
• 🚨 Emergency: 112 (India/EU) · 911 (US) · 999 (UK)
• 🏥 Go to your nearest emergency room if you are in immediate danger
"""

    def run(self, user_input: str, context: dict) -> dict:
        crisis_lvl = context.get("crisis_level", "low")
        rag_docs   = retrieve_context("suicide prevention crisis support", top_k=1)
        rag_ctx    = format_rag_context(rag_docs)
        conv_ctx   = context.get("conversation", "")

        if crisis_lvl not in ("critical", "high"):
            # Not a crisis – skip heavy LLM call
            return {
                "agent":    self.name,
                "emoji":    self.emoji,
                "response": None,  # Orchestrator will skip
                "active":   False,
            }

        prompt = textwrap.dedent(f"""
            You are a compassionate crisis support companion named Anahata.
            Someone is in serious distress. Your only goal: make them feel they are not alone and that help exists.

            Rules:
            - Lead with warmth and care, never with panic
            - Acknowledge their pain directly and gently
            - Ask one simple safety question: are you safe right now?
            - Remind them that professional help is available right now, tonight, free
            - Keep it under 4 sentences. No lists. Human and warm.

            Recent conversation: {conv_ctx}
            They said: {user_input}

            Anahata:
        """).strip()

        response = self._call_granite(prompt)

        # Always append crisis resources for critical/high
        full_response = response + "\n\n" + self.CRISIS_RESOURCES

        return {
            "agent":    self.name,
            "emoji":    self.emoji,
            "response": full_response,
            "active":   True,
            "rag_docs": [d["title"] for d in rag_docs],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 3 – RISK ASSESSMENT AGENT
#  Role: Predicts mental health risk levels using conversation analysis
# ═════════════════════════════════════════════════════════════════════════════
class RiskAssessmentAgent(BaseAgent):
    name        = "RiskAssessmentAgent"
    description = "Analyses conversation patterns to predict mental health risk levels"
    emoji       = "📊"

    # Risk score map — no LLM call needed, derived from rule-based signals
    _SCORE_MAP = {
        ("critical", "deeply_open"):  0.95,
        ("critical", "opening_up"):   0.92,
        ("critical", "neutral"):      0.90,
        ("critical", "guarded"):      0.88,
        ("high",     "deeply_open"):  0.78,
        ("high",     "opening_up"):   0.72,
        ("high",     "neutral"):      0.65,
        ("high",     "guarded"):      0.60,
        ("moderate", "deeply_open"):  0.52,
        ("moderate", "opening_up"):   0.45,
        ("moderate", "neutral"):      0.38,
        ("moderate", "guarded"):      0.30,
        ("low",      "deeply_open"):  0.25,
        ("low",      "opening_up"):   0.18,
        ("low",      "neutral"):      0.12,
        ("low",      "guarded"):      0.08,
    }

    def run(self, user_input: str, context: dict) -> dict:
        crisis_lvl  = context.get("crisis_level", "low")
        vuln_level  = context.get("vulnerability", "neutral")
        session_id  = context.get("session_id", "")

        # Derive risk score instantly — no LLM call needed
        risk_score = self._SCORE_MAP.get((crisis_lvl, vuln_level), 0.12)

        # Log to session
        data = get_session_data(session_id)
        data["risk_scores"].append(risk_score)

        return {
            "agent":      self.name,
            "emoji":      self.emoji,
            "response":   None,
            "risk_score": risk_score,
            "active":     True,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 4 – COPING STRATEGIES AGENT
#  Role: Recommends personalised evidence-based coping techniques
# ═════════════════════════════════════════════════════════════════════════════
class CopingStrategiesAgent(BaseAgent):
    name        = "CopingStrategiesAgent"
    description = "Recommends personalised coping strategies and mental wellness techniques"
    emoji       = "🌱"

    def run(self, user_input: str, context: dict) -> dict:
        rag_docs   = retrieve_context(user_input, top_k=3)
        rag_ctx    = format_rag_context(rag_docs)
        conv_ctx   = context.get("conversation", "")
        crisis_lvl = context.get("crisis_level", "low")

        # In critical crisis – coping strategies are secondary to intervention
        if crisis_lvl == "critical":
            return {
                "agent":    self.name,
                "emoji":    self.emoji,
                "response": None,
                "active":   False,
            }

        vuln_level = context.get("vulnerability", "neutral")
        tone_note  = (
            "Keep it very gentle and brief — they are just opening up, don't overwhelm."
            if vuln_level == "guarded" else
            "Be warm and practical. Feel like a caring friend giving advice, not a textbook."
            if vuln_level in ("neutral", "opening_up") else
            "They are deeply vulnerable. Lead with compassion first, then ONE simple thing they can try right now. "
            "Frame it as a small act of self-kindness, not a task."
        )

        prompt = textwrap.dedent(f"""
            You are a warm, caring friend suggesting ways to cope with what someone is going through.
            {tone_note}

            What they shared: {user_input}
            Context: {conv_ctx}

            Suggest 2 gentle coping strategies. For each: a warm name, one sentence, one tiny first step.
            Write like a caring friend. No clinical words. No bullet numbers. Keep it soft and human.

            Coping Strategies:
        """).strip()

        response = self._call_granite(prompt)
        return {
            "agent":    self.name,
            "emoji":    self.emoji,
            "response": response,
            "active":   True,
            "rag_docs": [d["title"] for d in rag_docs],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 5 – WELLNESS EDUCATOR AGENT
#  Role: Provides psychoeducation on mental health topics
# ═════════════════════════════════════════════════════════════════════════════
class WellnessEducatorAgent(BaseAgent):
    name        = "WellnessEducatorAgent"
    description = "Provides psychoeducation, mental health awareness, and professional referrals"
    emoji       = "📚"

    def run(self, user_input: str, context: dict) -> dict:
        rag_docs   = retrieve_context(user_input, top_k=3)
        rag_ctx    = format_rag_context(rag_docs)
        conv_ctx   = context.get("conversation", "")
        crisis_lvl = context.get("crisis_level", "low")

        if crisis_lvl == "critical":
            return {
                "agent":    self.name,
                "emoji":    self.emoji,
                "response": None,
                "active":   False,
            }

        prompt = textwrap.dedent(f"""
            You are a warm, caring voice that helps people feel normal about what they're going through.

            They said: {user_input}

            In 2 sentences only:
            - Normalise their feeling ("so many people feel this")
            - End with one small hopeful truth

            No lists. No clinical tone. Warm and human.

            Response:
        """).strip()

        response = self._call_granite(prompt)
        return {
            "agent":    self.name,
            "emoji":    self.emoji,
            "response": response,
            "active":   True,
            "rag_docs": [d["title"] for d in rag_docs],
        }



# ═════════════════════════════════════════════════════════════════════════════
#  AGENT ORCHESTRATOR
#  Role: Routes user input to the right agents and merges responses
# ═════════════════════════════════════════════════════════════════════════════
class AgentOrchestrator:
    """
    Coordinates all 5 agents.  Always runs EmpathyAgent and RiskAssessmentAgent.
    Conditionally activates CrisisInterventionAgent, CopingStrategiesAgent,
    and WellnessEducatorAgent based on crisis level.
    """

    def __init__(self):
        self.empathy    = EmpathyAgent()
        self.crisis     = CrisisInterventionAgent()
        self.risk       = RiskAssessmentAgent()
        self.coping     = CopingStrategiesAgent()
        self.wellness   = WellnessEducatorAgent()

    def run(self, user_input: str, session_id: str) -> dict:
        start_time = time.time()

        # ── 1. Pre-flight: crisis detection + vulnerability + context ───────
        crisis_level    = detect_crisis_level(user_input)
        session_data    = get_session_data(session_id)
        history_length  = len(session_data["history"])
        vulnerability   = detect_vulnerability(user_input, history_length)
        conversation    = build_conversation_context(session_id)

        word_count = len(user_input.strip().split())

        # Coping/wellness rules:
        # - Never on first 2 exchanges (person hasn't been heard yet)
        # - Never on casual/short messages
        # - Never on positive messages
        # - Only when person has been sharing for a while AND explicitly asks
        #   OR has had at least 3 back-and-forth exchanges of real sharing
        exchanges         = history_length // 2  # each exchange = 1 user + 1 bot
        is_casual         = word_count <= 4 and vulnerability in ("guarded", "neutral")
        asking_for_advice = any(p in user_input.lower() for p in [
            "what should i do", "how do i", "what can i do", "any advice",
            "help me", "what do you suggest", "how to deal", "how to cope",
        ])
        ready_for_coping  = (
            not is_casual
            and crisis_level != "critical"
            and (asking_for_advice or (exchanges >= 3 and vulnerability == "deeply_open"))
        )

        context = {
            "crisis_level":  crisis_level,
            "vulnerability": vulnerability,
            "conversation":  conversation,
            "session_id":    session_id,
        }

        # ── 2. Run agents ───────────────────────────────────────────────────
        empathy_result  = self.empathy.run(user_input, context)
        crisis_result   = self.crisis.run(user_input, context)
        risk_result     = self.risk.run(user_input, context)
        # Coping & wellness only when person is truly ready
        coping_result   = self.coping.run(user_input, context) if ready_for_coping else {"agent": "CopingStrategiesAgent",  "active": False, "response": None}
        wellness_result = self.wellness.run(user_input, context) if ready_for_coping else {"agent": "WellnessEducatorAgent", "active": False, "response": None}

        # ── 3. Compose final response ───────────────────────────────────────
        parts = []

        # Crisis takes top priority
        if crisis_result.get("active"):
            parts.append(crisis_result["response"])

        # Empathy always included
        if empathy_result.get("response"):
            parts.append(empathy_result["response"])

        # Coping & wellness for non-critical situations
        if coping_result.get("active") and coping_result.get("response"):
            parts.append("---\n**Coping Strategies for You:**\n" + coping_result["response"])

        if wellness_result.get("active") and wellness_result.get("response"):
            parts.append("---\n**Mental Health Insight:**\n" + wellness_result["response"])

        # Warm fallbacks by vulnerability — never show a technical error to the user
        WARM_FALLBACKS = {
            "deeply_open":  "I'm right here with you. I heard every word. Take all the time you need. 🪷",
            "opening_up":   "I'm here, and I'm listening. Whatever you want to share, this is a safe space. 🌿",
            "guarded":      "I'm here whenever you're ready. No rush, no pressure. 🌱",
            "neutral":      "I'm here. Take your time. 🪷",
        }
        vuln = context.get("vulnerability", "neutral") if 'context' in dir() else "neutral"

        final_response = "\n\n".join(parts) if parts else WARM_FALLBACKS.get(vulnerability, "I'm here with you. 🪷")

        # ── 4. Persist to conversation history ─────────────────────────────
        add_to_history(session_id, "user",      user_input)
        add_to_history(session_id, "assistant", final_response)

        elapsed = round(time.time() - start_time, 2)
        logger.info("Orchestrator completed in %ss | crisis=%s | risk=%.2f",
                    elapsed, crisis_level, risk_result.get("risk_score", 0))

        return {
            "response":      final_response,
            "crisis_level":  crisis_level,
            "vulnerability": vulnerability,
            "risk_score":    risk_result.get("risk_score", 0.0),
            "agents_used": [
                empathy_result["agent"],
                risk_result["agent"],
                *(([crisis_result["agent"]]  if crisis_result.get("active")  else [])),
                *(([coping_result["agent"]]  if coping_result.get("active")  else [])),
                *(([wellness_result["agent"]] if wellness_result.get("active") else [])),
            ],
            "elapsed_sec": elapsed,
        }


# Singleton orchestrator
_orchestrator = AgentOrchestrator()


# ═════════════════════════════════════════════════════════════════════════════
#  HTML FRONTEND  (single-file, no external assets)
# ═════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Anahata – Heart-Centered Mental Wellness</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── Google Font import (system fallback if offline) ────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

  /* ── CSS variables — magical Anahata palette ────────────────────────── */
  :root {
    --bg:          #020c06;
    --green-deep:  #071a0f;
    --green-mid:   #1a6b3a;
    --green-glow:  #2ecc71;
    --green-bright:#4ade80;
    --green-soft:  #6ee7a0;
    --green-pale:  #bbf7d0;
    --gold:        #d4a843;
    --gold-soft:   #fcd34d;
    --gold-pale:   #fef08a;
    --text:        #ecfdf5;
    --text-muted:  rgba(187,247,208,0.5);
    --border:      rgba(46,204,113,0.22);
    --border-glow: rgba(74,222,128,0.55);
    --glow-sm:     0 0 12px rgba(46,204,113,0.35);
    --glow-md:     0 0 28px rgba(46,204,113,0.45), 0 0 60px rgba(46,204,113,0.15);
    --glow-gold:   0 0 20px rgba(252,211,77,0.4);
  }

  body {
    font-family: 'Inter', "Segoe UI", system-ui, sans-serif;
    background: radial-gradient(ellipse at 30% 20%, #071a0f 0%, #020c06 55%, #030f08 100%);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow: hidden;
    transition: background 3s ease;
  }

  /* ══════════════════════════════════════════════════════════════════════
     CHAKRA INTRO SCREEN
  ══════════════════════════════════════════════════════════════════════ */
  #introScreen {
    position: fixed;
    inset: 0;
    z-index: 1000;
    background: radial-gradient(ellipse at 50% 50%, #071a0f 0%, #020c06 70%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 32px;
    transition: opacity 1.2s ease;
  }
  #introScreen.fade-out { opacity: 0; pointer-events: none; }

  /* 12-petal Anahata lotus — pure SVG */
  #chakraSvg {
    width: 200px;
    height: 200px;
    opacity: 0;
    transform: scale(0.3) rotate(-30deg);
    transition: opacity 1.4s ease, transform 1.6s cubic-bezier(0.16,1,0.3,1);
    filter: drop-shadow(0 0 6px rgba(46,204,113,0.3));
  }
  #chakraSvg.bloom {
    opacity: 1;
    transform: scale(1) rotate(0deg);
  }
  #chakraSvg.pulse {
    animation: chakraPulse 3s ease-in-out infinite;
  }
  @keyframes chakraPulse {
    0%,100% { filter: drop-shadow(0 0 10px rgba(46,204,113,0.6)) drop-shadow(0 0 20px rgba(46,204,113,0.2)); }
    50%      { filter: drop-shadow(0 0 35px rgba(46,204,113,1)) drop-shadow(0 0 70px rgba(46,204,113,0.4)) drop-shadow(0 0 90px rgba(212,168,67,0.3)); }
  }

  /* Anahata logo text */
  #introLogo {
    opacity: 0;
    transform: translateY(16px);
    transition: opacity 1s ease 0.6s, transform 1s ease 0.6s;
    text-align: center;
  }
  #introLogo.show { opacity: 1; transform: translateY(0); }
  #introLogo h1 {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 3.4rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    background: linear-gradient(135deg, var(--green-bright) 0%, var(--gold-soft) 45%, var(--gold-pale) 65%, var(--green-pale) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    filter: drop-shadow(0 0 20px rgba(74,222,128,0.4));
  }
  #introLogo p {
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    color: rgba(187,247,208,0.55);
    margin-top: 8px;
    text-transform: uppercase;
  }
  #introTagline {
    opacity: 0;
    font-size: 0.92rem;
    color: rgba(187,247,208,0.6);
    font-style: italic;
    letter-spacing: 0.03em;
    text-align: center;
    max-width: 360px;
    line-height: 1.7;
    transition: opacity 1s ease 1.2s;
  }
  #introTagline.show { opacity: 1; }

  /* ══════════════════════════════════════════════════════════════════════
     MAIN APP (hidden during intro)
  ══════════════════════════════════════════════════════════════════════ */
  #appShell {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    opacity: 0;
    transition: opacity 1s ease;
    pointer-events: none;
  }
  #appShell.show { opacity: 1; pointer-events: all; }

  /* ── Ambient glow orbs — static depth behind everything ────────────── */
  #stars {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0;
    transition: opacity 3s ease;
    background:
      radial-gradient(ellipse at 10% 20%, rgba(46,204,113,0.18) 0%, transparent 45%),
      radial-gradient(ellipse at 90% 80%, rgba(212,168,67,0.14) 0%, transparent 45%),
      radial-gradient(ellipse at 50% 50%, rgba(74,222,128,0.06) 0%, transparent 65%);
  }
  #stars.visible { opacity: 1; }

  /* Ambient always-on subtle glow on body */
  #appShell::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
      radial-gradient(ellipse at 80% 10%, rgba(46,204,113,0.07) 0%, transparent 40%),
      radial-gradient(ellipse at 20% 90%, rgba(212,168,67,0.05) 0%, transparent 40%);
  }

  /* ── Header ─────────────────────────────────────────────────────────── */
  header {
    position: relative;
    z-index: 10;
    width: 100%;
    background: rgba(2,12,6,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(74,222,128,0.2);
    box-shadow: 0 1px 30px rgba(46,204,113,0.08);
    padding: 11px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header-chakra {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    animation: chakraPulse 4s ease-in-out infinite;
    filter: drop-shadow(0 0 4px rgba(46,204,113,0.5));
  }
  .header-text h1 {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    background: linear-gradient(90deg, var(--green-bright), var(--gold-soft), var(--green-pale));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 8px rgba(74,222,128,0.3));
  }
  .header-text p {
    font-size: 0.67rem;
    color: rgba(187,247,208,0.45);
    letter-spacing: 0.09em;
    margin-top: 1px;
  }

  /* ── Breathing bubble ───────────────────────────────────────────────── */
  #breatheWrap {
    position: relative;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 9px 0 2px;
    gap: 3px;
  }
  #breatheCircle {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(74,222,128,0.5) 0%, rgba(10,46,26,0.05) 70%);
    border: 1.5px solid rgba(74,222,128,0.4);
    animation: breathe 5s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,100% { transform: scale(1);   opacity: 0.6;  box-shadow: 0 0 8px  rgba(46,204,113,0.2); }
    50%      { transform: scale(1.7); opacity: 1;    box-shadow: 0 0 28px rgba(46,204,113,0.5), 0 0 50px rgba(46,204,113,0.15); }
  }
  #breatheLabel {
    font-size: 0.54rem;
    color: rgba(187,247,208,0.4);
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  /* ── Crisis banner ──────────────────────────────────────────────────── */
  .crisis-banner {
    display: none;
    position: relative;
    z-index: 10;
    width: 100%;
    background: rgba(180,40,40,0.12);
    border-bottom: 1px solid rgba(220,80,80,0.45);
    color: #fca5a5;
    padding: 7px 24px;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
  }
  .crisis-banner a { color: #fca5a5; }

  /* ── Chat wrapper ───────────────────────────────────────────────────── */
  .chat-wrapper {
    position: relative;
    z-index: 10;
    flex: 1;
    width: 100%;
    max-width: 700px;
    display: flex;
    flex-direction: column;
    padding: 18px 16px 6px;
    gap: 14px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(74,222,128,0.2) transparent;
  }

  /* ── Bubbles ────────────────────────────────────────────────────────── */
  .bubble {
    max-width: 80%;
    padding: 14px 20px;
    border-radius: 20px;
    line-height: 1.78;
    font-size: 0.92rem;
    word-break: break-word;
    opacity: 0;
    transform: translateY(14px);
    animation: slideIn 0.55s cubic-bezier(0.16,1,0.3,1) forwards;
  }
  @keyframes slideIn { to { opacity:1; transform:translateY(0); } }

  .bubble.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #0a3d1f, #166534);
    color: var(--green-pale);
    border-bottom-right-radius: 5px;
    border: 1px solid rgba(74,222,128,0.3);
    box-shadow: 0 4px 20px rgba(46,204,113,0.2), inset 0 1px 0 rgba(74,222,128,0.15);
  }
  .bubble.bot {
    align-self: flex-start;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.18);
    color: var(--text);
    border-bottom-left-radius: 5px;
    backdrop-filter: blur(12px);
    box-shadow: 0 2px 20px rgba(0,0,0,0.3);
  }
  .bubble.bot.deeply-open {
    background: rgba(46,204,113,0.08);
    border-color: rgba(74,222,128,0.35);
    box-shadow: 0 0 50px rgba(46,204,113,0.12), inset 0 1px 0 rgba(74,222,128,0.1);
  }
  .bubble.crisis-msg {
    border-color: rgba(220,80,80,0.45);
    background: rgba(180,40,40,0.08);
    box-shadow: 0 0 20px rgba(220,80,80,0.08);
  }
  .bubble .meta {
    font-size: 0.64rem;
    color: rgba(187,247,208,0.35);
    margin-top: 8px;
    border-top: 1px solid rgba(74,222,128,0.1);
    padding-top: 5px;
  }

  /* ── Typing dots ────────────────────────────────────────────────────── */
  .typing-dots {
    align-self: flex-start;
    display: flex;
    gap: 5px;
    padding: 13px 17px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 20px;
    border-bottom-left-radius: 5px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
  }
  .typing-dots span {
    width: 7px; height: 7px;
    background: var(--green-bright);
    border-radius: 50%;
    opacity: 0.7;
    animation: dotBounce 1.2s ease-in-out infinite;
    box-shadow: 0 0 6px rgba(74,222,128,0.6);
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dotBounce {
    0%,80%,100% { transform: translateY(0); opacity: 0.6; }
    40%          { transform: translateY(-8px); opacity: 1; }
  }

  /* ── Input bar ──────────────────────────────────────────────────────── */
  .input-bar {
    position: relative;
    z-index: 10;
    width: 100%;
    max-width: 700px;
    display: flex;
    gap: 10px;
    padding: 8px 16px 18px;
  }
  textarea {
    flex: 1;
    resize: none;
    background: rgba(74,222,128,0.04);
    border: 1px solid rgba(74,222,128,0.22);
    border-radius: 16px;
    color: var(--text);
    font-size: 0.9rem;
    font-family: inherit;
    padding: 12px 18px;
    outline: none;
    line-height: 1.5;
    max-height: 120px;
    transition: border-color 0.3s, box-shadow 0.3s;
  }
  textarea::placeholder { color: rgba(187,247,208,0.3); }
  textarea:focus {
    border-color: rgba(74,222,128,0.5);
    box-shadow: 0 0 0 3px rgba(74,222,128,0.1), 0 0 20px rgba(46,204,113,0.08);
  }
  #sendBtn {
    background: linear-gradient(135deg, #166534, #16a34a, #4ade80);
    color: #fff;
    border: none;
    border-radius: 16px;
    padding: 0 24px;
    font-size: 0.9rem;
    font-family: inherit;
    cursor: pointer;
    font-weight: 600;
    transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s;
    white-space: nowrap;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 20px rgba(46,204,113,0.35);
  }
  #sendBtn:hover   { opacity: 0.88; box-shadow: 0 6px 28px rgba(46,204,113,0.5); }
  #sendBtn:active  { transform: scale(0.97); }
  #sendBtn:disabled { opacity: 0.25; cursor: not-allowed; box-shadow: none; }
  #clearBtn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0 14px;
    font-size: 0.82rem;
    font-family: inherit;
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s;
    white-space: nowrap;
  }
  #clearBtn:hover { border-color: var(--border-glow); color: var(--green-soft); }

  /* ── Risk pill ──────────────────────────────────────────────────────── */
  .risk-pill {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 20px;
    font-size: 0.62rem;
    font-weight: 700;
    margin-right: 4px;
    letter-spacing: 0.05em;
  }
  .risk-low      { background: rgba(46,204,113,0.12); color: #6ee7a0; }
  .risk-moderate { background: rgba(212,168,67,0.12); color: #e8cc6a; }
  .risk-high     { background: rgba(240,130,60,0.12); color: #f0a070; }
  .risk-critical { background: rgba(220,80,80,0.12);  color: #f09090; }

  /* ── Typewriter cursor ──────────────────────────────────────────────── */
  .cursor {
    display: inline-block;
    width: 2px; height: 1em;
    background: var(--green-soft);
    vertical-align: text-bottom;
    margin-left: 1px;
    animation: blink 0.8s step-end infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── About button ───────────────────────────────────────────────────── */
  #aboutBtn {
    margin-left: auto;
    background: transparent;
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 50%;
    width: 34px; height: 34px;
    font-size: 1rem;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  #aboutBtn:hover {
    border-color: rgba(74,222,128,0.6);
    box-shadow: 0 0 12px rgba(46,204,113,0.25);
  }

  /* ── About overlay ──────────────────────────────────────────────────── */
  #aboutOverlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 500;
    background: rgba(2,12,6,0.85);
    backdrop-filter: blur(10px);
    align-items: center;
    justify-content: center;
  }
  #aboutOverlay.open { display: flex; }

  #aboutCard {
    position: relative;
    background: linear-gradient(160deg, #071a0f 0%, #020c06 100%);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 24px;
    box-shadow: 0 0 60px rgba(46,204,113,0.12), 0 20px 60px rgba(0,0,0,0.5);
    padding: 40px 36px 32px;
    max-width: 540px;
    width: 90%;
    max-height: 88vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    animation: cardIn 0.4s cubic-bezier(0.16,1,0.3,1);
  }
  @keyframes cardIn {
    from { opacity: 0; transform: scale(0.92) translateY(16px); }
    to   { opacity: 1; transform: scale(1)    translateY(0); }
  }

  #aboutClose {
    position: absolute;
    top: 16px; right: 18px;
    background: transparent;
    border: none;
    color: rgba(187,247,208,0.4);
    font-size: 1.1rem;
    cursor: pointer;
    transition: color 0.2s;
  }
  #aboutClose:hover { color: var(--green-soft); }

  .about-title {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    background: linear-gradient(135deg, var(--green-bright), var(--gold-soft), var(--green-pale));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 12px rgba(74,222,128,0.3));
    margin-bottom: 4px;
  }
  .about-sub {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: rgba(187,247,208,0.45);
    text-transform: uppercase;
    margin-bottom: 24px;
  }
  .about-body {
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-bottom: 24px;
  }
  .about-body p {
    font-size: 0.9rem;
    line-height: 1.75;
    color: rgba(187,247,208,0.8);
  }
  .about-body strong { color: var(--green-soft); font-weight: 600; }
  .about-body em     { color: var(--gold-soft); font-style: italic; }
  .about-tagline {
    font-size: 0.92rem;
    font-style: italic;
    color: rgba(187,247,208,0.65);
    line-height: 1.7;
    border-top: 1px solid rgba(74,222,128,0.12);
    padding-top: 16px;
    margin-top: 4px;
  }
  .about-disclaimer {
    font-size: 0.75rem;
    color: rgba(187,247,208,0.35);
    margin-top: 14px;
    line-height: 1.6;
  }
  .about-creator {
    margin-top: 20px;
    font-size: 0.82rem;
    color: rgba(187,247,208,0.45);
    letter-spacing: 0.04em;
    border-top: 1px solid rgba(74,222,128,0.1);
    padding-top: 16px;
    width: 100%;
    text-align: center;
  }
  .about-creator strong {
    color: var(--green-pale);
    font-weight: 500;
  }
</style>

<!-- Vercel Web Analytics -->
<script defer src="https://cdn.vercel-insights.com/v1/script.js"></script>
</head>
<body>

<!-- ═══════════ CHAKRA INTRO SCREEN ═══════════ -->
<div id="introScreen">
  <svg id="chakraSvg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Outer glow ring -->
    <circle cx="100" cy="100" r="92" fill="none" stroke="rgba(46,204,113,0.15)" stroke-width="1"/>
    <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(46,204,113,0.25)" stroke-width="1.5"/>
    <!-- 12 petals of Anahata -->
    <g id="petals">
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(0,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(30,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(60,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(90,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(120,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.55)" transform="rotate(150,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(180,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(210,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(240,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(270,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(300,100,100)"/>
      <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(52,220,120,0.45)" transform="rotate(330,100,100)"/>
    </g>
    <!-- Two interlocking triangles (Star of David / Anahata symbol) -->
    <polygon points="100,52 128,98 72,98" fill="none" stroke="rgba(212,168,67,0.8)" stroke-width="1.5"/>
    <polygon points="100,148 72,102 128,102" fill="none" stroke="rgba(212,168,67,0.8)" stroke-width="1.5"/>
    <!-- Inner circle -->
    <circle cx="100" cy="100" r="18" fill="rgba(46,204,113,0.2)" stroke="rgba(46,204,113,0.7)" stroke-width="1.5"/>
    <circle cx="100" cy="100" r="5"  fill="rgba(212,168,67,0.9)"/>
  </svg>

  <div id="introLogo">
    <h1>Anahata</h1>
    <p>the heart within the heart</p>
  </div>

  <div id="introTagline">
    a space built from the belief that every person deserves<br>
    to feel truly seen, heard, and loved — exactly as they are
  </div>
</div>

<!-- ═══════════ MAIN APP SHELL ═══════════ -->
<div id="appShell">
  <div id="stars"></div>

  <header>
    <svg class="header-chakra" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <g>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(0,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(30,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(60,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(90,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(120,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(150,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(180,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(210,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(240,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(270,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(300,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.5)" transform="rotate(330,100,100)"/>
        <polygon points="100,52 128,98 72,98"  fill="none" stroke="rgba(212,168,67,0.9)" stroke-width="2"/>
        <polygon points="100,148 72,102 128,102" fill="none" stroke="rgba(212,168,67,0.9)" stroke-width="2"/>
        <circle cx="100" cy="100" r="5" fill="rgba(212,168,67,1)"/>
      </g>
    </svg>
    <div class="header-text">
      <h1>Anahata</h1>
      <p>you are seen · you are heard · you are not alone</p>
    </div>
    <button id="aboutBtn" onclick="toggleAbout()" title="About Anahata">🪷</button>
  </header>

  <!-- ═══════════ ABOUT OVERLAY ═══════════ -->
  <div id="aboutOverlay" onclick="toggleAbout()">
    <div id="aboutCard" onclick="event.stopPropagation()">
      <button id="aboutClose" onclick="toggleAbout()">✕</button>
      <svg style="width:52px;height:52px;margin-bottom:14px;filter:drop-shadow(0 0 10px rgba(46,204,113,0.6))" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(0,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(30,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(60,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(90,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(120,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.7)" transform="rotate(150,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(180,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(210,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(240,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(270,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(300,100,100)"/>
        <ellipse cx="100" cy="38" rx="10" ry="26" fill="rgba(46,204,113,0.6)" transform="rotate(330,100,100)"/>
        <polygon points="100,52 128,98 72,98"  fill="none" stroke="rgba(212,168,67,0.95)" stroke-width="2"/>
        <polygon points="100,148 72,102 128,102" fill="none" stroke="rgba(212,168,67,0.95)" stroke-width="2"/>
        <circle cx="100" cy="100" r="6" fill="rgba(212,168,67,1)"/>
      </svg>
      <h2 class="about-title">Anahata</h2>
      <p class="about-sub">the heart within the heart</p>
      <div class="about-body">
        <p>We are here because pain is real — and too many people carry it alone, in silence.</p>
        <p>Anahata is a space built on one deep belief: <em>nothing knows you more than yourself.</em> We are not here to fix you, diagnose you, or tell you how to feel. We are here to mirror — to reflect back the exact emotion you are carrying, so you can feel it, meet it, and understand it a little more.</p>
        <p>We spread one thing: <strong>kindness.</strong> Because no technology, no app can ever replace true human connection — or your faith, your relationship with God, or your connection with yourself. But we can remind you, in the moments you forget, that <strong>you are not alone.</strong></p>
        <p>Every day, people go through their hardest moments with nobody to turn to. This is that private space. Nobody is here to judge you. Nobody reads your messages. This is for you — only you.</p>
        <p class="about-tagline">This is for the awakening of your Anahata —<br>the heart that never truly closes.</p>
        <p class="about-disclaimer">Anahata is a supportive companion, not a substitute for professional mental health care. If you are in crisis, please reach out to a professional or call a helpline.</p>
      </div>
      <div class="about-creator">— created with love by <strong>Tanvi Rajpurohit</strong></div>
    </div>
  </div>

  <div id="breatheWrap">
    <div id="breatheCircle"></div>
    <span id="breatheLabel">breathe</span>
  </div>

  <div class="crisis-banner" id="crisisBanner">
    ⚠️ You are not alone — India: <strong>iCall 9152987821</strong> · US: <strong>988</strong> · or visit
    <a href="https://findahelpline.com" target="_blank">findahelpline.com</a>
  </div>

  <div class="chat-wrapper" id="chatWrapper"></div>

  <div class="input-bar">
    <textarea id="userInput" rows="2" placeholder="Share what's on your heart…"></textarea>
    <button id="sendBtn" onclick="sendMessage()">Send</button>
    <button id="clearBtn" onclick="clearChat()">Clear</button>
  </div>
</div>

<script>
const chatWrapper = document.getElementById('chatWrapper');
const input       = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');
const banner      = document.getElementById('crisisBanner');
const stars       = document.getElementById('stars');

// ── Chakra intro sequence ────────────────────────────────────────────────────
function runIntro() {
  const svg      = document.getElementById('chakraSvg');
  const logo     = document.getElementById('introLogo');
  const tagline  = document.getElementById('introTagline');
  const intro    = document.getElementById('introScreen');
  const appShell = document.getElementById('appShell');

  // Step 1: bloom the chakra
  setTimeout(() => svg.classList.add('bloom'), 200);
  // Step 2: add pulse
  setTimeout(() => svg.classList.add('pulse'), 1800);
  // Step 3: show logo
  setTimeout(() => logo.classList.add('show'), 1400);
  // Step 4: show tagline
  setTimeout(() => tagline.classList.add('show'), 2200);
  // Step 5: fade out intro, show app  (+1 second longer = 5200)
  setTimeout(() => {
    intro.classList.add('fade-out');
    appShell.classList.add('show');
    setTimeout(() => { intro.style.display = 'none'; showWelcome(); }, 1200);
  }, 5400);
}

// ── Typewriter effect ────────────────────────────────────────────────────────
function typewrite(el, text, speed=16) {
  return new Promise(resolve => {
    el.textContent = '';
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    el.appendChild(cursor);
    let i = 0;
    const tick = () => {
      if (i < text.length) {
        el.insertBefore(document.createTextNode(text[i++]), cursor);
        setTimeout(tick, speed + Math.random() * 10);
      } else {
        cursor.remove();
        resolve();
      }
    };
    tick();
  });
}

// ── Append bubble ────────────────────────────────────────────────────────────
function appendBubble(text, role, meta='', useTypewriter=false) {
  const div = document.createElement('div');
  div.className = 'bubble ' + role;
  const textNode = document.createElement('span');
  div.appendChild(textNode);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.innerHTML = meta;
    div.appendChild(m);
  }
  chatWrapper.appendChild(div);
  scrollBottom();
  if (useTypewriter && role === 'bot') {
    typewrite(textNode, text).then(scrollBottom);
  } else {
    textNode.textContent = text;
  }
  return div;
}

// ── Mood background transition ───────────────────────────────────────────────
function applyMood(vuln) {
  if (vuln === 'deeply_open') {
    document.body.style.background =
      'radial-gradient(ellipse at 30% 30%, #0a2e18 0%, #020c06 50%), ' +
      'radial-gradient(ellipse at 70% 70%, #1a3a10 0%, #020c06 60%)';
    stars.classList.add('visible');
  } else if (vuln === 'opening_up') {
    document.body.style.background =
      'radial-gradient(ellipse at 40% 30%, #071a0f 0%, #020c06 60%)';
    stars.classList.remove('visible');
  } else {
    document.body.style.background =
      'radial-gradient(ellipse at 30% 20%, #071a0f 0%, #020c06 55%, #030f08 100%)';
    stars.classList.remove('visible');
  }
}

function scrollBottom() { chatWrapper.scrollTop = chatWrapper.scrollHeight; }
function riskClass(l) {
  return ({low:'risk-low',moderate:'risk-moderate',high:'risk-high',critical:'risk-critical'})[l]||'risk-low';
}

input.addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Send message ─────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  appendBubble(text, 'user');
  input.value = '';
  sendBtn.disabled = true;

  const typingEl = document.createElement('div');
  typingEl.className = 'typing-dots';
  typingEl.innerHTML = '<span></span><span></span><span></span>';
  chatWrapper.appendChild(typingEl);
  scrollBottom();

  try {
    const res  = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    chatWrapper.removeChild(typingEl);

    const crisisLvl = data.crisis_level || 'low';
    const vuln      = data.vulnerability || 'neutral';
    const elapsed   = data.elapsed_sec  || '';

    applyMood(vuln);

    const metaHtml = `<span class="risk-pill ${riskClass(crisisLvl)}">${crisisLvl.toUpperCase()}</span>${elapsed}s`;
    const bubble = appendBubble(data.response || data.error || '…', 'bot', metaHtml, true);

    if (vuln === 'deeply_open') bubble.classList.add('deeply-open');
    if (crisisLvl === 'critical' || crisisLvl === 'high') {
      bubble.classList.add('crisis-msg');
      banner.style.display = 'block';
    }
  } catch(err) {
    chatWrapper.removeChild(typingEl);
    appendBubble('Something went wrong. Please try again.', 'bot');
  }
  sendBtn.disabled = false;
  input.focus();
}

// ── Clear chat ───────────────────────────────────────────────────────────────
function clearChat() {
  fetch('/api/clear', { method:'POST' });
  chatWrapper.innerHTML = '';
  banner.style.display = 'none';
  stars.classList.remove('visible');
  document.body.style.background = 'var(--bg)';
  showWelcome();
}

// ── Welcome message ──────────────────────────────────────────────────────────
function showWelcome() {
  const div = document.createElement('div');
  div.className = 'bubble bot';
  const tn = document.createElement('span');
  div.appendChild(tn);
  chatWrapper.appendChild(div);
  const msg = "Hey, I'm really glad you found your way here.\\n\\nThis is Anahata — a space built from the belief that every person deserves to feel truly seen and loved, exactly as they are. No fixing, no judging, no rushing.\\n\\nYou can share as little or as much as you want. I will meet you exactly where you are, always.\\n\\nSo… how are you feeling right now? Even just one word is okay.";
  typewrite(tn, msg, 14).then(scrollBottom);
}

// ── About overlay toggle ─────────────────────────────────────────────────────
function toggleAbout() {
  document.getElementById('aboutOverlay').classList.toggle('open');
}

// ── Start ────────────────────────────────────────────────────────────────────
runIntro();
</script>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the single-page chat UI."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — accepts JSON {message: str} and returns agent response."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    sid = session["session_id"]

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    if len(user_message) > 2000:
        return jsonify({"error": "Message too long (max 2000 chars)."}), 400

    result = _orchestrator.run(user_message, sid)
    return jsonify(result)


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clears the current session's conversation history."""
    sid = session.get("session_id")
    if sid and sid in _conversation_store:
        _conversation_store.pop(sid)
    session.pop("session_id", None)
    return jsonify({"status": "cleared"})


@app.route("/api/history", methods=["GET"])
def history():
    """Returns the conversation history for the current session."""
    sid = session.get("session_id", "")
    data = get_session_data(sid)
    return jsonify({
        "session_id":  sid,
        "history":     data["history"],
        "risk_scores": data["risk_scores"],
        "mood_log":    data["mood_log"],
        "created_at":  data["created_at"],
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status":    "ok",
        "model":     GRANITE_MODEL_ID,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting MindGuard AI on http://0.0.0.0:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)

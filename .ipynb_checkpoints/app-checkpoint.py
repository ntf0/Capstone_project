"""
app.py -- MVP prototype: AI-Powered Personalized Learning System

Scope (intentionally trimmed for a capstone presentation demo):
  1. Simple student ID login (no auth)
  2. One fixed 5-question assignment
  3. Real behavior tracking: accuracy, attempts, hints, response time
  4. Real prediction from the saved K-Means clustering pipeline
  5. Real Claude API call generating a personalized follow-up assignment
  6. Minimal SQLite logging of each attempt (no separate History page)

Run:
    pip install streamlit anthropic joblib scikit-learn pandas numpy
    export ANTHROPIC_API_KEY=sk-ant-...        (Mac/Linux)
    setx ANTHROPIC_API_KEY "sk-ant-..."         (Windows, new terminal after)
    streamlit run app.py
"""

import json
import os
import sqlite3
import time
from datetime import datetime

import joblib
import streamlit as st

from pipeline_utils import PERSONA_DESCRIPTIONS, PERSONA_NAMES, build_feature_vector
from google import genai

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
PIPELINE_PATH = "student_clustering_pipeline.joblib"
DB_PATH = "student_history.db"

FIXED_ASSIGNMENT = [
    {
        "skill": "Fractions",
        "question": "What is 1/2 + 1/4?",
        "answer": "3/4",
        "hints": ["Find a common denominator first.", "1/2 = 2/4, so 2/4 + 1/4 = ?"],
    },
    {
        "skill": "Fractions",
        "question": "Simplify 6/8 to lowest terms.",
        "answer": "3/4",
        "hints": ["Find the greatest common factor of 6 and 8.", "6/8 = (6÷2)/(8÷2)"],
    },
    {
        "skill": "Algebraic Expressions",
        "question": "Solve for x: 2x + 3 = 11",
        "answer": "4",
        "hints": ["Subtract 3 from both sides first.", "2x = 8, now divide both sides by 2."],
    },
    {
        "skill": "Algebraic Expressions",
        "question": "Simplify: 3x + 5x",
        "answer": "8x",
        "hints": ["Combine terms that have the same variable."],
    },
    {
        "skill": "Fractions",
        "question": "What is 2/3 of 9?",
        "answer": "6",
        "hints": ["Multiply 9 by 2, then divide by 3."],
    },
]


# ------------------------------------------------------------------
# Database (minimal -- one table, one row per completed assignment)
# ------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TEXT,
            accuracy REAL,
            avg_attempts REAL,
            persona TEXT,
            generated_assignment TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_attempt(student_id, accuracy, avg_attempts, persona, generated_assignment):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO attempts (student_id, timestamp, accuracy, avg_attempts, persona, generated_assignment) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            student_id,
            datetime.now().isoformat(),
            accuracy,
            avg_attempts,
            persona,
            json.dumps(generated_assignment),
        ),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# Model + AI helpers
# ------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def predict_persona(question_log):
    pipeline = load_pipeline()
    feature_row = build_feature_vector(question_log)

    feature_row["max_opportunity"] = len(question_log)
    
    cluster_label = int(pipeline.predict(feature_row)[0])
    persona = PERSONA_NAMES.get(cluster_label, f"Cluster {cluster_label}")
    return persona, feature_row.iloc[0].to_dict()


def generate_personalized_assignment(
    persona,
    features,
    skill
):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
You are an AI tutor that creates personalized mathematics practice
for middle-school students.

The student completed an assessment focused on ONE mathematics skill.
A machine-learning clustering model analyzed the student's learning
behavior and assigned them to a learning persona.

STUDENT LEARNING PROFILE

Skill being practiced:
{skill}

Learning Persona:
{persona}

Persona Description:
{PERSONA_DESCRIPTIONS.get(persona, "")}

Student Performance:
- Accuracy: {features['accuracy_rate']:.0%}
- Average attempts: {features['avg_attempts']:.1f}
- Retry rate: {features['retry_rate']:.0%}
- Hint dependency: {features['avg_hint_dependency']:.0%}
- Median response time: {features['median_response_time']:.1f} seconds

TASK

Generate exactly 3 NEW mathematics questions that practice ONLY this skill:
{skill}

Personalize the assignment using both the student's learning persona
and observed performance.

PERSONALIZATION RULES

- All 3 questions must focus on the same skill: {skill}.
- Do not repeat the questions from the original assessment.
- Adapt question difficulty based on the student's performance.
- If accuracy is low, use easier or moderately difficult questions
  that reinforce the foundations of the skill.
- If accuracy is high, make the questions progressively more challenging.
- If average attempts or retry rate is high, provide additional guidance.
- If hint dependency is high, provide clear scaffolded hints.
- If hint dependency is low, provide shorter strategy tips instead.
- Consider response time when deciding the level of complexity.
- Adapt the amount of support and difficulty to the student's learning persona.
- Keep the questions appropriate for middle-school students.
- Do not introduce unrelated mathematical skills.
- For each question, include a concise, exact "answer" field (e.g. a number or simplified fraction) that can be checked with a plain string match.


Return ONLY valid JSON in this format:

{{
    "assignment_title": "...",
    "skill": "...",
    "rationale": "...",
    "questions": [
        {{
            "question": "...",
            "answer": "...",
            "hint_or_tip": "..."
        }},
        {{
            "question": "...",
            "answer": "...",
            "hint_or_tip": "..."
        }},
        {{
            "question": "...",
            "answer": "...",
            "hint_or_tip": "..."
        }}
    ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    text = (
        text
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    return json.loads(text)


# ------------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Personalized Learning", page_icon="🎓")
    init_db()

    if "stage" not in st.session_state:
        st.session_state.stage = "login"
    if "log" not in st.session_state:
        st.session_state.log = []
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    # ---------------- Stage 1: Login ----------------
    if st.session_state.stage == "login":
        st.title("🎓 AI-Powered Personalized Learning -- Prototype")
        student_id = st.text_input("Enter your Student ID")
        if st.button("Start Assignment") and student_id:
            st.session_state.student_id = student_id
            st.session_state.stage = "assignment"
            st.session_state.q_start_time = time.time()
            st.rerun()
        return

    # ---------------- Stage 2: Fixed assignment ----------------
    if st.session_state.stage == "assignment":
        idx = st.session_state.q_index
        if idx >= len(FIXED_ASSIGNMENT):
            st.session_state.stage = "results"
            st.rerun()
            return

        q = FIXED_ASSIGNMENT[idx]
        st.subheader(f"Question {idx + 1} of {len(FIXED_ASSIGNMENT)}")
        st.write(q["question"])

        key_prefix = f"q{idx}"
        if f"{key_prefix}_attempts" not in st.session_state:
            st.session_state[f"{key_prefix}_attempts"] = 0
        if f"{key_prefix}_hints_used" not in st.session_state:
            st.session_state[f"{key_prefix}_hints_used"] = 0

        # Hint button
        hints = q["hints"]
        hints_used = st.session_state[f"{key_prefix}_hints_used"]
        if hints_used < len(hints):
            if st.button(f"Show hint ({hints_used + 1}/{len(hints)})", key=f"{key_prefix}_hintbtn"):
                st.session_state[f"{key_prefix}_hints_used"] += 1
                st.rerun()
        for i in range(hints_used):
            st.info(hints[i])

        answer = st.text_input("Your answer", key=f"{key_prefix}_answer")
        if st.button("Submit", key=f"{key_prefix}_submit"):
            st.session_state[f"{key_prefix}_attempts"] += 1
            correct = answer.strip().lower() == q["answer"].strip().lower()

            if correct or st.session_state[f"{key_prefix}_attempts"] >= 3:
                response_time = time.time() - st.session_state.q_start_time
                st.session_state.log.append(
                    {
                        "correct": correct,
                        "attempts": st.session_state[f"{key_prefix}_attempts"],
                        "hints_used": st.session_state[f"{key_prefix}_hints_used"],
                        "hints_available": len(hints),
                        "response_time_sec": response_time,
                        "opportunity": idx + 1,  # simple proxy: position in this session
                        "skill": q["skill"],
                    }
                )
                st.session_state.q_index += 1
                st.session_state.q_start_time = time.time()
                st.rerun()
            else:
                st.warning("Not quite -- try again.")
        return

    # ---------------- Stage 3: Results + AI-generated follow-up ----------------
    if st.session_state.stage == "results":
        st.title("Your Results")
    
        with st.spinner("Analyzing your learning behavior..."):
            persona, features = predict_persona(st.session_state.log)
    
        st.metric("Predicted Learning Persona", persona)
        st.caption(PERSONA_DESCRIPTIONS.get(persona, ""))
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{features['accuracy_rate']:.0%}")
        col2.metric("Avg. Attempts", f"{features['avg_attempts']:.1f}")
        col3.metric(
            "Hint Dependency",
            f"{features['avg_hint_dependency']:.0%}"
        )
    
        # One skill for the whole assignment
        skill = FIXED_ASSIGNMENT[0]["skill"]
    
        if "generated" not in st.session_state:
            with st.spinner(
                "Generating your personalized follow-up assignment..."
            ):
                st.session_state.generated = generate_personalized_assignment(
                    persona,
                    features,
                    skill
                )
    
                log_attempt(
                    st.session_state.student_id,
                    features["accuracy_rate"],
                    features["avg_attempts"],
                    persona,
                    st.session_state.generated,
                )
    
        gen = st.session_state.generated
    
        st.divider()
        st.subheader(f"📝 {gen['assignment_title']}")
        st.caption(f"Skill: {gen['skill']}")
        st.caption(gen["rationale"])
    
        if "gen_answers" not in st.session_state:
            st.session_state.gen_answers = {}
        
        for i, q in enumerate(gen["questions"], start=1):
            key = f"gen_q{i}"
            st.write(f"**{i}. {q['question']}**")
        
            show_hint_key = f"{key}_show_hint"
            if st.button(f"Show tip", key=f"{key}_hintbtn"):
                st.session_state[show_hint_key] = True
            if st.session_state.get(show_hint_key):
                st.info(f"💡 {q['hint_or_tip']}")
        
            user_answer = st.text_input("Your answer", key=f"{key}_input")
        
            if st.button("Check answer", key=f"{key}_check"):
                correct = user_answer.strip().lower() == q["answer"].strip().lower()
                st.session_state.gen_answers[key] = correct
        
            if key in st.session_state.gen_answers:
                if st.session_state.gen_answers[key]:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Not quite. Correct answer: {q['answer']}")    
                    
        if st.button("Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
    
            st.rerun()
    
        return

if __name__ == "__main__":
    main()

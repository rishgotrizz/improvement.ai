"""
EDUPREDICT - MCQ Quiz & Topic Weakness Analysis Service Layer
Handles MCQ quiz creation, topic tagging, student quiz-taking, automated grading, and topic-level weakness insights.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.database import get_db_connection
except ImportError:
    from services.database import get_db_connection

def create_quiz(class_id, subject_code, title, questions):
    """
    Create a new MCQ quiz with topic-tagged questions.
    `questions` is a list of dicts: {question_text, option_a, option_b, option_c, option_d, correct_option, topic}
    """
    if not class_id or not subject_code or not title or not questions:
        return False, None, "Class ID, subject code, title, and questions are required."

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO quizzes (class_id, subject_code, title)
            VALUES (?, ?, ?);
        """, (class_id, subject_code.strip(), title.strip()))
        quiz_id = cursor.lastrowid

        for q in questions:
            q_text = q.get("question_text") or q.get("questionText") or q.get("question") or "Question"
            opt_a = q.get("option_a") or q.get("optionA") or "Option A"
            opt_b = q.get("option_b") or q.get("optionB") or "Option B"
            opt_c = q.get("option_c") or q.get("optionC") or "Option C"
            opt_d = q.get("option_d") or q.get("optionD") or "Option D"
            corr = (q.get("correct_option") or q.get("correctOption") or q.get("correct") or "A").strip().upper()
            topic = (q.get("topic") or "General").strip()

            cursor.execute("""
                INSERT INTO quiz_questions (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_option, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (quiz_id, q_text, opt_a, opt_b, opt_c, opt_d, corr, topic))

        conn.commit()
        conn.close()
        return True, {"quizId": quiz_id, "title": title}, f"Quiz '{title}' created successfully with {len(questions)} topic-tagged questions."
    except Exception as e:
        conn.close()
        return False, None, f"Failed to create quiz: {str(e)}"

def get_class_quizzes(class_id):
    """Retrieve quizzes available for a classroom."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT q.quiz_id, q.class_id, q.subject_code, q.title, q.created_at,
               COUNT(qq.question_id) as question_count
        FROM quizzes q
        LEFT JOIN quiz_questions qq ON q.quiz_id = qq.quiz_id
        WHERE q.class_id = ?
        GROUP BY q.quiz_id
        ORDER BY q.created_at DESC;
    """, (class_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def get_quiz_questions(quiz_id, include_correct=False):
    """Retrieve questions for a quiz (hides correct answers unless explicitly requested)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM quizzes WHERE quiz_id = ?;", (quiz_id,))
    quiz = cursor.fetchone()
    if not quiz:
        conn.close()
        return None

    cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ? ORDER BY question_id ASC;", (quiz_id,))
    questions_rows = cursor.fetchall()
    conn.close()

    questions = []
    for q in questions_rows:
        item = {
            "questionId": q["question_id"],
            "questionText": q["question_text"],
            "optionA": q["option_a"],
            "optionB": q["option_b"],
            "optionC": q["option_c"],
            "optionD": q["option_d"],
            "topic": q["topic"]
        }
        if include_correct:
            item["correctOption"] = q["correct_option"]
        questions.append(item)

    return {
        "quizId": quiz["quiz_id"],
        "title": quiz["title"],
        "subjectCode": quiz["subject_code"],
        "questions": questions
    }

def submit_quiz_attempt(student_id, quiz_id, user_answers):
    """
    Submit student answers for a quiz.
    `user_answers` is a dict mapping questionId -> selectedOption ('A','B','C','D').
    Computes score and per-topic performance breakdown.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?;", (quiz_id,))
    questions = cursor.fetchall()
    if not questions:
        conn.close()
        return False, None, "Quiz questions not found."

    total_q = len(questions)
    correct_count = 0
    topic_stats = {}

    for q in questions:
        qid = str(q["question_id"])
        topic = q["topic"] or "General"
        correct_opt = q["correct_option"].strip().upper()
        user_opt = str(user_answers.get(qid) or user_answers.get(q["question_id"]) or "").strip().upper()

        if topic not in topic_stats:
            topic_stats[topic] = {"correct": 0, "total": 0, "percentage": 0.0}

        topic_stats[topic]["total"] += 1

        if user_opt == correct_opt:
            correct_count += 1
            topic_stats[topic]["correct"] += 1

    for t in topic_stats:
        cnt = topic_stats[t]["total"]
        corr = topic_stats[t]["correct"]
        topic_stats[t]["percentage"] = round((corr / cnt) * 100.0, 1) if cnt > 0 else 0.0

    score_pct = round((correct_count / total_q) * 100.0, 1) if total_q > 0 else 0.0
    topic_json = json.dumps(topic_stats)

    cursor.execute("""
        INSERT INTO quiz_attempts (quiz_id, student_id, score, total_questions, topic_breakdown)
        VALUES (?, ?, ?, ?, ?);
    """, (quiz_id, student_id, correct_count, total_q, topic_json))

    conn.commit()
    conn.close()

    return True, {
        "score": correct_count,
        "totalQuestions": total_q,
        "scorePercentage": score_pct,
        "topicBreakdown": topic_stats
    }, f"Quiz submitted! Scored {correct_count}/{total_q} ({score_pct}%)."

def get_student_topic_analysis(student_id):
    """
    Aggregate topic-level accuracy across all quiz attempts for a student.
    Identifies weak topics (< 70%) and provides educational insights.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT topic_breakdown FROM quiz_attempts
        WHERE student_id = ?
        ORDER BY attempted_at DESC;
    """, (student_id,))
    attempts = cursor.fetchall()
    conn.close()

    if not attempts:
        return {
            "hasData": False,
            "overallTopicStats": {},
            "weakTopics": [],
            "strongTopics": [],
            "insight": "No quiz attempts recorded yet. Complete classroom quizzes to generate topic weakness analysis."
        }

    topic_totals = {}
    for a in attempts:
        try:
            breakdown = json.loads(a["topic_breakdown"]) if a["topic_breakdown"] else {}
            for topic, stats in breakdown.items():
                if topic not in topic_totals:
                    topic_totals[topic] = {"correct": 0, "total": 0}
                topic_totals[topic]["correct"] += stats["correct"]
                topic_totals[topic]["total"] += stats["total"]
        except Exception:
            pass

    topic_summary = {}
    weak_topics = []
    strong_topics = []

    for topic, data in topic_totals.items():
        tot = data["total"]
        corr = data["correct"]
        pct = round((corr / tot) * 100.0, 1) if tot > 0 else 0.0

        item = {
            "topic": topic,
            "correct": corr,
            "total": tot,
            "percentage": pct,
            "status": "WEAK" if pct < 70.0 else "STRONG"
        }
        topic_summary[topic] = item

        if pct < 70.0:
            weak_topics.append(topic)
        else:
            strong_topics.append(topic)

    insight = ""
    if weak_topics:
        insight = f"Your recent quiz performance indicates lower scores in: {', '.join(weak_topics)}. Additional revision in these specific topics is recommended."
    else:
        insight = "Consistent performance demonstrated across all evaluated topics!"

    return {
        "hasData": True,
        "topicSummary": topic_summary,
        "weakTopics": weak_topics,
        "strongTopics": strong_topics,
        "insight": insight
    }

def get_class_topic_analytics(class_id):
    """
    Aggregate topic-level accuracy across an entire class for faculty analytics.
    Identifies topics where many students struggle.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT qa.topic_breakdown
        FROM quiz_attempts qa
        JOIN quizzes q ON qa.quiz_id = q.quiz_id
        WHERE q.class_id = ?;
    """, (class_id,))
    attempts = cursor.fetchall()
    conn.close()

    if not attempts:
        return {"hasData": False, "classTopicStats": {}, "strugglingTopics": []}

    topic_totals = {}
    for a in attempts:
        try:
            breakdown = json.loads(a["topic_breakdown"]) if a["topic_breakdown"] else {}
            for topic, stats in breakdown.items():
                if topic not in topic_totals:
                    topic_totals[topic] = {"correct": 0, "total": 0}
                topic_totals[topic]["correct"] += stats["correct"]
                topic_totals[topic]["total"] += stats["total"]
        except Exception:
            pass

    class_topic_stats = {}
    struggling_topics = []

    for topic, data in topic_totals.items():
        tot = data["total"]
        corr = data["correct"]
        pct = round((corr / tot) * 100.0, 1) if tot > 0 else 0.0

        class_topic_stats[topic] = {
            "topic": topic,
            "classAveragePercentage": pct,
            "totalAnswers": tot
        }
        if pct < 70.0:
            struggling_topics.append(topic)

    return {
        "hasData": True,
        "classTopicStats": class_topic_stats,
        "strugglingTopics": struggling_topics
    }

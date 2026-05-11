import json
import os
import sys
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.base import get_connection

st.set_page_config(page_title="Fahmai Email Monitor", layout="wide", page_icon="📬")

FEEDBACK_THRESHOLD = 50
ALL_TAGS = ["policy", "product", "store_info"]
TAG_COLORS = {"policy": "#0d6efd", "product": "#198754", "store_info": "#fd7e14"}


def _norm_tag(tag: str) -> str:
    """Normalize 'is_policy' -> 'policy' etc."""
    return tag[3:] if tag.startswith("is_") else tag

# ---------- DB helpers ----------

def fetch_emails() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT email_id, sender, subject, body, status,
                       has_attachments, gmail_message_id
                FROM emails ORDER BY received_at DESC NULLS LAST
            """)
            rows = cur.fetchall()
            cols = ["email_id", "sender", "subject", "body", "status", "has_attachments", "gmail_message_id"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def fetch_status_counts() -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) FROM emails GROUP BY status")
            rows = cur.fetchall()
            return {r[0]: r[1] for r in rows}
    finally:
        conn.close()


def fetch_daily_counts() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT email_id, status, received_at FROM emails ORDER BY received_at ASC NULLS LAST
            """)
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["email_id", "status", "received_at"])
            df["received_at"] = pd.to_datetime(df["received_at"])
            df = df.dropna(subset=["received_at"])
            if df.empty:
                return pd.DataFrame()
            df["day"] = df["received_at"].dt.date
            return df
    finally:
        conn.close()


def fetch_tag_daily_counts() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.received_at, p.predicted_tags,
                       fl.corrected_tags
                FROM emails e
                JOIN predictions p ON e.email_id = p.email_id
                LEFT JOIN feedback_logs fl ON e.email_id = fl.email_id
                WHERE e.received_at IS NOT NULL
                ORDER BY e.received_at ASC
            """)
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            records = []
            for eid, received_at, tags_json, corrected_json in rows:
                raw = corrected_json if corrected_json else tags_json
                tags = raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
                day = pd.to_datetime(received_at).date()
                for tag in tags:
                    records.append({"day": day, "tag": tag, "email_id": eid})
            return pd.DataFrame(records)
    finally:
        conn.close()


def fetch_console_rows() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.received_at, e.sender, e.subject, e.body,
                       e.status, e.manual_replied,
                       COALESCE(gr.final_text, gr.draft_text) AS reply_text,
                       p.predicted_tags, p.llm_refined_tags,
                       fl.corrected_tags
                FROM emails e
                LEFT JOIN generated_replies gr ON e.email_id = gr.email_id
                LEFT JOIN predictions p ON e.email_id = p.email_id
                LEFT JOIN feedback_logs fl ON e.email_id = fl.email_id
                WHERE e.status IN ('replied', 'send_failed', 'rejected')
                ORDER BY e.received_at DESC NULLS LAST
            """)
            rows = cur.fetchall()
            cols = ["email_id", "received_at", "sender", "subject", "body",
                    "status", "manual_replied", "reply_text", "predicted_tags", "llm_refined_tags", "corrected_tags"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def fetch_labeler_rows() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.received_at, e.subject, e.body,
                       p.predicted_tags, p.prediction_id, p.llm_refined_tags,
                       fl.corrected_tags, fl.include_in_training
                FROM emails e
                JOIN predictions p ON e.email_id = p.email_id
                LEFT JOIN feedback_logs fl ON e.email_id = fl.email_id
                WHERE p.predicted_tags IS NOT NULL AND p.predicted_tags::text != '[]'
                  AND e.status != 'error'
                ORDER BY e.received_at DESC NULLS LAST
            """)
            rows = cur.fetchall()
            cols = ["email_id", "received_at", "subject", "body",
                    "predicted_tags", "prediction_id", "llm_refined_tags",
                    "corrected_tags", "include_in_training"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def upsert_feedback(
    email_id: int,
    prediction_id: int,
    corrected_tags: list[str],
    include_in_training: bool | None,
) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE feedback_logs SET corrected_tags = %s, include_in_training = %s WHERE email_id = %s",
                    (json.dumps(corrected_tags), include_in_training, email_id),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO feedback_logs (email_id, prediction_id, corrected_tags, action, include_in_training) VALUES (%s, %s, %s, %s, %s)",
                        (email_id, prediction_id, json.dumps(corrected_tags), "edited", include_in_training),
                    )
    finally:
        conn.close()


def set_training_flag(email_id: int, include: bool | None) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE feedback_logs SET include_in_training = %s WHERE email_id = %s",
                    (include, email_id),
                )
    finally:
        conn.close()


def count_feedback() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_logs")
            return cur.fetchone()[0]
    finally:
        conn.close()


def _count_human_review() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM generated_replies gr
                JOIN emails e ON gr.email_id = e.email_id
                WHERE gr.needs_human_review = TRUE AND e.status NOT IN ('replied')
            """)
            return cur.fetchone()[0]
    finally:
        conn.close()


# ---------- Page: Dashboard ----------

def page_dashboard() -> None:
    st.markdown("""<style>
span[data-baseweb="tag"][aria-label^="policy"]    { background-color: #0d6efd !important; }
span[data-baseweb="tag"][aria-label^="product"]   { background-color: #198754 !important; }
span[data-baseweb="tag"][aria-label^="store_info"]{ background-color: #fd7e14 !important; }
</style>""", unsafe_allow_html=True)
    st.title("Dashboard")

    counts = fetch_status_counts()
    total = sum(counts.values())
    replied = counts.get("replied", 0)
    pending = counts.get("pending", 0) + counts.get("error", 0) + counts.get("awaiting_review", 0) + counts.get("send_failed", 0)
    processing = counts.get("processing", 0)

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Emails", total)
    c2.metric("Replied", replied)
    c3.metric("Pending", pending)
    c4.metric("Processing", processing)

    st.divider()

    # Chart selector
    chart_type = st.selectbox("Chart view", ["All emails over time", "By tag over time"])

    if chart_type == "All emails over time":
        df = fetch_daily_counts()
        if df.empty:
            st.info("No data yet")
            return
        daily = df.groupby("day").size().reset_index(name="count")
        fig = px.line(daily, x="day", y="count", markers=True,
                      title="Total email volume (daily)",
                      labels={"day": "Date", "count": "Emails"})
        fig.update_layout(
            hovermode="x unified",
            yaxis=dict(rangemode="tozero"),
            xaxis=dict(tickformat="%d/%m/%Y"),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        df = fetch_tag_daily_counts()
        if df.empty:
            st.info("No tag data yet")
            return
        df["day"] = pd.to_datetime(df["day"])
        df["tag"] = df["tag"].apply(_norm_tag)
        visible_tags = st.multiselect("Show tags", ALL_TAGS, default=ALL_TAGS)
        daily = df.groupby(["day", "tag"]).size().reset_index(name="count")
        daily = daily[daily["tag"].isin(visible_tags)]
        if not daily.empty:
            all_days = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
            full_index = pd.MultiIndex.from_product([all_days, visible_tags], names=["day", "tag"])
            daily = daily.set_index(["day", "tag"]).reindex(full_index, fill_value=0).reset_index()
        fig = px.line(daily, x="day", y="count", color="tag", markers=True,
                      title="Email volume by tag (daily)",
                      color_discrete_map=TAG_COLORS,
                      labels={"day": "Date", "count": "Emails", "tag": "Tag"})
        fig.update_layout(
            hovermode="x unified", legend_title="Tag",
            yaxis=dict(rangemode="tozero"),
            xaxis=dict(tickformat="%d/%m/%Y"),
        )
        st.plotly_chart(fig, use_container_width=True)


def move_email_to_pending(email_id: int) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE emails SET status = 'pending' WHERE email_id = %s", (email_id,))
    finally:
        conn.close()


# ---------- Page: Console ----------

def page_console() -> None:
    st.title("Console - Email Review & Feedback")

    st.divider()

    df = fetch_console_rows()
    if df.empty:
        st.info("ยังไม่มีอีเมล")
        return

    status_colors = {
        "replied": "#28a745", "pending": "#fd7e14", "processing": "#0d6efd",
        "rejected": "#dc3545", "send_failed": "#dc3545", "imported": "#6c757d",
    }

    hcols = st.columns([1, 2, 2, 3, 3, 2, 3])
    for col, label in zip(hcols, ["ID", "Time", "Sender", "Subject", "Body", "Predicted Tags", "การตอบกลับ"]):
        col.markdown(f"**{label}**")
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        eid = int(row.email_id)
        time_str = row.received_at.strftime("%d/%m %H:%M") if pd.notna(row.received_at) else "-"
        raw_tags = row.get("predicted_tags") if "predicted_tags" in row.index else None
        tags = raw_tags if isinstance(raw_tags, list) else (json.loads(raw_tags) if raw_tags else [])

        c_id, c_time, c_sender, c_subject, c_body, c_tags, c_reply = st.columns([1, 2, 2, 3, 3, 2, 3])

        c_id.write(f"#{eid}")
        c_time.caption(time_str)
        c_sender.caption(row.sender or "-")
        c_subject.caption((row.subject or "-")[:50])

        with c_body:
            expand_key = f"con_body_{eid}_{i}"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False
            full_body = row.body or ""
            if st.session_state[expand_key]:
                st.caption(full_body)
                if st.button("ย่อ", key=f"con_body_col_{eid}_{i}"):
                    st.session_state[expand_key] = False
                    st.rerun()
            else:
                st.caption((full_body[:80].replace("\n", " ") + ("..." if len(full_body) > 80 else "")))
                if st.button("ดูเพิ่ม", key=f"con_body_exp_{eid}_{i}"):
                    st.session_state[expand_key] = True
                    st.rerun()

        with c_tags:
            raw_llm = row.get("llm_refined_tags") if "llm_refined_tags" in row.index else None
            llm_tags = raw_llm if isinstance(raw_llm, list) else (json.loads(raw_llm) if raw_llm else [])
            raw_corrected = row.get("corrected_tags") if "corrected_tags" in row.index else None
            corrected_tags = raw_corrected if isinstance(raw_corrected, list) else (json.loads(raw_corrected) if raw_corrected else [])
            display_tags = corrected_tags if corrected_tags else (llm_tags if llm_tags else tags)
            is_relabelled = bool(corrected_tags)
            is_llm_refined = bool(llm_tags) and not corrected_tags
            if display_tags:
                for raw_tag in display_tags:
                    tag = _norm_tag(raw_tag)
                    color = TAG_COLORS.get(tag, "#6c757d")
                    st.markdown(f'<div style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;margin-bottom:3px;display:inline-block">{tag}</div>', unsafe_allow_html=True)
                if is_relabelled:
                    st.markdown('<div style="background:#6f42c1;color:white;padding:1px 5px;border-radius:4px;font-size:0.65em;margin-top:2px;display:inline-block">Re-labelled</div>', unsafe_allow_html=True)
                elif is_llm_refined:
                    st.markdown('<div style="background:#0dcaf0;color:#000;padding:1px 5px;border-radius:4px;font-size:0.65em;margin-top:2px;display:inline-block">LLM refined</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:white;font-size:0.75em">not tagged</span>', unsafe_allow_html=True)

        with c_reply:
            reply_text = row.reply_text or ""
            if reply_text:
                rk = f"con_reply_{eid}_{i}"
                if rk not in st.session_state:
                    st.session_state[rk] = False
                if st.session_state[rk]:
                    st.caption(reply_text)
                    if st.button("ย่อ", key=f"con_reply_col_{eid}_{i}"):
                        st.session_state[rk] = False
                        st.rerun()
                else:
                    st.caption((reply_text[:80].replace("\n", " ") + ("..." if len(reply_text) > 80 else "")))
                    if st.button("ดูเพิ่ม", key=f"con_reply_exp_{eid}_{i}"):
                        st.session_state[rk] = True
                        st.rerun()
            else:
                st.caption("-")

        st.divider()


def fetch_pending_emails() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.received_at, e.sender, e.subject, e.body,
                       e.manual_replied, e.status,
                       gr.reply_id, gr.draft_text, gr.review_reason, gr.needs_human_review,
                       p.predicted_tags, fl.corrected_tags
                FROM emails e
                LEFT JOIN generated_replies gr ON e.email_id = gr.email_id
                LEFT JOIN predictions p ON e.email_id = p.email_id
                LEFT JOIN feedback_logs fl ON e.email_id = fl.email_id
                WHERE e.status IN ('pending', 'error', 'awaiting_review', 'send_failed')
                ORDER BY e.received_at DESC NULLS LAST
            """)
            rows = cur.fetchall()
            cols = ["email_id", "received_at", "sender", "subject", "body",
                    "manual_replied", "status",
                    "reply_id", "draft_text", "review_reason", "needs_human_review", "predicted_tags", "corrected_tags"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def set_manual_replied(email_id: int, replied: bool | None) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE emails SET manual_replied = %s WHERE email_id = %s",
                    (replied, email_id),
                )
    finally:
        conn.close()


def _count_sent_for_training() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_logs WHERE include_in_training = TRUE")
            return cur.fetchone()[0]
    finally:
        conn.close()


def _get_last_trained_count_mlflow() -> int:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.config import BERT_MODEL_NAME, MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{BERT_MODEL_NAME}'")
        if not versions:
            return 0
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        return int(run.data.params.get("labeled_count", "0"))
    except Exception:
        return 0



def fetch_review_queue() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.sender, e.subject, e.body,
                       e.manual_replied,
                       p.predicted_tags,
                       gr.reply_id, gr.draft_text, gr.review_reason, gr.retry_count
                FROM emails e
                LEFT JOIN predictions p ON e.email_id = p.email_id
                LEFT JOIN generated_replies gr ON e.email_id = gr.email_id
                WHERE gr.needs_human_review = TRUE
                  AND e.status NOT IN ('replied')
                ORDER BY e.received_at DESC NULLS LAST
            """)
            rows = cur.fetchall()
            cols = ["email_id", "sender", "subject", "body", "manual_replied",
                    "predicted_tags", "reply_id", "draft_text", "review_reason", "retry_count"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def approve_draft(email_id: int, reply_id: int) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generated_replies SET needs_human_review = FALSE WHERE reply_id = %s",
                    (reply_id,),
                )
                cur.execute(
                    "UPDATE emails SET status = 'replied' WHERE email_id = %s",
                    (email_id,),
                )
    finally:
        conn.close()


def reject_draft(email_id: int) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE emails SET status = 'rejected' WHERE email_id = %s",
                    (email_id,),
                )
    finally:
        conn.close()


def set_rq_manual_replied(email_id: int, replied: bool | None) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE emails SET manual_replied = %s WHERE email_id = %s",
                    (replied, email_id),
                )
    finally:
        conn.close()


# ---------- Page: Review Queue ----------

def page_review_queue() -> None:
    st.title("Review Queue - รอ Human Review")

    df = fetch_review_queue()
    if df.empty:
        st.info("ไม่มีอีเมลที่รอ Review")
        return

    st.caption(f"ทั้งหมด {len(df)} รายการ - AI ส่งมาเพราะไม่มั่นใจพอ ต้องให้คนตรวจก่อนส่ง")
    st.divider()

    # Table header
    hcols = st.columns([1, 2, 3, 4, 4, 3])
    for col, label in zip(hcols, ["ID", "Sender", "Subject", "Body", "Draft", "Actions"]):
        col.markdown(f"**{label}**")
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        eid = int(row.email_id)
        reply_id = int(row.reply_id) if pd.notna(row.reply_id) else None
        manual_replied = row["manual_replied"] if pd.notna(row["manual_replied"]) else None

        c_id, c_sender, c_subject, c_body, c_draft, c_actions = st.columns([1, 2, 3, 4, 4, 3])

        c_id.write(f"#{eid}")
        c_sender.write(row.sender or "-")
        c_subject.write((row.subject or "-")[:50])

        # Body expandable
        with c_body:
            body_key = f"rq_body_exp_{eid}_{i}"
            if body_key not in st.session_state:
                st.session_state[body_key] = False
            full_body = row.body or ""
            if st.session_state[body_key]:
                st.caption(full_body)
                if st.button("ย่อ", key=f"rq_body_collapse_{eid}_{i}"):
                    st.session_state[body_key] = False
                    st.rerun()
            else:
                st.caption((full_body[:80].replace("\n", " ") + ("..." if len(full_body) > 80 else "")))
                if st.button("ดูเพิ่ม", key=f"rq_body_expand_{eid}_{i}"):
                    st.session_state[body_key] = True
                    st.rerun()

        # Draft expandable
        with c_draft:
            draft_key = f"rq_draft_exp_{eid}_{i}"
            if draft_key not in st.session_state:
                st.session_state[draft_key] = False
            full_draft = row.draft_text or ""
            if st.session_state[draft_key]:
                st.caption(full_draft)
                if st.button("ย่อ", key=f"rq_draft_collapse_{eid}_{i}"):
                    st.session_state[draft_key] = False
                    st.rerun()
            else:
                st.caption((full_draft[:80].replace("\n", " ") + ("..." if len(full_draft) > 80 else "")))
                if st.button("ดูเพิ่ม", key=f"rq_draft_expand_{eid}_{i}"):
                    st.session_state[draft_key] = True
                    st.rerun()

        # Actions
        with c_actions:
            if reply_id:
                if st.button("Approve & Send", key=f"rq_approve_{eid}_{i}", type="primary", use_container_width=True):
                    approve_draft(eid, reply_id)
                    st.rerun()

            # Mark as manually replied
            if manual_replied is True:
                if st.button("ยังไม่ตอบ", key=f"rq_manual_{eid}_{i}", use_container_width=True):
                    set_rq_manual_replied(eid, None)
                    st.rerun()
                st.markdown('<span style="background:#28a745;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">ตอบแล้ว</span>', unsafe_allow_html=True)
            else:
                if st.button("ตอบแล้ว", key=f"rq_manual_{eid}_{i}", use_container_width=True):
                    set_rq_manual_replied(eid, True)
                    st.rerun()

            if st.button("Reject", key=f"rq_reject_{eid}_{i}", use_container_width=True):
                reject_draft(eid)
                st.rerun()

        st.divider()


# ---------- Page: Labeler ----------

def page_labeler() -> None:
    st.title("Labeler - ติด Tag สำหรับ Training")

    df = fetch_labeler_rows()
    if df.empty:
        st.info("ยังไม่ได้รับผลพยากรณ์จาก BERT")
        return

    send_count = _count_sent_for_training()
    last_trained = _get_last_trained_count_mlflow()
    new_since_last = send_count - last_trained
    ready = new_since_last >= FEEDBACK_THRESHOLD
    card_color = "#28a745" if ready else "#6c757d"

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.caption(f"BERT tag ได้ {len(df)} รายการ | ติด label แล้ว: **{send_count}**")
    with top_right:
        st.markdown(
            f"""
            <div style="background:#1e1e2e;border:1px solid #444;border-radius:10px;padding:10px 16px;text-align:center">
                <div style="font-size:0.75em;color:#aaa;margin-bottom:4px">ใหม่ตั้งแต่เทรนล่าสุด</div>
                <div style="font-size:1.6em;font-weight:bold;color:{card_color}">{new_since_last} / {FEEDBACK_THRESHOLD}</div>
                <div style="font-size:0.7em;color:#888;margin-top:4px">เทรนล่าสุด: {last_trained} records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()

    hcols = st.columns([1, 2, 3, 4, 2, 3])
    for col, label in zip(hcols, ["ID", "Time", "Subject", "Body", "Bert Predicted", "Feedback"]):
        col.markdown(f"**{label}**")
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        eid = int(row.email_id)
        pid = int(row.prediction_id) if pd.notna(row.prediction_id) else 0
        time_str = row.received_at.strftime("%d/%m %H:%M") if pd.notna(row.received_at) else "-"
        raw_tags = row.predicted_tags
        tags = raw_tags if isinstance(raw_tags, list) else (json.loads(raw_tags) if raw_tags else [])
        raw_corrected = row.corrected_tags
        corrected = raw_corrected if isinstance(raw_corrected, list) else (json.loads(raw_corrected) if pd.notna(row.corrected_tags) and row.corrected_tags else [])
        include = row.include_in_training

        c_id, c_time, c_subject, c_body, c_tags, c_feedback = st.columns([1, 2, 3, 4, 2, 3])

        c_id.caption(f"#{eid}")
        c_time.caption(time_str)
        c_subject.caption((row.subject or "-")[:50])

        with c_body:
            bk = f"lab_body_{eid}_{i}"
            if bk not in st.session_state:
                st.session_state[bk] = False
            full_body = row.body or ""
            if st.session_state[bk]:
                st.caption(full_body)
                if st.button("ย่อ", key=f"lab_body_col_{eid}_{i}"):
                    st.session_state[bk] = False
                    st.rerun()
            else:
                st.caption((full_body[:80].replace("\n", " ") + ("..." if len(full_body) > 80 else "")))
                if st.button("ดูเพิ่ม", key=f"lab_body_exp_{eid}_{i}"):
                    st.session_state[bk] = True
                    st.rerun()

        with c_tags:
            for raw_tag in tags:
                tag = _norm_tag(raw_tag)
                color = TAG_COLORS.get(tag, "#6c757d")
                st.markdown(f'<div style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;margin-bottom:3px;display:inline-block">{tag}</div>', unsafe_allow_html=True)
            if not tags:
                st.markdown('<span style="color:#888;font-size:0.75em">-</span>', unsafe_allow_html=True)

        with c_feedback:
            working_tags = corrected if corrected else [t for t in tags if t in ALL_TAGS]
            new_tags = []
            for tag in ALL_TAGS:
                color = TAG_COLORS[tag]
                ch_label, ch_box = st.columns([5, 1])
                with ch_label:
                    st.markdown(f'<div style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;display:inline-block;margin-top:6px">{tag}</div>', unsafe_allow_html=True)

                with ch_box:
                    if st.checkbox("", value=(tag in working_tags), key=f"lab_chk_{tag}_{eid}_{i}", label_visibility="collapsed", disabled=(include is True)):
                        new_tags.append(tag)

            b_btn, b_status = st.columns([3, 3])
            with b_btn:
                if include is True:
                    if st.button("Unsend", key=f"lab_btn_{eid}_{i}", use_container_width=True):
                        upsert_feedback(eid, pid, new_tags, False)
                        st.rerun()
                else:
                    if st.button("Send", key=f"lab_btn_{eid}_{i}", use_container_width=True):
                        upsert_feedback(eid, pid, new_tags if new_tags else [], True)
                        st.rerun()
            with b_status:
                if include is True:
                    st.markdown('<span style="background:#28a745;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">ส่งแล้ว</span>', unsafe_allow_html=True)
                elif include is False:
                    st.markdown('<span style="background:#dc3545;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">ไม่ส่ง</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="background:#6c757d;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">Waiting</span>', unsafe_allow_html=True)

        st.divider()


# ---------- Page: Pending ----------

def page_pending() -> None:
    st.title("Pending - รอตรวจ")

    df = fetch_pending_emails()
    if df.empty:
        st.info("ไม่มีอีเมลที่รอ")
        return

    st.caption(f"ทั้งหมด {len(df)} รายการ")
    st.divider()

    hcols = st.columns([1, 2, 2, 3, 3, 2, 3])
    for col, label in zip(hcols, ["ID", "Time", "Sender", "Subject", "Body", "Predicted Tags", "การตอบกลับ"]):
        col.markdown(f"**{label}**")
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        eid = int(row.email_id)
        reply_id = int(row.reply_id) if pd.notna(row.reply_id) else None
        replied = row.manual_replied
        is_ai_review = row.needs_human_review is True
        time_str = row.received_at.strftime("%d/%m %H:%M") if pd.notna(row.received_at) else "-"
        raw_tags = row.get("predicted_tags") if "predicted_tags" in row.index else None
        tags = raw_tags if isinstance(raw_tags, list) else (json.loads(raw_tags) if raw_tags else [])

        if is_ai_review:
            st.markdown('<div style="border-left:4px solid #dc3545;padding-left:6px;margin-bottom:2px"><small style="color:#dc3545">AI Review</small></div>', unsafe_allow_html=True)

        c_id, c_time, c_sender, c_subject, c_body, c_tags_col, c_action = st.columns([1, 2, 2, 3, 3, 2, 3])

        c_id.caption(f"#{eid}")
        c_time.caption(time_str)
        c_sender.caption(row.sender or "-")
        c_subject.caption((row.subject or "-")[:50])

        with c_body:
            bk = f"pend_body_{eid}_{i}"
            if bk not in st.session_state:
                st.session_state[bk] = False
            full_body = row.body or ""
            if st.session_state[bk]:
                st.caption(full_body)
                if st.button("ย่อ", key=f"pend_body_col_{eid}_{i}"):
                    st.session_state[bk] = False
                    st.rerun()
            else:
                st.caption((full_body[:80].replace("\n", " ") + ("..." if len(full_body) > 80 else "")))
                if st.button("ดูเพิ่ม", key=f"pend_body_exp_{eid}_{i}"):
                    st.session_state[bk] = True
                    st.rerun()

        row_status = row.get("status", "")
        with c_tags_col:
            if row_status == "error":
                st.markdown('<span style="color:#dc3545;font-size:0.75em">error</span>', unsafe_allow_html=True)
            else:
                raw_corrected_p = row.get("corrected_tags") if "corrected_tags" in row.index else None
                corrected_tags_p = raw_corrected_p if isinstance(raw_corrected_p, list) else (json.loads(raw_corrected_p) if raw_corrected_p else [])
                display_tags_p = corrected_tags_p if corrected_tags_p else tags
                is_relabelled_p = bool(corrected_tags_p)
                if display_tags_p:
                    for raw_tag in display_tags_p:
                        tag = _norm_tag(raw_tag)
                        color = TAG_COLORS.get(tag, "#6c757d")
                        st.markdown(f'<div style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;margin-bottom:3px;display:inline-block">{tag}</div>', unsafe_allow_html=True)
                    if is_relabelled_p:
                        st.markdown('<div style="background:#6f42c1;color:white;padding:1px 5px;border-radius:4px;font-size:0.65em;margin-top:2px;display:inline-block">Re-labelled</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:white;font-size:0.75em">not tagged</span>', unsafe_allow_html=True)

        with c_action:
            if is_ai_review and reply_id:
                if st.button("Approve & Send", key=f"pend_approve_{eid}_{i}", type="primary", use_container_width=True):
                    approve_draft(eid, reply_id)
                    st.rerun()
                draft = row.draft_text or ""
                if draft:
                    dk = f"pend_draft_{eid}_{i}"
                    if dk not in st.session_state:
                        st.session_state[dk] = False
                    if st.session_state[dk]:
                        st.caption(draft)
                        if st.button("ย่อ draft", key=f"pend_draft_col_{eid}_{i}"):
                            st.session_state[dk] = False
                            st.rerun()
                    else:
                        if st.button("ดู draft", key=f"pend_draft_exp_{eid}_{i}"):
                            st.session_state[dk] = True
                            st.rerun()

            b_btn, b_status = st.columns([3, 3])
            with b_btn:
                if replied is True:
                    if st.button("ยังไม่ตอบ", key=f"pend_tog_false_{eid}_{i}", use_container_width=True):
                        set_manual_replied(eid, None)
                        st.rerun()
                elif replied is False:
                    if st.button("Waiting", key=f"pend_tog_wait_{eid}_{i}", use_container_width=True):
                        set_manual_replied(eid, True)
                        st.rerun()
                else:
                    if st.button("ตอบแล้ว", key=f"pend_tog_true_{eid}_{i}", use_container_width=True):
                        set_manual_replied(eid, True)
                        st.rerun()
            with b_status:
                if replied is True:
                    st.markdown('<span style="background:#28a745;color:white;padding:3px 10px;border-radius:4px;font-size:0.85em">ส่งแล้ว</span>', unsafe_allow_html=True)
                elif replied is False:
                    st.markdown('<span style="background:#dc3545;color:white;padding:3px 10px;border-radius:4px;font-size:0.85em">ยังไม่ตอบ</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="background:#6c757d;color:white;padding:3px 10px;border-radius:4px;font-size:0.85em">Waiting</span>', unsafe_allow_html=True)

        st.divider()


# ---------- Sidebar navigation ----------

st.sidebar.title("Fahmai Email Monitor")

counts = fetch_status_counts()
pending_count = counts.get("pending", 0) + _count_human_review()

pending_label = f"Pending ({pending_count})" if pending_count > 0 else "Pending"

page = st.sidebar.radio("Navigation", ["Dashboard", "Console", pending_label, "Labeler"])

if page == "Dashboard":
    page_dashboard()
elif page == "Console":
    page_console()
elif page == "Labeler":
    page_labeler()
else:
    page_pending()

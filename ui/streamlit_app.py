"""Streamlit front end for the Enterprise Document QA Assistant.

The UI is a thin client over the API: it holds a JWT in session state and
never touches the index or the documents directly, so every permission
decision stays server-side.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120

CATEGORY_LABELS = {
    "company_policy": "Company policy",
    "audit_procedure": "Audit procedure",
    "risk_report": "Risk report",
    "cybersecurity": "Cybersecurity",
    "general": "General",
}

st.set_page_config(
    page_title="Enterprise Document QA", page_icon="📄", layout="wide"
)


def api(method: str, path: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    return requests.request(
        method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs
    )


def logout() -> None:
    for key in ("token", "user"):
        st.session_state.pop(key, None)


# --- login -------------------------------------------------------------


def login_view() -> None:
    st.title("📄 Enterprise Document QA Assistant")
    st.caption(
        "Ask questions about company policies, audit procedures, risk reports "
        "and cybersecurity documentation. Every answer is cited, and the "
        "assistant abstains when the documents do not support one."
    )

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if not submitted:
        return

    try:
        response = requests.post(
            f"{API_URL}/auth/token",
            data={"username": username, "password": password},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        return

    if response.status_code != 200:
        st.error("Incorrect username or password.")
        return

    st.session_state["token"] = response.json()["access_token"]
    st.session_state["user"] = api("GET", "/me").json()
    st.rerun()


# --- main app ----------------------------------------------------------


def sidebar(user: dict) -> tuple[str, int]:
    with st.sidebar:
        st.subheader(f"Signed in as {user['username']}")
        st.caption(f"Role: **{user['role']}**")

        categories = user["allowed_categories"] or ["all categories"]
        st.markdown("**Cleared to read**")
        st.markdown(
            "\n".join(
                f"- {CATEGORY_LABELS.get(item, item)}" for item in categories
            )
            + f"\n- up to *{user['max_sensitivity']}*"
        )

        st.divider()
        st.markdown("**Retrieval**")
        mode = st.selectbox(
            "Method",
            ["hybrid", "dense", "sparse"],
            help="Hybrid fuses dense embeddings with BM25 keyword search.",
        )
        top_k = st.slider("Passages retrieved", 1, 12, 6)

        st.divider()
        st.markdown("**Documents you can see**")
        documents = api("GET", "/documents")
        if documents.ok:
            for meta in documents.json():
                st.caption(
                    f"{meta['filename']} · {meta['page_count']} pages · "
                    f"{meta['sensitivity']}"
                )
        else:
            st.caption("Could not load the document list.")

        st.divider()
        if st.button("Sign out"):
            logout()
            st.rerun()

    return mode, top_k


def render_answer(payload: dict) -> None:
    if payload["abstained"]:
        st.warning(
            "**Not enough evidence in the document collection.**\n\n"
            "This assistant only answers from the documents you are cleared to "
            "read, and will not guess."
        )
        if payload.get("reason"):
            st.caption(payload["reason"])
        return

    st.markdown(payload["answer"])

    st.markdown("#### Sources")
    for index, citation in enumerate(payload["citations"], start=1):
        label = f"{index}. {citation['filename']} · page {citation['page']}"
        if citation.get("section"):
            label += f" · {citation['section']}"
        with st.expander(label):
            quote = citation.get("quote")
            st.markdown(f"> {quote}" if quote else "*No quote returned.*")

    st.caption(
        f"Model: {payload['model']} · {len(payload['contexts'])} passages considered "
        f"· {payload['latency_ms']} ms"
    )


def app_view(user: dict) -> None:
    mode, top_k = sidebar(user)

    st.title("📄 Enterprise Document QA Assistant")
    question = st.text_input(
        "Ask a question about the document collection",
        placeholder="e.g. How often must privileged account passwords be rotated?",
    )

    if not question:
        st.info(
            "Try: *What must happen within 4 hours of an incident report?* · "
            "*What sample size applies to a monthly control?* · "
            "*Which risks are above appetite this quarter?*"
        )
        return

    with st.spinner("Retrieving passages and checking the evidence…"):
        response = api(
            "POST",
            "/query",
            json={"question": question, "mode": mode, "top_k": top_k},
        )

    if response.status_code == 401:
        logout()
        st.warning("Session expired. Please sign in again.")
        st.rerun()
    elif not response.ok:
        st.error(response.json().get("detail", "The query failed."))
        return

    render_answer(response.json())

    with st.expander("Retrieved passages (grounding debug)"):
        hits = api(
            "POST",
            "/search",
            json={"question": question, "mode": mode, "top_k": top_k},
        )
        if hits.ok:
            st.dataframe(hits.json(), use_container_width=True)


def main() -> None:
    if "token" not in st.session_state:
        login_view()
        return

    user = st.session_state.get("user")
    if not user:
        response = api("GET", "/me")
        if not response.ok:
            logout()
            st.rerun()
        user = st.session_state["user"] = response.json()

    app_view(user)


main()

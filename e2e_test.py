"""
╔══════════════════════════════════════════════════════════════════════════╗
║          END-TO-END BACKEND TEST SUITE                                   ║
║          Auth → Conversation → Chat → Upload → RAG → Cleanup            ║
║                                                                          ║
║  Usage:                                                                  ║
║      pip install requests colorama                                       ║
║      python e2e_test.py                                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import requests
import json
import time
import sys
import os

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
import os as _os
PDF_PATH = _os.getenv("E2E_PDF_PATH", "./tests/fixtures/sample.pdf")
TEST_EMAIL    = f"e2etest_{int(time.time())}@test.com"
TEST_PASSWORD = "TestPass@123"
TEST_NAME     = "E2E Tester"

# ── Colors ────────────────────────────────────
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    GREEN = Fore.GREEN; RED = Fore.RED; CYAN = Fore.CYAN
    BOLD = Style.BRIGHT; RESET = Style.RESET_ALL
except ImportError:
    GREEN = RED = CYAN = BOLD = RESET = ""

# ── Persistent cookie session ─────────────────
# Backend uses HTTP-only cookies (access_token), NOT Bearer tokens.
# requests.Session stores + resends the cookie automatically like a browser.
session = requests.Session()

# ── Shared state ──────────────────────────────
state = {"conversation_id": None, "ingested_file": None, "rag_doc_id": None}

# ── Result tracker ────────────────────────────
results = []

def run(name, fn):
    print(f"\n{'─'*60}")
    print(f"{CYAN}{BOLD}TEST: {name}{RESET}")
    try:
        fn()
        print(f"{GREEN}✅  PASSED{RESET}")
        results.append(("PASS", name))
    except AssertionError as e:
        print(f"{RED}❌  FAILED — {e}{RESET}")
        results.append(("FAIL", name, str(e)))
    except Exception as e:
        print(f"{RED}💥  ERROR  — {e}{RESET}")
        results.append(("ERROR", name, str(e)))

def assert_ok(resp, expected=200, label=""):
    if resp.status_code != expected:
        try:   body = resp.json()
        except: body = resp.text[:300]
        raise AssertionError(f"{label} — Expected {expected}, got {resp.status_code}\n   Body: {body}")

def show_cookie():
    has = "access_token" in dict(session.cookies)
    print(f"   🍪 access_token cookie present: {has}")
    return has


# ══════════════════════════════════════════════
#  1 & 2 — HEALTH / ROOT
# ══════════════════════════════════════════════
def test_health():
    r = session.get(f"{BASE_URL}/health", timeout=15)
    assert_ok(r, 200, "GET /health")
    body = r.json()
    print(f"   {body}")
    assert body.get("status") == "healthy"

def test_root():
    r = session.get(f"{BASE_URL}/", timeout=15)
    assert_ok(r, 200, "GET /")
    body = r.json()
    print(f"   {body}")
    assert "message" in body


# ══════════════════════════════════════════════
#  3 & 4 — SIGNUP
# ══════════════════════════════════════════════
def test_signup():
    r = session.post(f"{BASE_URL}/auth/signup",
                     json={"email": TEST_EMAIL, "password": TEST_PASSWORD, "name": TEST_NAME},
                     timeout=15)
    assert_ok(r, 200, "POST /auth/signup")
    body = r.json()
    print(f"   {body}")
    assert "user" in body or "message" in body
    show_cookie()

def test_signup_duplicate():
    # Fresh session (no cookie) to test rejection
    r = requests.post(f"{BASE_URL}/auth/signup",
                      json={"email": TEST_EMAIL, "password": TEST_PASSWORD, "name": TEST_NAME},
                      timeout=15)
    assert r.status_code in [400, 409, 422], \
        f"Expected 400/409/422, got {r.status_code}: {r.text}"
    print(f"   Correctly rejected: {r.status_code}")


# ══════════════════════════════════════════════
#  5 & 6 — LOGIN
# ══════════════════════════════════════════════
def test_login():
    r = session.post(f"{BASE_URL}/auth/login",
                     json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
                     timeout=15)
    assert_ok(r, 200, "POST /auth/login")
    body = r.json()
    print(f"   {body}")
    # The backend sets the JWT as an HTTP-only cookie — NOT in body.
    # Session will store the cookie automatically.
    assert body.get("message") == "Login successful" or "user" in body, \
        f"Unexpected: {body}"
    has = show_cookie()
    assert has, "No access_token cookie set after login!"

def test_login_wrong_password():
    r = requests.post(f"{BASE_URL}/auth/login",
                      json={"email": TEST_EMAIL, "password": "BadPass!"},
                      timeout=15)
    assert r.status_code in [400, 401, 422], \
        f"Expected 401, got {r.status_code}"
    print(f"   Correctly blocked: {r.status_code}")


# ══════════════════════════════════════════════
#  7 & 8 — /auth/me
# ══════════════════════════════════════════════
def test_get_me():
    r = session.get(f"{BASE_URL}/auth/me", timeout=15)
    assert_ok(r, 200, "GET /auth/me")
    body = r.json()
    print(f"   Current user: {body}")
    assert "email" in body, f"No email in response: {body}"
    assert body["email"] == TEST_EMAIL

def test_me_unauthenticated():
    r = requests.get(f"{BASE_URL}/auth/me", timeout=15)  # fresh — no cookie
    assert r.status_code in [401, 403], f"Expected 401/403, got {r.status_code}"
    print(f"   Blocked correctly: {r.status_code}")


# ══════════════════════════════════════════════
#  9–11 — CONVERSATIONS
# ══════════════════════════════════════════════
def test_create_conversation():
    r = session.post(f"{BASE_URL}/conversations",
                     json={"title": "E2E Test Conversation"}, timeout=15)
    assert_ok(r, 200, "POST /conversations")
    body = r.json()
    print(f"   Response: {body}")
    conv_id = (body.get("_id") or body.get("id") or body.get("conversation_id")
               or (body.get("conversation") or {}).get("_id")
               or (body.get("conversation") or {}).get("id"))
    assert conv_id, f"No conv ID: {body}"
    state["conversation_id"] = str(conv_id)
    print(f"   Saved conv ID: {state['conversation_id']}")

def test_list_conversations():
    r = session.get(f"{BASE_URL}/conversations", timeout=15)
    assert_ok(r, 200, "GET /conversations")
    body = r.json()
    assert isinstance(body, list), f"Expected list, got {type(body)}"
    print(f"   Count: {len(body)}")

def test_get_messages():
    conv_id = state["conversation_id"]
    assert conv_id
    r = session.get(f"{BASE_URL}/conversations/{conv_id}/messages", timeout=15)
    assert_ok(r, 200, "GET /conversations/{id}/messages")
    body = r.json()
    print(f"   Messages: {len(body) if isinstance(body, list) else body}")


# ══════════════════════════════════════════════
#  12 & 13 — CHAT (SSE streaming)
# ══════════════════════════════════════════════
def test_chat_basic():
    r = session.post(f"{BASE_URL}/chat/stream",
                     json={
                         "message": "Say hello in exactly one sentence.",
                         "conversation_id": state["conversation_id"],
                         "model": "gemini-2.5-flash-lite",
                         "enabled_tools": [],
                         "selected_files": []
                     }, stream=True, timeout=60)
    assert_ok(r, 200, "POST /chat/stream")
    chunks, text = 0, ""
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            d = line[5:].strip()
            if d and d != "[DONE]":
                try:
                    obj = json.loads(d)
                    text += obj.get("content") or obj.get("text") or obj.get("delta") or str(obj)
                except Exception:
                    text += d
                chunks += 1
    print(f"   Chunks: {chunks}  |  Preview: {text[:120]}")
    assert chunks > 0, "No SSE chunks received"

def test_chat_with_tool():
    r = session.post(f"{BASE_URL}/chat/stream",
                     json={
                         "message": "What is the current time?",
                         "conversation_id": state["conversation_id"],
                         "model": "gemini-2.5-flash-lite",
                         "enabled_tools": ["get_current_time"],
                         "selected_files": []
                     }, stream=True, timeout=60)
    assert_ok(r, 200, "POST /chat/stream [tool]")
    chunks = sum(1 for l in r.iter_lines(decode_unicode=True)
                 if l.startswith("data:") and l.strip() != "data: [DONE]")
    print(f"   Tool-chat chunks: {chunks}")
    assert chunks > 0


# ══════════════════════════════════════════════
#  14 — PDF UPLOAD + INGESTION
# ══════════════════════════════════════════════
def test_pdf_upload():
    if not os.path.exists(PDF_PATH):
        print(f"   ⚠️  PDF not found at {PDF_PATH}. Skipping upload test.")
        print(f"   Set E2E_PDF_PATH env var to a real PDF path to enable this test.")
        results.append(("SKIP", "PDF Upload + Ingestion"))
        return
    assert os.path.exists(PDF_PATH), f"PDF not found: {PDF_PATH}"
    mb = os.path.getsize(PDF_PATH) / 1024 / 1024
    print(f"   File: {os.path.basename(PDF_PATH)} ({mb:.2f} MB)")
    print(f"   ⏳ LlamaParse cloud may take 2-5 min — please wait...")

    with open(PDF_PATH, "rb") as f:
        r = session.post(
            f"{BASE_URL}/api/v1/ingest/upload",
            files={"file": (os.path.basename(PDF_PATH), f, "application/pdf")},
            data={"document_type": "Academic"},
            timeout=360
        )
    assert_ok(r, 200, "POST /api/v1/ingest/upload")
    body = r.json()
    print(f"   Result:\n{json.dumps(body, indent=2)[:600]}")

    details = body.get("details", {})
    state["ingested_file"] = details.get("file_name") or os.path.basename(PDF_PATH)
    state["rag_doc_id"]    = details.get("doc_store_id") or details.get("json_id")
    print(f"   ingested_file = {state['ingested_file']}")
    print(f"   rag_doc_id    = {state['rag_doc_id']}")


# ══════════════════════════════════════════════
#  15 — RAG LIST FILES
# ══════════════════════════════════════════════
def test_rag_list_files():
    r = session.get(f"{BASE_URL}/api/v1/rag/files", timeout=30)
    assert_ok(r, 200, "GET /api/v1/rag/files")
    body = r.json()
    files = body.get("files", [])
    print(f"   Files in Qdrant: {files}")
    assert isinstance(files, list)


# ══════════════════════════════════════════════
#  16 — RAG RETRIEVE (raw vector search)
# ══════════════════════════════════════════════
def test_rag_retrieve():
    r = session.post(f"{BASE_URL}/api/v1/rag/retrieve",
                     json={"message": "image based question generation", "selected_files": None},
                     timeout=30)
    assert_ok(r, 200, "POST /api/v1/rag/retrieve")
    body = r.json()
    chunks = body.get("chunks", [])
    print(f"   Chunks: {body.get('count', len(chunks))}")
    if chunks:
        print(f"   First: {chunks[0].get('content','')[:120]}...")
    assert isinstance(chunks, list)


# ══════════════════════════════════════════════
#  17 — RAG AGENTIC CHAT (full pipeline)
# ══════════════════════════════════════════════
def test_rag_chat():
    questions = [
        "What is the main goal of the image based question generation feature?",
        "How are images handled in the question generation pipeline?",
        "What are the key implementation steps mentioned in the document?",
    ]
    for q in questions:
        print(f"\n   Q: {q}")
        r = session.post(f"{BASE_URL}/api/v1/rag/chat",
                         json={"message": q, "selected_file_ids": None},
                         timeout=180)  # 429 SDK retries can take 30s+
        assert_ok(r, 200, "POST /api/v1/rag/chat")
        body = r.json()
        answer = body.get("answer", "")
        print(f"   A: {answer[:200]}...")
        print(f"   Sources: {body.get('sources')} | Hallucination: {body.get('hallucination_warning')}")
        assert len(answer) > 10, f"Answer too short: '{answer}'"
        time.sleep(5)  # let rate-limit quota replenish between questions


def test_rag_chat_file_filter():
    if not state["ingested_file"]:
        print("   ⚠️  Skipping — no ingested_file")
        return
    r = session.post(f"{BASE_URL}/api/v1/rag/chat",
                     json={
                         "message": "Summarize the document",
                         "selected_file_ids": [state["ingested_file"]]
                     },
                     timeout=180)
    assert_ok(r, 200, "POST /api/v1/rag/chat [filter]")
    body = r.json()
    print(f"   Filtered answer: {body.get('answer','')[:200]}...")
    assert body.get("answer")


# ══════════════════════════════════════════════
#  19 — READ PAGE (DocStore)
# ══════════════════════════════════════════════
def test_read_page():
    if not state["rag_doc_id"]:
        print("   ⚠️  Skipping — no rag_doc_id")
        return
    r = session.post(f"{BASE_URL}/api/v1/rag/read-page",
                     json={"doc_id": state["rag_doc_id"], "page": 1},
                     timeout=30)
    assert_ok(r, 200, "POST /api/v1/rag/read-page")
    body = r.json()
    print(f"   Page 1: {str(body)[:200]}...")
    assert body


# ══════════════════════════════════════════════
#  20 — TOOLS
# ══════════════════════════════════════════════
def test_list_tools():
    r = session.get(f"{BASE_URL}/api/tools", timeout=15)  # correct prefix
    assert_ok(r, 200, "GET /api/tools")
    print(f"   {str(r.json())[:200]}")


# ══════════════════════════════════════════════
#  21 — LOGOUT
# ══════════════════════════════════════════════
def test_logout():
    r = session.post(f"{BASE_URL}/auth/logout", timeout=15)
    assert r.status_code in [200, 204], f"Logout: {r.status_code}"
    print(f"   Logout: {r.status_code}")


# ══════════════════════════════════════════════
#  22 — DELETE CONVERSATION
# ══════════════════════════════════════════════
def test_delete_conversation():
    conv_id = state["conversation_id"]
    if not conv_id:
        print("   ⚠️  Skipping")
        return
    r = session.delete(f"{BASE_URL}/conversations/{conv_id}", timeout=15)
    assert r.status_code in [200, 204, 404], f"Delete: {r.status_code}"
    print(f"   Deleted {conv_id}: {r.status_code}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"{BOLD}{CYAN}  E2E TEST SUITE — {BASE_URL}{RESET}")
    print(f"{'═'*60}")
    print(f"  User : {TEST_EMAIL}")
    print(f"  PDF  : {PDF_PATH}")
    print(f"  Auth : HTTP-only cookie via requests.Session()")
    print(f"{'═'*60}\n")

    run("1.  Health Check",                 test_health)
    run("2.  Root Endpoint",                test_root)
    run("3.  Signup",                       test_signup)
    run("4.  Signup Duplicate (4xx)",       test_signup_duplicate)
    run("5.  Login",                        test_login)
    run("6.  Login Wrong Password (401)",   test_login_wrong_password)
    run("7.  GET /auth/me (authed)",        test_get_me)
    run("8.  GET /auth/me (no cookie)",     test_me_unauthenticated)
    run("9.  Create Conversation",          test_create_conversation)
    run("10. List Conversations",           test_list_conversations)
    run("11. Get Messages",                 test_get_messages)
    run("12. Basic Chat SSE",               test_chat_basic)
    run("13. Chat with Tool",               test_chat_with_tool)
    run("14. PDF Upload + Ingestion",       test_pdf_upload)
    run("15. RAG List Files",               test_rag_list_files)
    run("16. RAG Retrieve (raw)",           test_rag_retrieve)
    run("17. RAG Agentic Chat (3 Qs)",      test_rag_chat)
    run("18. RAG Chat (file filter)",       test_rag_chat_file_filter)
    run("19. RAG Read Page (DocStore)",     test_read_page)
    run("20. List Tools",                   test_list_tools)
    run("21. Delete Conversation",           test_delete_conversation)  # cleanup before logout
    run("22. Logout",                        test_logout)

    print(f"\n{'═'*60}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{'═'*60}")
    passed  = sum(1 for r in results if r[0] == "PASS")
    failed  = sum(1 for r in results if r[0] == "FAIL")
    errored = sum(1 for r in results if r[0] == "ERROR")
    for res in results:
        icon = f"{GREEN}✅" if res[0] == "PASS" else f"{RED}❌"
        msg  = f" → {res[2]}" if len(res) > 2 else ""
        print(f"  {icon}  {res[1]}{msg}{RESET}")
    print(f"\n  {BOLD}Total {len(results)} | {GREEN}Pass {passed}{RESET} | {RED}Fail {failed} | Error {errored}{RESET}")
    print(f"{'═'*60}\n")
    sys.exit(0 if (failed + errored) == 0 else 1)

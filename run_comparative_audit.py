"""
Comparative Audit Test Runner
==============================
Executes two threads sending the same 5 questions to Gemini 3.8 Flash:
  Thread 1: Official `agy` CLI
  Thread 2: `curl` requests to the hosted LLMProxy router

Intercepts, logs, and compares the exact wire traffic transferred
between this local environment and Google Antigravity servers.
"""

import http.server
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List
import httpx

QUESTIONS = [
    "What is the capital city of France? Answer with only the city name.",
    "What is 17 multiplied by 6? Answer with only the number.",
    "What is the boiling point of water in Celsius? Answer with only the number.",
    "Name the third planet from the Sun. Answer in 1 word.",
    "What is the chemical symbol for gold? Answer with only the symbol.",
]

THREAD_1_LOG = "/tmp/mitm/traffic_thread_1_agy.jsonl"
THREAD_2_LOG = "/tmp/mitm/traffic_thread_2_router.jsonl"

for p in (THREAD_1_LOG, THREAD_2_LOG):
    if os.path.exists(p):
        os.remove(p)

def read_chunked(rfile):
    body = bytearray()
    while True:
        line = rfile.readline()
        if not line:
            break
        chunk_len_str = line.strip().split(b";")[0]
        if not chunk_len_str:
            continue
        chunk_len = int(chunk_len_str, 16)
        if chunk_len == 0:
            rfile.readline()
            break
        chunk_data = rfile.read(chunk_len)
        body.extend(chunk_data)
        rfile.readline()
    return bytes(body)

class AuditMitmHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_CONNECT(self):
        self.send_response(200, "Connection Established")
        self.end_headers()

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile="/tmp/mitm/srv.crt", keyfile="/tmp/mitm/srv.key")
        try:
            tls_conn = context.wrap_socket(self.connection, server_side=True)
        except Exception:
            return

        try:
            rfile = tls_conn.makefile("rb")
            while True:
                req_line = rfile.readline().decode("utf-8", errors="replace")
                if not req_line:
                    break

                parts = req_line.strip().split()
                if len(parts) < 2:
                    break
                method, path = parts[0], parts[1]

                headers = {}
                while True:
                    line = rfile.readline().decode("utf-8", errors="replace")
                    if line in ("\r\n", "\n", ""):
                        break
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip()] = v.strip()

                is_chunked = headers.get("Transfer-Encoding", "").lower() == "chunked"
                if is_chunked:
                    body_bytes = read_chunked(rfile)
                else:
                    content_length = int(headers.get("Content-Length", 0))
                    body_bytes = rfile.read(content_length) if content_length > 0 else b""

                target_host = self.path.split(":")[0]
                full_url = f"https://{target_host}{path}"

                client_ip = self.client_address[0]
                caller = "thread_1_agy" if client_ip == "127.0.0.1" else "thread_2_router"

                sanitized_headers = dict(headers)
                if "Authorization" in sanitized_headers:
                    sanitized_headers["Authorization"] = "Bearer REDACTED_OAUTH_TOKEN"

                log_entry = {
                    "caller": caller,
                    "client_ip": client_ip,
                    "timestamp": time.time(),
                    "method": method,
                    "url": full_url,
                    "headers": sanitized_headers,
                    "body": body_bytes.decode("utf-8", errors="replace"),
                }

                with httpx.Client(timeout=45.0) as client:
                    fwd_headers = {
                        k: v for k, v in headers.items()
                        if k.lower() not in ("host", "content-length", "accept-encoding", "transfer-encoding")
                    }
                    if "stream" in path or "alt=sse" in path:
                        with client.stream(method, full_url, headers=fwd_headers, content=body_bytes) as resp:
                            tls_conn.sendall(f"HTTP/1.1 {resp.status_code} {resp.reason_phrase}\r\n".encode("utf-8"))
                            for hk, hv in resp.headers.items():
                                if hk.lower() not in ("transfer-encoding", "content-length", "content-encoding"):
                                    tls_conn.sendall(f"{hk}: {hv}\r\n".encode("utf-8"))
                            tls_conn.sendall(b"Transfer-Encoding: chunked\r\n\r\n")

                            captured_chunks = []
                            for chunk in resp.iter_bytes():
                                if chunk:
                                    captured_chunks.append(chunk.decode("utf-8", errors="replace"))
                                    hex_len = hex(len(chunk))[2:].encode("utf-8") + b"\r\n"
                                    tls_conn.sendall(hex_len + chunk + b"\r\n")
                            tls_conn.sendall(b"0\r\n\r\n")
                            log_entry["response_status"] = resp.status_code
                            log_entry["response_body"] = "".join(captured_chunks)
                    else:
                        resp = client.request(method, full_url, headers=fwd_headers, content=body_bytes)
                        tls_conn.sendall(f"HTTP/1.1 {resp.status_code} {resp.reason_phrase}\r\n".encode("utf-8"))
                        for hk, hv in resp.headers.items():
                            if hk.lower() not in ("transfer-encoding", "content-length", "content-encoding"):
                                tls_conn.sendall(f"{hk}: {hv}\r\n".encode("utf-8"))
                        tls_conn.sendall(f"Content-Length: {len(resp.content)}\r\n\r\n".encode("utf-8"))
                        tls_conn.sendall(resp.content)
                        log_entry["response_status"] = resp.status_code
                        log_entry["response_body"] = resp.text

                target_log_file = THREAD_1_LOG if caller == "thread_1_agy" else THREAD_2_LOG
                with open(target_log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()

        except Exception as e:
            pass
        finally:
            try:
                tls_conn.close()
            except Exception:
                pass


def run_mitm_server():
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 9999), AuditMitmHandler)
    server.serve_forever()


def run_thread_1_agy() -> List[Dict[str, Any]]:
    """Thread 1: Executes official `agy` CLI for all 5 questions."""
    results = []
    print("\n[Thread 1] Starting execution via official `agy` CLI...")
    env = dict(os.environ)
    env["HTTPS_PROXY"] = "http://127.0.0.1:9999"
    env["AGY_CA_CERT"] = "/tmp/mitm/ca.crt"
    env["SSL_CERT_FILE"] = "/tmp/mitm/ca.crt"

    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [Thread 1] (Q{i}/5): {q}")
        start = time.time()
        proc = subprocess.run(
            ["/root/.local/bin/agy", "-p", q, "--model", "gemini-3.8-flash-high"],
            env=env,
            capture_output=True,
            text=True,
        )
        duration = round(time.time() - start, 2)
        answer = proc.stdout.strip()
        print(f"    -> Answer ({duration}s): {answer}")
        results.append({
            "question_index": i,
            "question": q,
            "answer": answer,
            "stderr": proc.stderr.strip()[:300],
            "returncode": proc.returncode,
            "duration": duration,
        })
        time.sleep(1)
    return results


def run_thread_2_router() -> List[Dict[str, Any]]:
    """Thread 2: Executes `curl` commands to local LLMProxy router."""
    results = []
    print("\n[Thread 2] Starting execution via curl to LLMProxy router...")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [Thread 2] (Q{i}/5): {q}")
        start = time.time()
        payload = json.dumps({
            "model": "gemini-3.8-flash-high",
            "messages": [{"role": "user", "content": q}],
        })
        proc = subprocess.run(
            [
                "curl", "-s", "-X", "POST", "http://127.0.0.1:8000/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True,
            text=True,
        )
        duration = round(time.time() - start, 2)
        raw_output = proc.stdout.strip()
        try:
            parsed = json.loads(raw_output)
            answer = parsed["choices"][0]["message"]["content"].strip()
        except Exception:
            answer = f"ERROR: {raw_output}"

        print(f"    -> Answer ({duration}s): {answer}")
        results.append({
            "question_index": i,
            "question": q,
            "answer": answer,
            "raw_response": raw_output[:300],
            "duration": duration,
        })
        time.sleep(1)
    return results


def main():
    # 1. Start MITM proxy in background thread
    proxy_thread = threading.Thread(target=run_mitm_server, daemon=True)
    proxy_thread.start()
    time.sleep(1.5)

    # 2. Run Thread 1 (agy CLI)
    t1_results = run_thread_1_agy()

    # 3. Run Thread 2 (curl to LLMProxy)
    t2_results = run_thread_2_router()

    # 4. Parse captured traffic
    t1_traffic = []
    if os.path.exists(THREAD_1_LOG):
        with open(THREAD_1_LOG) as f:
            for line in f:
                t1_traffic.append(json.loads(line))

    t2_traffic = []
    if os.path.exists(THREAD_2_LOG):
        with open(THREAD_2_LOG) as f:
            for line in f:
                t2_traffic.append(json.loads(line))

    # Filter to main streamGenerateContent requests
    t1_chats = [t for t in t1_traffic if "streamGenerateContent" in t["url"]]
    t2_chats = [t for t in t2_traffic if "streamGenerateContent" in t["url"]]

    # Save comprehensive reports
    report_t1 = {
        "thread": "Thread 1: Official agy CLI",
        "questions_count": len(QUESTIONS),
        "results": t1_results,
        "total_requests_captured": len(t1_traffic),
        "stream_generate_content_requests": t1_chats,
    }
    with open("report_thread_1_agy.json", "w") as f:
        json.dump(report_t1, f, indent=2)

    report_t2 = {
        "thread": "Thread 2: LLMProxy Router (via curl)",
        "questions_count": len(QUESTIONS),
        "results": t2_results,
        "total_requests_captured": len(t2_traffic),
        "stream_generate_content_requests": t2_chats,
    }
    with open("report_thread_2_router.json", "w") as f:
        json.dump(report_t2, f, indent=2)

    print("\n" + "=" * 60)
    print("REPORTS GENERATED:")
    print("  - report_thread_1_agy.json")
    print("  - report_thread_2_router.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

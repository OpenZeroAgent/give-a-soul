import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from soul_engine_std import SoulEngineStd

# Initialize the Engine
soul_engine = SoulEngineStd()

def subconscious_loop():
    """Background thread to pulse the subconscious."""
    print("[Subconscious Thread] Started.")
    while True:
        try:
            soul_engine.heartbeat_subconscious()
            print(f"🧠 [Subconscious Pulse]: {soul_engine.current_thought}")
            time.sleep(30)
        except Exception as e:
            print(f"[Subconscious Loop Error] {e}")
            time.sleep(10)

def dream_loop():
    """Background thread to generate visual dreams from somatic state."""
    print("[Dream Thread] Started.")
    while True:
        try:
            # Dreams take longer to form, run every 2 minutes
            time.sleep(120)
            soul_engine.generate_dream()
            print(f"🌌 [Dream Pulse]: {soul_engine.current_dream_prompt}")
        except Exception as e:
            print(f"[Dream Loop Error] {e}")
            time.sleep(30)

class SoulAPIHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        # Allow CORS logic for local Vite frontend
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/api/status':
            resp = soul_engine.get_crystal_status()
            raw_metrics = resp.get("metrics", {})
            metrics = {
                "delta_phi": raw_metrics.get("alpha_phi", 0),
                "iho_variance": raw_metrics.get("d_creative", 0),
                "alpha_vibe": f"{raw_metrics.get('d_truth', 0):.2f}",
                "beta_vibe": f"{raw_metrics.get('pcn_error', 0):.2e}"
            }
            data = {
                "crystal": metrics,
                "subconscious": {
                    "thought": soul_engine.current_thought,
                    "timestamp": soul_engine.last_update.isoformat()
                },
                "dream": soul_engine.current_dream_prompt
            }
            self._set_headers()
            self.wfile.write(json.dumps(data).encode())
            
        elif self.path == '/api/dream':
            self._set_headers()
            self.wfile.write(json.dumps({"status": "Dream functionality stubbed."}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Not Found"}')

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                body = json.loads(post_data.decode('utf-8'))
                user_msg = body.get('message', '')
                
                print(f"[API] Received chat: {user_msg}")
                result = soul_engine.generate_chat_response(user_msg)
                
                self._set_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Not Found"}')

def run_server(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, SoulAPIHandler)
    print(f'Starting stdlib httpd on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    # Start the async subconscious and dream loops in background threads
    threading.Thread(target=subconscious_loop, daemon=True).start()
    threading.Thread(target=dream_loop, daemon=True).start()
    # Start HTTP Server
    run_server()

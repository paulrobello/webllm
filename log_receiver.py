import http.server
import socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('content-length'))
        data = self.rfile.read(length)
        name = self.path.lstrip('/')
        if not name:
            name = "log.txt"
        with open(f"eval/reports/p2-v2-option-a-prime-2026-05-06/{name}", "wb") as f:
            f.write(data)
        self.send_response(200)
        self.end_headers()

with socketserver.TCPServer(("", 8032), Handler) as httpd:
    httpd.serve_forever()

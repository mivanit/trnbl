"Usage: python start_server.py path/to/directory [port]"

import os
import http.server
import socketserver

def start_server(path: str, port: int = 8000) -> None:
	"""Starts a server to serve the files in the given path."""
	os.chdir(path)
	with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
		print(f"Serving at http://localhost:{port}")
		httpd.serve_forever()

if __name__ == "__main__":
	import sys
	if len(sys.argv) == 1:
		print(__doc__)
		sys.exit(1)
	elif len(sys.argv) == 2:
		start_server(sys.argv[1])
	elif len(sys.argv) == 3:
		start_server(sys.argv[1], int(sys.argv[2]))
	else:
		print(f"Invalid number of arguments!\n{__doc__}")
		sys.exit(1)
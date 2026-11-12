# Load the setup again in a running kernel. The server takes one connection at a time, so close the old one first.
CLIENT.close()
exec(open("runs/osworld-4b-teacher/setup.py").read())
SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld3.sock")

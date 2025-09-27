import socket


def has_internet(host="duckduckgo.com", port=53, timeout=3):
    """Check for internet connectivity by attempting to connect to a reliable host."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except OSError:
        return False

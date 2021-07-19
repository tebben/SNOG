import os

from rest.manage import create_app

HOSTNAME = "SNOG_HOSTNAME"
PORT = "SNOG_PORT"

if __name__ == "__main__":
    """Run SNOG REST service in debug mode"""

    app = create_app()
    host_name = os.getenv(HOSTNAME, "localhost")
    port = os.getenv(PORT, "5000")
    app.run(debug=True, host=host_name, port=int(port))
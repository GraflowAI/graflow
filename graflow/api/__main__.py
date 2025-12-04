"""Make graflow.api package executable as a module.

This allows running the API server with:
    python -m graflow.api [args]
"""

from graflow.api.main import main

if __name__ == "__main__":
    main()

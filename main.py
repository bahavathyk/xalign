"""Run the German Credit XAI comparison dashboard locally."""
import os

from app import app


def main():
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8050")), debug=False)


if __name__ == "__main__":
    main()

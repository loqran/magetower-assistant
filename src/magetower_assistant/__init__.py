def main() -> None:
    from .chat import main as chat_main

    raise SystemExit(chat_main())


__version__ = "0.1.0"

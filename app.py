import sys
from cli import VanillaLLM, BASE_URL, MODEL
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import os


SYSTEM = "You are a helpful assistant."
HISTORY_FILE = os.path.expanduser("~/.vanillachat_history")


if __name__ == "__main__":
    conv = [{"role": "system", "content": SYSTEM}]
    session = PromptSession(history=FileHistory(HISTORY_FILE), vi_mode=True)

    with VanillaLLM(BASE_URL) as llm:
        if len(sys.argv) > 1:
            conv.append({"role": "user", "content": " ".join(sys.argv[1:])})
            llm.ask(conv, model=MODEL)
        else:
            print("Welcome to VanillaChat! Type 'exit' or press Ctrl+D to quit.")
            while True:
                try:
                    user = session.prompt("You: ")
                except EOFError:
                    print("\nBye."); break
                if user.lower() in ("exit", "quit"):
                    print("Bye."); break
                conv.append({"role": "user", "content": user})
                conv.append({"role": "assistant", "content": llm.ask(conv, model=MODEL)})
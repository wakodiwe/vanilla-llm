import sys
from cli import llm, MODEL

SYSTEM = "You are a helpful assistant."

if __name__ == "__main__":
    conv = [{"role": "system", "content": SYSTEM}]

    if len(sys.argv) > 1:
        conv.append({"role": "user", "content": " ".join(sys.argv[1:])})
            llm.ask(conv, model=MODEL)
    else:
        print("Welcome to VanillaChat! Type 'exit' to quit.")
        while True:
            try:
                user = input("\nYou: ")
            except EOFError:
                print("\nBye."); break
            if user.lower() in ("exit", "quit"):
                print("Bye."); break
            conv.append({"role": "user", "content": user})
            conv.append({"role": "assistant", "content": llm.ask(conv, model=MODEL)})
        llm.close()

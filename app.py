import sys
from cli import llm, MODEL

SYSTEM = "You are a helpful assistant."

if __name__ == "__main__":
    msg = lambda r, c: {"role": r, "content": c}
    conv = [msg("system", SYSTEM)]

    if len(sys.argv) > 1:
        conv.append(msg("user", " ".join(sys.argv[1:])))
        llm.stream_print(conv, model=MODEL)
    else:
        print("Welcome to VanillaChat! Type 'exit' to quit.")
        while True:
            try:
                user = input("\nYou: ")
            except EOFError:
                print("\nBye."); break
            if user.lower() in ("exit", "quit"):
                print("Bye."); break
            conv.append(msg("user", user))
            conv.append(msg("assistant", llm.stream_print(conv, model=MODEL)))
        llm.close()

def tester(text="Too short"):
    print(text if len(text) >= 10 else "Too short")

def main():
    while True:
        user_input = input("Write something (quit to exit): ")
        if user_input.lower() == "quit":
            break
        tester(user_input)

if __name__ == "__main__":
    main()
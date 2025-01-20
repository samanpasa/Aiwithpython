def main():
    grocery_list = []

    while True:
        choice = input("(1) Add (2) Remove (3) Quit: ")

        if choice == '1':
            grocery_list.append(input("Add item: "))
        elif choice == '2':
            if grocery_list:
                print(f"Items: {grocery_list}")
                try:
                    grocery_list.pop(int(input("Index to remove: ")))
                except (ValueError, IndexError):
                    print("Invalid selection.")
            else:
                print("List is empty.")
        elif choice == '3':
            print("Final list:", grocery_list)
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
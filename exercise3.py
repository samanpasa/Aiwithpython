def main():
    prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
    total = 0

    print("Supermarket\n===========")
    while True:
        try:
            product = int(input("Select product (1-10), 0 to quit: "))
            if product == 0:
                break
            if 1 <= product <= 10:
                total += prices[product - 1]
                print(f"Product {product} - Price: {prices[product - 1]}")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    payment = float(input(f"Total: {total}, Payment: "))
    print(f"Change: {payment - total}")


if __name__ == "__main__":
    main()
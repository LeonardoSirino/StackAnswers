def get_float_input(prompt: str) -> float:
    try:
        data = input(prompt)
        data = float(data)

        return data
    except ValueError:
        raise ValueError('Please enter a valid number')


def main():
    while True:
        print("-" * 30 + "MENU" + "-" * 30)
        print("|" + " " * 19 + "1. Hourly to annual pay" + " " * 20 + "|")
        print("|" + " " * 19 + "2. Annual to hourly pay" + " " * 20 + "|")
        print("|" + " " * 19 + "3. Exit" + " " * 36 + "|")
        print("-" * 64)

        choice = input("Choose an option 1-3: ")
        if choice == "1":
            try:
                hourly = get_float_input("Enter hourly pay: ")
            except ValueError as e:
                print(e)
                continue

            annual = hourly * 2080
            print("Annual pay is: ")
            print("{:0.2f}".format(annual))

        elif choice == "2":
            try:
                a = get_float_input("Enter annual pay: ")
            except ValueError as e:
                print(e)
                continue

            h = a / 2080
            x = round(h, 2)
            print("{:0.2f}".format(x))

        elif choice == "3":
            print("Program exited gracefully.")
            raise SystemExit(0)

        else:
            print("Unrecognized command - choose 1-3")


if __name__ == '__main__':
    main()

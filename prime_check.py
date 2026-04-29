# Take input from the user as text.
user_input = input("Enter a number: ")

try:
    # Convert the user input to an integer.
    number = int(user_input)

    # Numbers less than 2 are not prime.
    if number < 2:
        print(f"{number} is not a prime number.")
    else:
        is_prime = True

        # Check divisibility from 2 up to the square root of the number.
        # If any number divides it exactly, then it is not prime.
        for divisor in range(2, int(number ** 0.5) + 1):
            if number % divisor == 0:
                is_prime = False
                break

        # Print the final result.
        if is_prime:
            print(f"{number} is a prime number.")
        else:
            print(f"{number} is not a prime number.")

except ValueError:
    # This runs if the user enters something that cannot be converted to int.
    print("Invalid input. Please enter a valid integer.")

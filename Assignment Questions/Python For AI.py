# Assignment :- 1
# Python For AI ---------------------------------------------------------------------------------------

'''
Question 1: 
Write a Python program to determine the largest among three numbers entered by the user?
'''

def find_largest(num1, num2, num3):
    if num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3

def main():
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))
    num3 = float(input("Enter the third number: "))

    largest = find_largest(num1, num2, num3)
    print("The largest number is:", largest)

if __name__ == "__main__":
    main()




'''
Question ¹:

Write a Python program to check if a given year is a leap year?

Hints -
1. Check if the year is divisible by 4
2. Check if the year is divisible by 100, but not divisible by 400
3. If the year satisfies either of the above conditions, it's a leap year
'''


def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

def main():
    year = int(input("Enter a year: "))

    if is_leap_year(year):
        print(year, "is a leap year.")
    else:
        print(year, "is not a leap year.")

if __name__ == "__main__":
    main()



'''
Question 3:

Write a Python program to print the following pattern:
'''


def print_number_triangle(rows):
    for i in range(1, rows + 1):
        for j in range(1, i + 1):
            print(j, end=" ")
        print()





'''
Question 4:
Write a Python program to print the following pattern:
'''


def print_normal_star_triangle(rows):
    for i in range(1, rows + 1):
        print(" " * (rows - i) + "*" * (2 * i - 1))

def main():
    rows = int(input("Enter the number of rows for the triangle: "))
    print("Normal Star Triangle:")
    print_normal_star_triangle(rows)

if __name__ == "__main__":
    main()




'''
Question 5:

Write a Python program to print the following pattern:
'''



def print_opposite_side_triangle(rows):
    for i in range(1, rows + 1):
        print(" " * (rows - i) + "*" * i + "*" * (i - 1))

def main():
    rows = int(input("Enter the number of rows for the triangle: "))
    print("Opposite Side Triangle:")
    print_opposite_side_triangle(rows)

if __name__ == "__main__":
    main()



'''
Question 6:

Write a program that prints the numbers 1 to 100. However, for multiples of 3, print "Fizz" instead of the 
number. For multiples of 5, print "Buzz". For numbers that are multiples of both 3 and 5, print "FizzBuzz".
'''

def fizz_buzz():
    for i in range(1, 101):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

def main():
    fizz_buzz()

if __name__ == "__main__":
    main()



'''
Question 7: 

Write a Python program to find the sum of all prime numbers up to n.
'''

def is_prime(num):
    if num <= 1:
        return False
    elif num == 2:
        return True
    elif num % 2 == 0:
        return False
    else:
        for i in range(3, int(num**0.5) + 1, 2):
            if num % i == 0:
                return False
        return True

def sum_of_primes_up_to_n(n):
    sum_primes = 0
    for i in range(2, n + 1):
        if is_prime(i):
            sum_primes += i
    return sum_primes

def main():
    n = int(input("Enter a number (n): "))
    sum_primes = sum_of_primes_up_to_n(n)
    print("The sum of all prime numbers up to", n, "is:", sum_primes)

if __name__ == "__main__":
    main()



'''
Question 8:

Write a Python program to print the Fibonacci sequence up to n terms. The Fibonacci series is a sequence of 
numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1.

i.e 
'''

def fibonacci_sequence(n):
    fibonacci = [0, 1]
    for i in range(2, n):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    return fibonacci





'''
Question 9:

Write a Python program to print the following pattern:
''' 


def print_pyramid_triangle(rows):
    for i in range(1, rows + 1):
        print(" " * (rows - i) + "*" * (2 * i - 1))

def main():
    rows = int(input("Enter the number of rows for the pyramid triangle: "))
    print("Pyramid Triangle:")
    print_pyramid_triangle(rows)

if __name__ == "__main__":
    main()



'''
Question 10:

Write a Python program to print the following pattern:
''' 




def print_alternating_letters_pyramid(rows):
    letters = ['A', 'B', 'C', 'D', 'E']
    letter_index = 0
    for i in range(1, rows + 1):
        print(" " * (rows - i) + " ".join(letters[:i]) + " " + " ".join(reversed(letters[:i-1])))
        
# dado n entero
# regresa 'Fizz' si n es divisible entre 3
# regresa 'Buzz' si n es divisible entre 5
# regresa 'FizzBuzz' si n es divisible entre 3 y 5

def fizzbuzz(n):
    ans = []
    for i in range(1,n+1):
        if i%3==0 and i%5==0:
            ans.append("FizzBuzz")
        elif i%3==0:
            ans.append("Fizz")
        elif i%5==0:
            ans.append("Buzz")
        else:
            ans.append(i)

    return ans

if __name__ == "__main__":
    print(fizzbuzz(100))
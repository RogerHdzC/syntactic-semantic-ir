// Factorial: simple linear recursion
int factorial(int n) {
    if (n <= 1) return 1;
    else return n * factorial(n - 1);
}

// Fibonacci: branching recursion
int fibonacci(int n) {
    if (n <= 0) return 0;
    else if (n == 1) return 1;
    else return fibonacci(n - 1) + fibonacci(n - 2);
}

// Power: recursion with two parameters
int power(int base, int exp) {
    if (exp == 0) return 1;
    else return base * power(base, exp - 1);
}

// GCD: Euclidean recursion
int gcd(int a, int b) {
    if (b == 0) return a;
    else return gcd(b, a % b);
}

// Mutual recursion example
int isEven(int n);
int isOdd(int n);

int isEven(int n) {
    if (n == 0) return 1;
    else return isOdd(n - 1);
}

int isOdd(int n) {
    if (n == 0) return 0;
    else return isEven(n - 1);
}

// SUM WITH WHILE
int sumWhile(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n;
        n--;
    }
    return sum;
}

// SUM WITH DO-WHILE
int sumDoWhile(int n) {
    int sum = 0;
    if (n > 0) {
        do {
            sum += n;
            n--;
        } while (n > 0);
    }
    return sum;
}

// SUM WITH FOR
int sumFor(int n) {
    int sum = 0;
    int i;
    for (i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}

// SWITCH–CASE EXAMPLE
int monthDays(int m) {
    int days;
    switch (m) {
        case 1:  days = 31; break;
        case 2:  days = 28; break;
        case 3:  days = 31; break;
        case 4:  days = 30; break;
        default: days = 0;  break;
    }
    return days;
}

// FLOAT CALCULATIONS
float circleArea(float r) {
    const float pi = 3.14159f;
    return pi * r * r;
}

float floatCalc(float a, float b) {
    // combinación de suma, resta, multiplicación, división en float
    return (a + b) * (a - b) / (a * b);
}

int main() {
    // Recursión
    printf("factorial(5)   = %d\n", factorial(5));
    printf("fibonacci(10)  = %d\n", fibonacci(10));
    printf("power(2, 8)    = %d\n", power(2, 8));
    printf("gcd(48, 18)    = %d\n", gcd(48, 18));
    printf("isEven(10)     = %d\n", isEven(10));
    printf("isOdd(10)      = %d\n\n", isOdd(10));

    // Bucles
    printf("sumWhile(10)   = %d\n", sumWhile(10));
    printf("sumDoWhile(10) = %d\n", sumDoWhile(10));
    printf("sumFor(10)     = %d\n\n", sumFor(10));

    // Switch–case
    printf("monthDays(2)   = %d\n", monthDays(2));
    printf("monthDays(4)   = %d\n\n", monthDays(4));

    // Cálculos en float
    printf("circleArea(2.5)= %f\n", circleArea(2.5f));
    printf("floatCalc(3.2,4.8)= %f\n", floatCalc(3.2f, 4.8f));

    return 0;
}

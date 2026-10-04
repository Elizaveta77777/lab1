"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    text = text.lower()
    count = 0
    for letter in text:
        if letter in "aeiou":
            count = count + 1
    return count

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    seen = []
    for symbol in text:
        if symbol in seen:
            return False
        seen.append(symbol)
    return True

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    binary = bin(number)
    count = 0
    for char in binary:
        if char == "1":
            count = count + 1
    return count

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        s = str(number)
        for i in range(len(s)):
            digit = int(s[i])
            product = product * digit
        number = product
        steps = steps + 1
    return steps

def mse(data: VectorPairInput) -> float:
    predicted = data.predicted
    expected = data.expected
    n = len(predicted)
    if n == 0:
        return 0.0
    total = 0
    for i in range(n):
        diff = predicted[i] - expected[i]
        total = total + diff * diff
    return total / n

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    d = 2
    while d * d <= number:
        while number % d == 0:
            if d in factors:
                factors[d] = factors[d] + 1
            else:
                factors[d] = 1
            number = number // d
        d = d + 1
    if number > 1:
        if number in factors:
            factors[number] = factors[number] + 1
        else:
            factors[number] = 1
    
    result = ""
    keys = sorted(factors.keys())
    for p in keys:
        if factors[p] == 1:
            result = result + "(" + str(p) + ")"
        else:
            result = result + "(" + str(p) + "**" + str(factors[p]) + ")"
    return result

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 1
    total = 0
    while total + k * k <= cube_count:
        total = total + k * k
        if total == cube_count:
            return k
        k = k + 1
    return "It is impossible"

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    if length % 2 == 0:
        mid = length // 2
        left = s[:mid]
        right = s[mid:]
    else:
        mid = length // 2
        left = s[:mid]
        right = s[mid + 1:]
    
    sum_left = 0
    for c in left:
        sum_left = sum_left + int(c)
    
    sum_right = 0
    for c in right:
        sum_right = sum_right + int(c)
    
    return sum_left == sum_right

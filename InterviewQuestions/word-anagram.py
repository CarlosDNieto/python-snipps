# función que valida si dos strings son anagramas

def anagram(s1, s2):
    c1 = {}
    c2 = {}
    def count_chars(s):
        h = {}
        for char in s:
            if char in h:
                h[char] += 1
            else:
                h[char] = 1
        return h
    return count_chars(s1) == count_chars(s2)


if __name__ == "__main__":
    print(anagram('word', 'wodr'))
    print(anagram('dog', 'dogg'))
    print(anagram('racecar', 'carrace'))
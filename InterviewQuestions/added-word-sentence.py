# dada una oración sent1 y la misma oración pero con una palabra añadida (sent2)

# regresa la palabra agregada

def extra_word(sent1, sent2):
    words1 = sent1.split()
    words2 = sent2.split()
    result = []

    for word in words2:
        if word not in words1:
            result.append(word)
        
    return " ".join(result)

if __name__ ==  "__main__":
    sent1 = "This is a dog"
    sent2 = "This is a fast dog"
    print(extra_word(sent1,sent2))
    sent1 = "es una perra muy buena"
    sent2 = "Lola es una perra muy buena"
    print(extra_word(sent1,sent2))
    sent1 = "Se me antojo un vaso de agua"
    sent2 = "Se me antojo un vaso de agua ahorita"
    print(extra_word(sent1,sent2))
    sent1 = "Checar si 2 strings son (tienen las mismas letras pero acomodadas dif)"
    sent2 = "Checar si 2 strings  son anagramas (tienen las mismas letras pero acomodadas dif)"
    print(extra_word(sent1,sent2))
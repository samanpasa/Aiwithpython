def my_split(sentence, separator):
    return sentence.split(separator)

def my_join(words, separator):
    return separator.join(words)

def main():
    sentence = input("Enter sentence: ")
    words = my_split(sentence, " ")
    print(",".join(words))
    for word in words:
        print(word)

if __name__ == "__main__":
    main()

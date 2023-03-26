with open("C:/Users/edly1/Documents/GitHub/CG4002-LaserTag/Output_Biases1.txt", "r") as text:
    text_data = text.readlines()
    print(type(text_data))
    for index, word in enumerate(text_data):
        word = word.strip()
        if "," in word:
            result = "{" + word
            if index == len(text_data)-1:
                result += "}"
            else:
                result += "},"
        else:
            result = word
            if index != len(text_data)-1:
                result += ","
        print(result, end="\n")

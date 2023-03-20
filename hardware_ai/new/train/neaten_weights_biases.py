with open("Output_Weights1.txt", "r") as text:
    text_data = text.readlines()
    for word in text_data:
        result = word.split()

        print(result)

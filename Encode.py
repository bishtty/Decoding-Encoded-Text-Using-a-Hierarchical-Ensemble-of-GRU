import random
import string
import csv

def read_random_string_from_file(file_path, size=50):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        start_index = random.randint(0, len(content) - size)
        random_string = content[start_index:start_index + size]
        return random_string.strip()

def preprocess_input(input_string):
    return input_string.upper()

def encode_string_multiply(input_string):
    encoded_string = ''
    
    for i, char in enumerate(input_string):
        if 'A' <= char <= 'Z':
            encoded_char = chr(((ord(char) - 65) * (i + 1)) % 26 + 65)
        else:
            encoded_char = char
        encoded_string += encoded_char
    
    return encoded_string, 'LavaMan'

def encode_string_add_sub(input_string):
    encoded_string = ''
    
    for i, char in enumerate(input_string):
        if 'A' <= char <= 'Z':
            encoded_char = chr(((ord(char) - 65 + i)) % 26 + 65)
        else:
            encoded_char = char
        encoded_string += encoded_char
    
    return encoded_string, 'Dipper'

def encode_string_xor(input_string):
    encoded_string = ''
    
    for i, char in enumerate(input_string):
        if 'A' <= char <= 'Z':
            key = i + 1  
            encoded_char = chr((ord(char) ^ key) % 26 + 65)
        else:
            encoded_char = char
        encoded_string += encoded_char
    
    return encoded_string, 'Custard'

def encode_string_affine(input_string, a=11, b=20):
    encoded_string = ''

    for char in input_string:
        if 'A' <= char <= 'Z':
            encoded_char = chr(((a * (ord(char) - 65) + b) % 26) + 65)
        else:
            encoded_char = char
        encoded_string += encoded_char

    return encoded_string, 'Anakin'

def write_to_csv(iteration, input_string, output_string, method_name):
    with open('data3.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['No.', 'Person', 'Said', 'Decipher']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'No.': iteration, 'Person': method_name, 'Said': output_string, 'Decipher': input_string})

def main():
    file_path = 'data0.txt'  # Change to the path of your .txt file containing random strings
    string_size = 200  # Change the size of the random string as needed
    iterations = 80000  # Change the number of iterations as needed

    for i in range(1, iterations + 1):
        random_string = read_random_string_from_file(file_path, string_size)
        processed_input = preprocess_input(random_string)
        choices = [1, 2, 3, 4]
        random_choice = random.choice(choices)

        if random_choice == 1:
            result, method_name = encode_string_multiply(processed_input)
        elif random_choice == 2:
            result, method_name = encode_string_add_sub(processed_input)
        elif random_choice == 3:
            random_key = random.randint(1, 100)
            result, method_name = encode_string_xor(processed_input)
        elif random_choice == 4:
            random_shift = random.randint(1, 10)
            result, method_name = encode_string_affine(processed_input, random_shift)

       # print(f"\nIteration {i}")
       # print("Person:", method_name)
        #print(f"Said: {result}")
        #print(f"Decipher: {random_string}")

        write_to_csv(i, random_string, result, method_name)

if __name__ == "__main__":
    main()

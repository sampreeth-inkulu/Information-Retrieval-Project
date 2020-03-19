import re
import sys
import pickle

# Helper functions
def func(x):
    return x[1]

# Defining compare function as the docs are custom sorted
def isLessThan(a, b):
    dir_a = ""
    i = 3
    while a[i] != '.':
        i += 1
    i += 1
    while a[i] != '.':
        dir_a += a[i] 
        i += 1
    
    dir_b = ""
    i = 3
    while b[i] != '.':
        i += 1
    i += 1
    while b[i] != '.':
        dir_b += b[i]
        i += 1
    
    if (dir_a < dir_b):
        return True
    elif (dir_a > dir_b):
        return False
    else:
        return a < b 

# Merge routine
def merge(listA, listB):
    
    i = 0
    j = 0
    lenA = len(listA)
    lenB = len(listB)

    result = []

    while i < lenA and j < lenB:

        if (listA[i] == listB[j]):
            result.append(listA[i])
            i += 1
            j += 1
        elif isLessThan(listA[i], listB[j]):
            i += 1
        else:
            j += 1
    
    return result

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment1_01_bool.py <path to model> <path to query file>")

    model_path = sys.argv[1]
    query_path = sys.argv[2]

    # Loading index
    with open(model_path, 'rb') as file:
        index = pickle.load(file)
    
    # Reading query file
    with open(query_path, 'r') as query_file:

        # Opening output file
        with open("Assignment1_01_results.txt", 'w') as result_file:

            # Reading query
            query_line = query_file.readline()
            while len(query_line) > 0:

                comma_pos = query_line.find(',')
                
                if comma_pos > 0:
                    # Query num and text
                    query_id = int( query_line[:comma_pos] )
                    query_text = query_line[comma_pos + 1:]

                    words = re.split(r'\W+', query_text)
                    
                    # To sort tokens by document frequency
                    word_and_freq = []
                    for word in words:
                        if len(word) > 0:
                            try:
                                word_and_freq.append([word, index[word][0]])
                            except:
                                pass
                    
                    word_and_freq = sorted(word_and_freq, key = func)
                    
                    # Obtaining result
                    try:
                        result = index[word_and_freq[0][0]][1]
                    except:
                        pass

                    for token in word_and_freq[1:]:
                        try:
                            result = merge(result, index[token[0]][1])
                        except:
                            pass

                    result_line = str(query_id) + " : "

                    for doc in result:
                        result_line += (doc + " ")

                    result_line += "\n"

                # Outputting result
                result_file.write(result_line)

                # Reading next query
                query_line = query_file.readline()
    
    print("Assignment1_01_results.txt was output")
        

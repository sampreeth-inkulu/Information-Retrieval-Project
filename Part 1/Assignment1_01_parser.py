import re
import sys
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment1_01_parser.py <path to the query file>")
    
    path = sys.argv[1]

    stop_words = set(stopwords.words('english'))

    # Reading query file
    with open(path, 'r') as f:
        content = f.read()
    
    words1 = re.findall('<num>[0-9]+</num>', content)
    words2 = re.findall("<title>[0-9A-Za-z'\-,. ?]+</title>", content)

    # Opening output file 
    with open('queries_01.txt', 'w') as f:
    
        for i in range(len(words1)):
            number = re.split(r'\W+', words1[i])

            # Query number
            query_line = number[2] + ", "

            # Query text tokenization
            words = word_tokenize(words2[i])
            
            words = words[3:-3]
            stop_words.add("'s")
            stop_words.add("'t")
            
            lemmatizer = WordNetLemmatizer()

            tokens = [(lemmatizer.lemmatize(w)).lower() for w in words if w not in stop_words and w not in string.punctuation]

            for token in set(tokens):

                query_line += token + " "
            
            query_line += "\n"

            # Writing output
            f.write(query_line)
    
    print("queries_01.txt was output")
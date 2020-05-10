import csv
import sys
from random import seed
from random import randint

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        sys.exit("No. of arguments mismatch")
    output_file_name = sys.argv[1]
    seed = int(sys.argv[2])

    with open("myRankedRelevantDocList.csv", 'r') as csv_file:
        reader = csv.reader(csv_file)

        with open(output_file_name, 'w') as output_file:
            writer = csv.writer(output_file)
            
            result = []
            for row in reader:
                out_row = []
                value = randint(0, 10)
                if value == 6:
                    out_row.append([int(row[0]), "something"])
                else:
                    out_row.append([int(row[0]), row[1]])
                result.append(out_row)
            writer.writerows(result)



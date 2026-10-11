import re
from sklearn import feature_extraction
requirements = "requirements-3nfr-80fr-2.txt"

def txttoList(txtFile):
    with open(txtFile, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    
    return lines

class textProcessing():
    def __init__(self, txt):
        self.text = [re.sub(r"[^a-z0-9\s]", " ", line.lower()) for line in txttoList(txt)]
        self.NFR = [line for line in self.text if line.startswith("nfr")]
        self.FR = [line for line in self.text if line.startswith("fr")]
        
def main():
    text = textProcessing(requirements)
    for line in text.NFR:
        print(line)
    for line in text.FR:
        print(line)
    
if __name__ == "__main__":
    main()
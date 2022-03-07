import sys
from utils import classify

try:
    filePath=sys.argv[1]
except Exception as E:
    print('enter file path as cmd arg')
    sys.exit(1)
def runner():
    classify(filePath)

if __name__ =='__main__':
    runner()

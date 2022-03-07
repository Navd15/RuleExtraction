## Files:
 - ### main.py - contains runner code 
 - ### utils.py - contains most of the code to read json, convert json to string, extract from string & write data to csv file
 - ### main.ipynb - contains initial code for testing
<hr>

### 1. `pip install -r requirements.txt`
### 2. `python -m spacy download en_core_web_sm`
### 3. `python main.py` `<path to json>`
### csv file will be saved to `cwd/output/{inputfilename}_output_classified.csv`
import json
import sys
import os
import spacy

import csv
from spacy.matcher import Matcher
from spacy.language import Language
import dotenv
dotenv.load_dotenv('.env')



def _decodeJSON(filePath,line_end='\n'):
    """
            Parameters:
                filePath (str): path of json file
                line_end (str) optional: char to insert at the end of each line
            
            Returns:
                ext_str (str): concatenated string
    """
    assert filePath; 'file path is required'
    ext_str=""
    try:
        filePath=os.path.normpath(filePath)
        with open(filePath,'r') as file:
            file_in=json.loads(file.read())
            for i in range(len(file_in['text_annotations'])):
                ext_str+=file_in['text_annotations'][i]['block_details']['block_description']+line_end 
        
    except Exception as E:
        print(E)
        sys.exit(1)
    else:
        return ext_str


def _writeCSV(doc,fileIn,fileOutName=os.getenv('OUT_FILE')):
    """
        Parameters:
            doc (spacy Doc): Doc object from spacy 
            fileIn (str): input file path
            fileOutName (str) optional:output file name   (default: OUT_FILE from .env)  
        Returns:
            filePath (str): output file name on success else None
    """
    assert type(doc)==spacy.tokens.doc.Doc, 'doc type not spacy.Doc'
    try:
        fileOutName=fileOutName.format(_getFileName(fileIn))
        filePath=os.path.join(os.getcwd()+'/output',fileOutName)
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        with open(filePath,'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['vendor_name','invoice_number','due_date','balance'])
            writer.writerow([i.text for i in doc.ents])
    except Exception as E:
        print('Error occured during writing csv')
        sys.exit(1)
    else:
        print(f'File {_getFileName(filePath)} written at {filePath}')
        return filePath

def _getFileName(filePath):
    head, tail = os.path.split(filePath)
    return tail or os.path.basename(head)

def _matcher(nlp):
    """
    Adds patterns to the default token matcher.
    Parameters:
        nlp (spacy.lang.en.English): spacy pretrained model
    Returns:
        matcher (spacy.matcher.matcher.Matcher): spacy matcher object
    """
    matcher = Matcher(nlp.vocab)
    # pattern for invoice number
    invoice_pat=[[{'text':{'in':['INVOICE','BILL']}},{'ORTH':{'IN':['#','@']}},{'IS_DIGIT':True}]]
    # pattern to extract balance/money statement
    balance_pat=[[{'IS_CURRENCY': True}, {'IS_SPACE': True, 'OP': '?'}, {'LIKE_NUM': True}]]
    # pattern to extract date eg DD-MM-YYY DD/MM/YYYY etc
    date_pat=[[{'text':{'REGEX':r'\b(\d{4}|\d{2})[\/\-]\d{2}[\/\-](\d{4}|\d{2})'}}],
    [{'IS_DIGIT': True,'length':{'in':[2,4]}}, {'ORTH': '-'}, {'IS_DIGIT': True,'length':2}, {'ORTH': '-'}, {'IS_DIGIT': True,'length':{'in':[2,4]}}],
    [{'IS_DIGIT': True}, {'ORTH': '-'}, {'IS_ALPHA': True}, {'ORTH': '-'}, {'IS_DIGIT': True}],
    [{'IS_ALPHA': True}, {'ORTH': '-'}, {'IS_DIGIT': True}, {'ORTH': '-'}, {'IS_DIGIT': True}]]
    # add matcher pattern for balance,date and invoice
    matcher.add('bal_matcher',balance_pat)
    matcher.add('dat_matcher',date_pat)
    matcher.add('inv_matcher',invoice_pat)
    return matcher

def classify(filePath):
    '''
    wrapper over the inner methods in util file. initializes nlp object
    and add the custom component to the default pipeline. calls the _matcher and 
    all other inner functions. 
        Parameters:
            filePath (str): input file path
        Returns:
            str : output file path on success
    '''
    assert filePath, "input file path required"
    nlp = spacy.load("en_core_web_sm")
    matcher=_matcher(nlp)
    # custom pipeline component to add custom NER to doc.ent 
    @Language.component("custom_ent")
    def custom_ent(doc):
        new_ents=[]
        matches=matcher(doc)
        saved_ents=set()
      # check for required matches
        for i in matches:
            string_id=nlp.vocab.strings[i[0]]
            # if first of its type match date, balance
            if string_id not in saved_ents:
                if string_id=='bal_matcher':
                    new_ent=('balance',i[1],i[2],)
                    saved_ents.add(string_id)
                    new_ents.append(new_ent)
                    continue
                if string_id=='dat_matcher' :
                    new_ent=('due_date',i[1],i[2],)
                    saved_ents.add(string_id)
                    new_ents.append(new_ent)
                    continue
                if string_id=='inv_matcher' :
                    new_ent=('invoice_number',i[1]+2,i[2],)
                    saved_ents.add(string_id)
                    new_ents.append(new_ent)
                    continue
        # check for organization ent from the default statstical model of spacy
        for ents in doc.ents:
            if ents.label_ and (ents.label_ not in saved_ents):
                new_ent=('vendor_name',ents.start,ents.end)
                new_ents.append(new_ent)
                break
            # override the default ents
        doc.ents=new_ents
        return doc
    # add new component to nlp pipeline after NER
    nlp.add_pipe('custom_ent',after='ner')
    doc=nlp(_decodeJSON(filePath))
    return _writeCSV(doc,filePath)
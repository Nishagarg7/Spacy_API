import json
from flask import render_template,redirect, url_for, request,Flask
#from flask_sqlalchemy import SQLAlchemy
import nltk
import spacy
import classy_classification
from spacy.tokens import SpanGroup
nlp = spacy.load("en_core_web_md")
#nltk.download('omw-1.4')
app = Flask(__name__)



text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
#doc = nlp(text)
@app.route('/attribute_ruler', methods = ['POST'])
def attribute_ruler():
    d=[]
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[
                {"text": "each"}, {"text": "player"}
            ]]
    attrs = {"TAG": "NNP", "POS": "PROPN"}
    ruler.add(patterns=patterns,attrs=attrs, index=1)
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.pos_,token.tag_))
    return d

@app.route('/lemma', methods = ['POST'])    
def lemmatizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.lemma_))
    return d 
    
@app.route('/morph', methods = ['POST'])    
def morphologizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),str(token.morph)))
    return d 

@app.route('/tagger', methods = ['POST'])    
def tagger():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.tag_,token.pos_))
    return d 

@app.route('/dependency_parser', methods = ['POST'])    
def parser():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.dep_))
    return d 


@app.route('/entity_r', methods = ['POST'])    
def entity_recognizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for word in doc.ents:
            d.append((word.text,word.label_))
    return d 


@app.route('/entity_ruler', methods = ['POST'])
def entity_ruler():
    d=[]
    patterns = [
                {"label": "Game", "pattern": "Backgammon"}
            ]  
    ruler = nlp.add_pipe("entity_ruler",after='ner') 
    ruler.add_patterns(patterns)
     
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for word in doc.ents:
            d.append((word.text,word.label_))
    return d

@app.route('/entity_linker', methods = ['POST'])
def entity_linker():
    d=[]
    #nlp.add_pipe("entityLinker", last=True)
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        all_linked_entities = doc._.linkedEntities
        #for sent in doc.sents:
            #d.append(sent._.linkedEntities)
    return all_linked_entities 


@app.route('/sentencizer', methods = ['POST'])
def sentencizer():
    sents_list = []
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for sent in doc.sents:
            sents_list.append(sent.text)
    return sents_list   

def spanCategorizer(text):
    
    #nlp.add_pipe("spancat")
    doc=nlp(text)
    #print(doc.spans[spanCategorizer])
    spans = doc.spans[spanCategorizer]
    for span, confidence in zip(spans, spans.attrs["scores"]):
        print(span.label_, confidence)

@app.route('/textCategorizer', methods = ['POST'])
def textCategorizer():
    data = {
        "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
        "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
           }
    nlp.add_pipe("text_categorizer", 
        config={
            "data": data,
            "model": "spacy"
               }
                )
    doc=nlp(text)
    return(doc._.cats)

#lemmatizer(doc)
#morphologizer(doc)
#tagger(doc)
#dependency_parser(doc)
#entity_recognizer(doc)
#entity_ruler(doc)
#attribute_ruler(doc)
#entity_linker(doc)
#sentencizer(doc)
#spanCategorizer(text)
#textCategorizer(text)
#to run the app in debug mode
if __name__ == "__main__":
    app.run(debug = True)
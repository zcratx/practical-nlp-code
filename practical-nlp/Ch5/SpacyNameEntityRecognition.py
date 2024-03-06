import spacy
nlp = spacy.load("en_core_web_lg")

mytext = open('../Data/myarticle.txt').read()

doc = nlp(mytext)
for ent in doc.ents:
    print(ent.text, "\t", ent.label_)
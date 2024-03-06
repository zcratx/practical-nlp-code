import spacy
import textacy

from textacy.extract import keyterms




from textacy import make_spacy_doc

en = textacy.load_spacy_lang("en_core_web_sm")

mytext = open('Data/nlphistory.txt').read()

#convert the text into a spacy document.
doc = textacy.make_spacy_doc(mytext, lang=en)

keyterms.textrank(doc, topn=5)

#textacy.spacier.core.textrank(doc, topn=5)

#Print the keywords using TextRank algorithm, as implemented in Textacy.
print("Textrank output: ", [kps for kps, weights in keyterms.textrank(doc, normalize="lemma", topn=5)])\
#Print the key words and phrases, using SGRank algorithm, as implemented in Textacy
print("SGRank output: ", [kps for kps, weights in keyterms.sgrank(doc, topn=5)])
import spacy
import textacy

from textacy.extract import keyterms, utils




from textacy import make_spacy_doc

en = textacy.load_spacy_lang("en_core_web_sm")

#mytext = open('Data/nlphistory.txt').read()
mytext = open('Data/myarticle.txt').read()

#last part of the document is generally filled with information that we don't need
# so we take them out
textlength = mytext.__len__()
mytext = mytext[:textlength-20]

#convert the text into a spacy document.
doc = textacy.make_spacy_doc(mytext, lang=en)

keyterms.textrank(doc, topn=5)

#textacy.spacier.core.textrank(doc, topn=5)

#Print the keywords using TextRank algorithm, as implemented in Textacy.
print("Textrank output: ", [kps for kps, weights in keyterms.textrank(doc, normalize="lemma", topn=10)])\
#Print the key words and phrases, using SGRank algorithm, as implemented in Textacy
print("SGRank output: ", [kps for kps, weights in keyterms.sgrank(doc, topn=10)])


#To address the issue of overlapping key phrases, textacy has a function: aggregage_term_variants.
#Choosing one of the grouped terms per item will give us a list of non-overlapping key phrases!
#BELOW ONE WORKS TOO
#terms = set([term for term,weight in keyterms.sgrank(doc)])

terms = set([term for term,weight in keyterms.textrank(doc)])
print(utils.aggregate_term_variants(terms))
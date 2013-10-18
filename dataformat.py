from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import sys
import re
import os

def process_review(review): 
 review=review.lower()
 #Convert www.* or https?://* to ''
 review = re.sub('((www\.[\S]+)|(https?://[^\S]+))','',review)
  #Convert @username to ''
 review = re.sub('@[^\S]+','',review)
  #Remove additional white spaces
 review = re.sub('[\s]+', ' ', review)
  #Replace #word with word
 review = re.sub(r'#([^\S]+)', r'\1', review)
 review = review.strip('\'"')
 review = re.sub('(([\@]\S*))','',review)
 review = re.sub('(([\?]))','',review)
 review = re.sub('(([\!]))','',review)
 review = re.sub('(([\:]))','',review)
 review = re.sub('(([\)|\(]))',' ',review)
 review=re.sub(r'([\d]+)','',review)
 review=re.sub(r'([\-])','',review)
 review=re.sub(r'([\<]\S*\>)','',review) 
 pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
 return pattern.sub(r"\1\1", review)

def extract_words(text):
 stemmer = PorterStemmer()
 tokenizer = WordPunctTokenizer()
 tokens = tokenizer.tokenize(text) 
 result =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
 return result 
    
def getfeaturevector(review):
 featureVector=[]
 words=extract_words(review)
 for w in words:
  w = w.strip('\'"?,.')
  val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
  if (val is None):
   continue
  else:
   featureVector.append(w.lower())
 return featureVector
 
def formatted_dataset(data):
 data=data[1:]
 reviews=[]
 for line in data:
  rating=int(line[2])
  if rating <3: sentiment=0 #'negative'
  elif rating >=3 and rating <=5: sentiment=1 #'normal'and 'postive'
#  elif rating >3.5 and rating <=5 : sentiment=2 #'positive'
  review=line[3]
  processedReview=process_review(review)
  fvector=getfeaturevector(processedReview)
  reviews.append((fvector,sentiment))
 return reviews
    
def globalFeatureVector(reviews):
 #this function will make a list of global features from the formatted_dataset
 #dic_feature={}
 #reviews=formatted_dataset()
 #reviews=balance_dataset(reviews)
 temp=[li for line in reviews for li in line]
 #for key in temp: dic_feature[key]='TRUE'
 return list(set(temp))
 
def transformToGFV(reviewFeatureVector,gfeaturelist):
 unique_words=set(reviewFeatureVector)
 g_reviewfeaturedic=[]
 val=0
 for word in gfeaturelist:
  if word in unique_words: val=1
  else: val=0
  g_reviewfeaturedic.append(val)
 return g_reviewfeaturedic


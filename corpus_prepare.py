import os, os.path
import dataformat
try:
   import cPickle as pickle
except:
   import pickle

class Corpus:
  def __init__(self):
    print "enter corpus directory path"
    self.path = raw_input()
    self.corpus = [] #list of list 
    self.BOW =[] #bag of words
    self.main()
  
  def read_corpus(self):
    files_in_dir = os.listdir(self.path)
    for directory in files_in_dir:
      if os.path.isdir(self.path+'/'+directory):
        i=0
        for f in os.listdir(self.path+'/'+directory):
          if i<100:
           file_des=open(self.path+'/'+directory +"/"+f)
           data_list=file_des.readlines() #list containing data points
           data = ' '.join(data_list)
           processed_data=dataformat.process_review(data)
           fvector=dataformat.getfeaturevector(processed_data) #feature vector for the document
           self.corpus.append(fvector)
           print str(i)+" "+ f
          i+=1
   
  def getGlobalFeatureVector(self):
    self.globalFeatureVector=dataformat.globalFeatureVector(self.corpus)
  
  def bagOfWords(self):
    for doc in self.corpus:
      self.BOW.append(dataformat.transformToGFV(doc,self.globalFeatureVector))   
 
  def main(self):
    self.read_corpus()
    self.getGlobalFeatureVector()
    self.bagOfWords() 
    bagofwords=open('bag_of_words.pkl','wb')
    gfv=open('global_feature_vector.pkl','wb')
    pickle.dump(self.BOW,bagofwords)
    pickle.dump(self.globalFeatureVector,gfv)
    bagofwords.close()
    gfv.close()
if __name__ == "__main__":
 obj=Corpus()
    

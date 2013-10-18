import corpus_prepare
import pymf
import numpy as np
import pickle
corpus=corpus_prepare.Corpus()
bag_of_words= np.array(corpus.BOW)
#bag_of_words = np.random.normal(6,1,1000000).reshape(1000,1000)
print bag_of_words.shape
#nmf_mdl = pymf.NMF(bag_of_words, num_bases=10, niter=10)
#nmf_mdl.initialization()
#nmf_mdl.factorize()
#print nmf_mdl.W.shape
#print nmf_mdl.H.shape

m = pymf.SVD(bag_of_words, show_progress=False, rrank=0, crank=0)
m.factorize()
print m.S.shape
#for i in range(m.S.shape[0]):
# print m.S[i,i]
print m.U.shape
print "________________"
print m.V.shape
U_lowrank = open('U_lowrank.pkl','wb')
S_lowrank = open('S_lowrank.pkl','wb')
V_lowrank = open('V_lowrank.pkl','wb')
pickle.dump(m.U,U_lowrank)
pickle.dump(m.S,S_lowrank)
pickle.dump(m.V,V_lowrank)
U_lowrank.close()
S_lowrank.close()
V_lowrank.close()



# coding: utf-8
import numpy as np
import pandas as pd
from scipy import optimize
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk import FreqDist
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
import json,random,math,logging
from collections import defaultdict
import string
import heapq
from scipy import rank

from utils import *
from config import config
modelDataDir = config.lrr_in_root_dir
import sys
print(sys.version)

# LRR
class LRR(object):
    def __init__(self,beta_):

        input_root = config.lrr_in_root_dir
        self.vocab = []
        self.vocab = load_json(input_root+"vocab.json")
        
        self.aspectKeywords={}
        self.aspectKeywords = load_json(input_root+"aspectKeywords.json")
        
        # word to its index in the corpus mapping
        self.wordIndexMapping={}
        self.reversewordIndexMapping={}
        self.createWordIndexMapping()
        
        # aspect to its index in the corpus mapping
        self.aspectIndexMapping={} 
        self.reverseAspIndexmapping={}
        self.createAspectIndexMapping()
        
        # wList 
        # [ 
        #     { "location": {"downtown":10,"traffic":0,},"value":{} }, # aspect="location"
        #     { "location": {"distance":8} },
        # ]
        self.wList=[]
        self.wList = load_json(input_root+"wList.json")

        
        # [
        #     {"Overall":4.6}, 
        #     {"Overall":2.1},
        # ]
        self.ratingsList = []
        self.ratingsList = load_json(input_root+"ratingsList.json")
        
        # ID[19938412,312442345,134992345,......]
        self.reviewIdList=[]
        self.reviewIdList = load_json(input_root+"reviewIdList.json")
        
        # number of reviews in the corpus 
        self.R = len(self.reviewIdList)
        
        # breaking dataset into 3:1 ratio, 3 parts for training and 1 for testing
        self.trainIndex = random.sample(range(0, self.R), int(config.train_test_split_ratio*self.R))
        self.testIndex = list(set(range(0, self.R)) - set(self.trainIndex))

        # number of aspects
        self.k = len(self.aspectIndexMapping)
        
        # number of training reviews in the corpus
        # train
        self.Rn = len(self.trainIndex)
        
        # vocab size 
        self.n = len(self.wordIndexMapping)
        
        # delta - is simply a number
        self.delta = 1.0 
        
        # matrix of aspect rating vectors (Sd) of all reviews k * Rn
        self.S = np.empty(shape=(self.k, self.Rn), dtype=np.float64)
        
        # matrix of alphas (Alpha-d) of all reviews - k * Rn
        # each column represents Aplha-d vector for a review # dirichlet
        # k * Rn
        self.alpha = np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)
        for i in range(self.Rn-1):
            self.alpha = np.hstack((self.alpha, np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)))
        
        # vector mu - k*1 vector
        self.mu = np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)
        
        # matrix Beta for the whole corpus (for all aspects, for all words) - k * n matrix
        self.init_beta(beta_)
        # Following is help taken from: S = W*Wt + b
        # https://stats.stackexchange.com/questions/124538/
        W = np.random.randn(self.k, self.k-1) # k*(k-1)
        S = np.add(np.dot(W, W.transpose()), np.diag(np.random.rand(self.k))) # k*k
        D = np.diag(np.reciprocal(np.sqrt(np.diagonal(S)))) # k*k 
        self.sigma = np.dot(D, np.dot(S, D)) # k*k
        self.sigmaInv = np.linalg.inv(self.sigma) #
        
        # setting up logger
        if os.path.exists(config.lrr_log_file): os.remove(config.lrr_log_file) 
        self.logger = logging.getLogger("LRR")
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(config.lrr_log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    # beta value
    def init_beta(self,beta_):
        self.beta = np.random.uniform(low=-0.1, high=0.1, size=(self.k, self.n))
        for word,value in beta_.items():
            word_idx = self.wordIndexMapping.get(word)
            if word_idx is None: continue
            for i in range(self.k):
                self.beta[i][word_idx] = value
        self.beta = config.max_word_polarities*np.tanh(self.beta)

    # word->id
    def createWordIndexMapping(self):
        for i,word in enumerate(self.vocab,0):
            self.wordIndexMapping[word] = i
            self.reversewordIndexMapping[i] = word
        # print(self.wordIndexMapping)

    # aspect->id and id->aspect
    def createAspectIndexMapping(self):
        for i,aspect in enumerate(self.aspectKeywords.keys()):
            self.aspectIndexMapping[aspect] = i
            self.reverseAspIndexmapping[i] = aspect
        # print(self.aspectIndexMapping)
    
    # given a dictionary as in every index of self.wList, 
    # creates a W matrix as was in the paper
    # W(k*n)
    def createWMatrix(self, w):
        W = np.zeros(shape=(self.k, self.n))
        for aspect, Dict in w.items():
            for word, freq in Dict.items():
                aspect_id = self.aspectIndexMapping[aspect]
                word_id = self.wordIndexMapping[word]
                W[aspect_id][word_id] = freq
        # sigma( word_freq ) = 1.0
        W = np.array( map(lambda x: np.zeros(x.shape) if np.sum(x)==0 else 1.0*x/np.sum(x),W) ) 
        # print(W); exit();
        return W
    
    # Computing aspectRating array for each review given Wd->W matrix for review 'd'
    def calcAspectRatings(self,Wd):
        # Sd = np.einsum('ij,ij->i',self.beta,Wd).reshape((self.k,)) 
        Sd = np.sum(self.beta*Wd,axis=1)
        # print(Sd)
        max_asp_rating = config.aspect_max_rating
        max_word_polarities = config.max_word_polarities
        try: 
            Sd =  max_asp_rating *( Sd + max_word_polarities)/ (2 * max_word_polarities)
            # Sd = 1.0*max_asp_rating/(1.0 + np.exp(-1.0*Sd)) 
        except Exception as inst:
            self.logger.info("Exception in calcAspectRatings : %s", Sd)
        fun = lambda x: 0.0 if abs(x) < config.eps else x 
        return np.array(map(fun,Sd))

    # update mu
    # calculates mu for (t+1)th iteration
    def calcMu(self): 
        self.mu = np.sum(self.alpha, axis=1).reshape((self.k, 1)) / self.Rn
    
    # update diagonal entries only
    # updatesigma 
    def calcSigma(self, updateDiagonalsOnly):
        self.sigma.fill(0) 
        for i in range(self.Rn):
            columnVec = self.alpha[:, i].reshape((self.k, 1))
            columnVec = columnVec - self.mu
            if updateDiagonalsOnly:
                for k in range(self.k):
                    self.sigma[k][k] += columnVec[k]*columnVec[k]
            else:
                self.sigma = self.sigma + np.dot(columnVec, columnVec.transpose())
        for i in range(self.k):
                self.sigma[i][i] = (1.0+self.sigma[i][i])/(1.0+self.Rn)
        self.sigmaInv = np.linalg.inv(self.sigma)
    
    #  overall rating
    def calcOverallRating(self,alphaD,Sd):
        # overall = (aspect weight)^T * (aspect rating)
        overall = np.dot(alphaD.transpose(),Sd)[0][0] 
        overall = overall*config.Overall_max_rating/config.aspect_max_rating 
        return 0.0 if abs(overall) < config.eps else overall

    def calcDeltaSquare(self):
        self.delta = 0.0
        for i in range(self.Rn): 
            alphaD = self.alpha[:,i].reshape((self.k, 1)) 
            Sd = self.S[:,i].reshape((self.k, 1)) 
            #i overall rating
            Rd = float(self.ratingsList[self.trainIndex[i]]["Overall"]) 
            temp = Rd - self.calcOverallRating(alphaD,Sd)
            # print(Rd,self.calcOverallRating(alphaD,Sd),temp)
            try: self.delta += (temp*temp)
            except Exception:
                self.logger.info("Exception in Delta calc")
        self.delta /= self.Rn
    
    # calcBeta 
    def maximumLikelihoodBeta(self,x,*args):
        beta = x
        beta = beta.reshape((self.k,self.n))
        innerBracket = np.empty(shape=self.Rn)
        for d in range(self.Rn):
            tmp = 0.0
            rIdx = self.trainIndex[d] #review index in wList
            for i in range(self.k):
                W = self.createWMatrix(self.wList[rIdx])
                tmp += self.alpha[i][d]*np.dot(beta[i, :].reshape((1, self.n)), W[i, :].reshape((self.n, 1)))[0][0]
            innerBracket[d] = tmp - float(self.ratingsList[rIdx]["Overall"])
        mlBeta = 0.0
        for d in range(self.Rn):
            mlBeta += innerBracket[d] * innerBracket[d]
        return mlBeta / (2*self.delta)
    
    # calcBeta 
    def gradBeta(self,x,*args):
        beta = x
        beta = beta.reshape((self.k,self.n))
        gradBetaMat = np.empty(shape = ((self.k,self.n)), dtype='float64')
        innerBracket = np.empty(shape = self.Rn)
        for d in range(self.Rn):
            tmp = 0.0
            rIdx = self.trainIndex[d] # review index in wList
            for i in range(self.k):
                W = self.createWMatrix(self.wList[rIdx])
                tmp += self.alpha[i][d] * np.dot(beta[i, :].reshape((1, self.n)), W[i, :].reshape((self.n, 1)))[0][0]
            innerBracket[d] = tmp - float(self.ratingsList[rIdx]["Overall"])
        
        for i in range(self.k):
            beta_i=np.zeros(shape=(1,self.n))
            for d in range(self.Rn):
                rIdx = self.trainIndex[d] #review index in wList
                W = self.createWMatrix(self.wList[rIdx])
                beta_i += innerBracket[d] * self.alpha[i][d] *  W[i, :]
            gradBetaMat[i,:] = beta_i
        return gradBetaMat.reshape((self.k*self.n, ))
    
    # update beta(k * n) 
    def calcBeta(self):
        # 
        beta, retVal, flags = optimize.fmin_l_bfgs_b(
            func = self.maximumLikelihoodBeta, # Function to minimise. 目标函数
            x0 = self.beta, # Initial guess.
            fprime = self.gradBeta, # The gradient of func. 
            args = (), 
            m = 5, # The maximum number of variable metric corrections used to define the limited memory matrix.
            maxiter = 30000, 
        )
        converged = True
        if flags['warnflag'] != 0: converged = False
        self.logger.info("Beta converged : %d", flags['warnflag'])
        beta = config.max_word_polarities*np.tanh(beta) # range [-1,1]
        return beta.reshape((self.k,self.n)), converged
    
    # calcAlphaD 
    def maximumLikelihoodAlpha(self, x, *args):
        alphad = x
        alphad = alphad.reshape((self.k, 1))
        rd,Sd,deltasq,mu,sigmaInv = args
        temp1 = (rd-np.dot(alphad.transpose(),Sd)[0][0])
        temp1 *= temp1
        temp1 /= (deltasq*2)
        temp2 = (alphad-mu)
        temp2 = np.dot(np.dot(temp2.transpose(),sigmaInv),temp2)[0][0]
        temp2 /= 2
        return temp1 + temp2
    
    # calcAlphaD 
    def gradAlpha(self, x,*args):
        alphad = x
        alphad = alphad.reshape((self.k, 1))
        rd,Sd,deltasq,mu,sigmaInv = args
        temp1 = (np.dot(alphad.transpose(),Sd)[0][0]-rd)*Sd 
        temp1 /= deltasq
        temp2 = np.dot(sigmaInv,(alphad-mu))
        return (temp1 + temp2).reshape((self.k,))
    
    # update alphaD(k * 1) 
    def calcAlphaD(self,i):
        alphaD = self.alpha[:,i].reshape((self.k,1))
        rIdx = self.trainIndex[i]
        rd = float(self.ratingsList[rIdx]["Overall"]) 
        Sd = self.S[:,i].reshape((self.k,1))
        Args = (rd,Sd,self.delta,self.mu,self.sigmaInv)
        bounds = [(0,1)]*self.k
        #self.gradf(alphaD, *Args)
        alphaD, retVal, flags = optimize.fmin_l_bfgs_b(
            func = self.maximumLikelihoodAlpha, 
            x0 = alphaD, 
            fprime = self.gradAlpha,
            args = Args,
            bounds = bounds,
            m = 5, 
            maxiter = 30000, 
        )
        converged = True
        if flags['warnflag']!=0: converged = False
        # self.logger.info("Alpha Converged: %d", flags['warnflag'])
        #Normalizing alphaD so that it follows dirichlet distribution
        # alphaD = 1.0/(1.0+np.exp(-1.0*alphaD)) 
        alphaD = np.exp(alphaD) 
        alphaD = alphaD / np.sum(alphaD) 
        return alphaD.reshape((self.k,)), converged
    
    # beta
    def betaLikelihood(self):
        likelihood = 0.0
        for d in range(self.Rn): 
            rIdx = self.trainIndex[d]
            Rd = float(self.ratingsList[rIdx]["Overall"]) 
            W = self.createWMatrix(self.wList[rIdx]) 
            Sd = self.calcAspectRatings(W).reshape((self.k, 1))
            alphaD = self.alpha[:,d].reshape((self.k, 1))
            temp = Rd - self.calcOverallRating(alphaD,Sd)
            try: likelihood += (temp * temp)
            except Exception: self.logger.debug("Exception in betaLikelihood")
        likelihood /= self.delta
        return likelihood
    
    # alpha
    def alphaLikelihood(self):
        likelihood = 0.0
        for d in range(self.Rn): 
            alphad = self.alpha[:,d].reshape((self.k, 1))
            temp2 = (alphad - self.mu)
            temp2 = np.dot(np.dot(temp2.transpose(),self.sigmaInv),temp2)[0] 
            likelihood += temp2
        try: likelihood += np.log(np.linalg.det(self.sigma))
        except FloatingPointError:
            self.logger.debug("Exception in alphaLikelihood: %f", np.linalg.det(self.sigma))
        return likelihood
    
    # alpha、beta and delta
    def calcLikelihood(self):
        likelihood = 0.0
        likelihood += np.log(self.delta) # delta likelihood
        likelihood += self.betaLikelihood() # data likelihood - will capture beta likelihood too
        likelihood += self.alphaLikelihood() # alpha likelihood
        return likelihood
    
    # alpha(k*Rn)、S(k*Rn)
    def EStep(self):
        for i in range(self.Rn): 
            rIdx = self.trainIndex[i]
            W = self.createWMatrix(self.wList[rIdx]) 
            self.S[:,i] = self.calcAspectRatings(W) 
            alphaD, converged = self.calcAlphaD(i) 
            if converged: self.alpha[:,i] = alphaD 
            # self.logger.info("Alpha calculated")
    
    # M
    def MStep(self):
        likelihood = 0.0
        self.calcMu() 
        self.logger.info("Mu calculated")
        self.calcSigma(False) #  sigma
        self.logger.info("Sigma calculated : %s " % np.linalg.det(self.sigma))
        likelihood += self.alphaLikelihood() #alpha likelihood
        self.logger.info("alphaLikelihood calculated") 
        beta, converged = self.calcBeta() # beta(k * n) 
        if converged: self.beta = beta
        self.logger.info("Beta calculated")
        likelihood += self.betaLikelihood() # data likelihood - will capture beta likelihood too
        self.logger.info("betaLikelihood calculated")
        self.calcDeltaSquare() # delta
        self.logger.info("Deltasq calculated")
        likelihood += np.log(self.delta) # delta likelihood
        return likelihood
            
    # EM 
    def em_algorithm(self, maxIter, coverge):
        self.logger.info("Training started")
        old_likelihood = self.calcLikelihood() # delta
        self.logger.info("initial calcLikelihood calculated, det(Sig): %s" % np.linalg.det(self.sigma)) 
        diff = 10.0; iteration = 0
        while(iteration < min(8, maxIter) or (iteration<maxIter and diff>coverge)):
            self.EStep() # E alpha(k*Rn)、S(k*Rn)
            self.logger.info("EStep completed") 
            likelihood = self.MStep() # M、beta
            self.logger.info("MStep completed")
            diff = abs((old_likelihood - likelihood) / old_likelihood)
            old_likelihood = likelihood
            self.logger.info("*"*20+" iter = %d. diff = %f "%(iteration,diff)+"*"*20)
            iteration += 1
        self.logger.info("Training completed")
    
    # output
    def output_alpha_and_S(self,dump_file):
        output = dict()
        for d in range(self.Rn):
            roomid = str(self.reviewIdList[d])
            rIdx = self.trainIndex[d]
            Rd = float(self.ratingsList[rIdx]["Overall"]) 
            W = self.createWMatrix(self.wList[rIdx]) 
            Sd = self.calcAspectRatings(W).reshape((self.k, 1))
            alphaD = self.alpha[:,d].reshape((self.k, 1))
            overallRating = self.calcOverallRating(alphaD,Sd)
            output[roomid] = {
                "aspect weight": list(alphaD.reshape(self.k,)),
                "Real Overall rating": Rd,
                "predict Overall rating": overallRating,
                "predict Aspect rating": list(Sd.reshape(self.k,)),
            }
        store_json(dump_file,output)
    
    
    def output_top_k(self,dump_file,k):
        output = dict();
        aspect_names = []
        for i in range(self.k): aspect_names.append( self.reverseAspIndexmapping[i] )
        beta_df = pd.DataFrame(self.beta.transpose(),columns=aspect_names)
        words = []
        for i in range(self.n): words.append( self.reversewordIndexMapping[i] )
        words_df = pd.DataFrame(words,columns=["word"])
        beta_df = pd.concat([beta_df,words_df],axis=1)
        for aspect in aspect_names:
            beta_df = beta_df.sort_values(aspect,ascending=False).reset_index(drop=True)
            pos_words = dict(beta_df.ix[:int(k),["word",aspect]].values)
            nag_words = dict(beta_df.ix[len(beta_df.index)-int(k):,["word",aspect]].values)
            output[aspect] = {"positive words":pos_words,"nagtive words":nag_words}
        store_json(dump_file,output)

    def testing_all_data(self):
        pred_y = []; real_y = [];
        for i in range(self.R):
            rIdx = self.trainIndex[i]
            W = self.createWMatrix(self.wList[rIdx])
            Sd = self.calcAspectRatings(W).reshape((self.k,1))
            overallRating = self.calcOverallRating(self.mu,Sd)
            room_pred = []; room_real = []
            for aspect, rating in self.ratingsList[rIdx].items():
                if aspect != "Overall" and (aspect in self.aspectIndexMapping.keys()):
                    r = self.aspectIndexMapping[aspect]
                    room_pred.append( Sd[r][0] )
                    room_real.append(rating)
            pred_y.append(room_pred);
            real_y.append(room_real);
            # print("Aspect:",aspect," Rating:",rating, "Predic:", Sd[r][0])
            # if overallRating > 3.0:   print("Positive Review")
            # else:                     print("Negative Review")
        # print(np.array(real_y).shape,np.array(pred_y).shape)
        # for i in range(len(real_y)): print(real_y[i])
        # for i in range(len(pred_y)): print(pred_y[i])
        output_evaluate(real_y,pred_y)

# boot-strap
class BootStrap(object):
    def __init__(self, readDataObj):
        self.corpus = readDataObj
        # Aspect, Word -> freq matrix - frequency of word in that aspect
        self.aspectWordMat = defaultdict(lambda: defaultdict(int)) 
        # Aspect --> total count of words tagged in that aspect
        # = sum of all row elements in a row in aspectWordMat matrix
        self.aspectCount = defaultdict(int)
        # Word --> frequency of jth tagged word(in all aspects) 
        # = sum of all elems in a column in aspectWordMat matrix
        self.wordCount = defaultdict(int)
        # Top p words from the corpus related to each aspect to update aspect keyword list
        self.p = 5
        self.iter = 7
        
        #List of W matrix
        self.wList = []
        #List of ratings Dictionary belonging to review class
        self.ratingsList = []
        #List of Review IDs
        self.reviewIdList = []
        
    def assignAspect(self, sentence): #assigns aspects to sentence
        sentence.assignedAspect = []
        count = defaultdict(int) #count used for aspect assignment as in paper
        #print("IN ASSIGN ASPECT FUNCTION:",len(sentence.wordFreqDict))
        for word in sentence.wordFreqDict.keys():
            for aspect, keywords in self.corpus.aspectKeywords.items():
                if word in keywords:
                    count[aspect] += 1
        if count: #if count is not empty
            maxi = max(count.values())
            for aspect, cnt in count.items():
                if cnt==maxi:
                    sentence.assignedAspect.append(aspect)
        if(len(sentence.assignedAspect)==1): #if only 1 aspect assigned to it
            self.corpus.aspectSentences[sentence.assignedAspect[0]].append(sentence)
            
    def populateAspectWordMat(self):
        self.aspectWordMat.clear()
        for aspect, sentences in self.corpus.aspectSentences.items():
            for sentence in sentences:
                for word,freq in sentence.wordFreqDict.items():
                    self.aspectWordMat[aspect][word]+=freq
                    self.aspectCount[aspect]+=freq
                    self.wordCount[word]+=freq
    
    def chiSq(self, aspect, word):
        # Total number of (tagged) word occurrences
        C = sum(self.aspectCount.values())
        # Frequency of word W in sentences tagged with aspect Ai
        C1 = self.aspectWordMat[aspect][word]
        # Frequency of word W in sentences NOT tagged with aspect Ai
        C2 = self.wordCount[word] - C1
        # Number of sentences of aspect A, NOT contain W
        C3 = self.aspectCount[aspect] - C1 
        # Number of sentences of NOT aspect A, NOT contain W
        C4 = C - C1
        deno = (C1+C3) * (C2+C4) * (C1+C2) * (C3+C4)
        # print(aspect, word, C, C1, C2, C3, C4)
        if deno!=0:
            return (C*(C1*C4 - C2*C3)*(C1*C4 - C2*C3))/deno
        else:
            return 0.0
        
    def calcChiSq(self):
        topPwords = {}
        for aspect in self.corpus.aspectKeywords.keys():
            topPwords[aspect] = []
        for word in self.corpus.wordFreq.keys():
            maxChi = 0.0 #max chi-sq value for this word
            maxAspect = "" #corresponding aspect
            for aspect in self.corpus.aspectKeywords.keys():
                self.aspectWordMat[aspect][word] = self.chiSq(aspect,word)
                if self.aspectWordMat[aspect][word] > maxChi:
                    maxChi = self.aspectWordMat[aspect][word]
                    maxAspect = aspect
            if maxAspect!="":
                topPwords[maxAspect].append((maxChi, word))
                
        changed=False
        for aspect in self.corpus.aspectKeywords.keys():
            for t in heapq.nlargest(self.p,topPwords[aspect]):
                if t[1] not in self.corpus.aspectKeywords[aspect]:
                    changed=True
                    self.corpus.aspectKeywords[aspect].append(t[1])
        return changed
    
    # Populate wList,ratingsList and reviewIdList
    def populateLists(self):
        for review in self.corpus.allReviews:
            #Computing W matrix for each review
            W = defaultdict(lambda: defaultdict(int))
            for sentence in review.sentences:
                if len(sentence.assignedAspect)==1:
                    for word,freq in sentence.wordFreqDict.items():
                        W[sentence.assignedAspect[0]][word] += freq
            if len(W)!=0:
                self.wList.append(W)
                self.ratingsList.append(review.ratings)
                self.reviewIdList.append(review.reviewId)  
                
    def bootStrap(self):
        changed=True
        while self.iter>0 and changed:
            self.iter-=1
            self.corpus.aspectSentences.clear()
            for review in self.corpus.allReviews:
                for sentence in review.sentences:
                    self.assignAspect(sentence)
            self.populateAspectWordMat()
            changed=self.calcChiSq()
        self.corpus.aspectSentences.clear()
        for review in self.corpus.allReviews:
            for sentence in review.sentences:
                self.assignAspect(sentence)
        # print(self.corpus.aspectKeywords)

# 
class ReadData(object):
    def __init__(self):
        self.aspectKeywords = {} # {"aspect_name":["keyword_name1",...]}
        self.stopWords = [] # 
        self.wordFreq = {} # dict with of all words and their freq in the corpus
        self.lessFrequentWords = set() # words which have frequency<5 in the corpus
        self.allReviews = [] # list of Review objects from the whole corpus
        self.aspectSentences = defaultdict(list) # aspect to Sentences mapping
        self.wordIndexMapping = {} # word to its index in the corpus mapping
        self.aspectIndexMapping = {} # aspect to its index in the corpus mapping
    
    # 
    def createWordIndexMapping(self):
        for i,word in enumerate(self.wordFreq.keys(),0):
            self.wordIndexMapping[word] = i
        # print(self.wordIndexMapping)
    
    # 
    def createAspectIndexMapping(self):
        for i,aspect in enumerate(self.aspectKeywords.keys(),0):
            self.aspectIndexMapping[aspect] = i
        #print(self.aspectIndexMapping)

    # step 1. seed words
    def readAspectSeedWords(self):
        with open(config.seedwords_file) as fd:
            seedWords = json.load(fd)
            for aspect in seedWords["aspects"]:
                self.aspectKeywords[aspect["name"]] = aspect["keywords"]

    # step 2. stopwords 
    def readStopWords(self):
        with open(config.stopwords_file) as fd:
            for stopWord in fd:
                self.stopWords.append(stopWord.strip())
        for stopWord in stopwords.words('english'):
            if stopWord not in self.stopWords:
                self.stopWords.append(stopWord)
        #print(self.stopWords)

    # sentence
    def stemmingStopWRemoval(self, review, vocab):
        reviewObj = Review()
        # copying ratings into reviewObj
        for ratingType, rating in review["Ratings"].items():
            reviewObj.ratings[ratingType] = rating
        reviewObj.reviewId = review["ReviewID"]
        
        stemmer = PorterStemmer() 
        reviewContent = review["Content"]
        #TODO: Append title too!
        sentencesInReview = nltk.sent_tokenize(reviewContent)
        puncs = set(string.punctuation) #punctuation marks 
        for sentence in sentencesInReview:
            wordList=[]
            words = nltk.word_tokenize(sentence)
            for word in words:
                if not all(c.isdigit() or c in puncs for c in word):
                    word = word.lower()
                    if word not in self.stopWords:
                        word=stemmer.stem(word.lower())
                        vocab.append(word)
                        wordList.append(word)
            if wordList:
                sentenceObj = Sentence(wordList)
                reviewObj.sentences.append(sentenceObj)
        if reviewObj.sentences:
            self.allReviews.append(reviewObj)
            # print(reviewObj)

    def readReviewsFromJson(self):
        ''' Reads reviews frm the corpus, calls stemmingStopWRemoval
        and creates list of lessFrequentWords (frequency<5)
        '''
        vocab = []
        with open(config.reviews_json_file) as fd:
            data = json.load(fd)
            for review in data["comments"]:
                self.stemmingStopWRemoval(review,vocab)
        self.wordFreq = FreqDist(vocab)
        for word,freq in self.wordFreq.items():
            if freq < 5:
                self.lessFrequentWords.add(word)
        for word in self.lessFrequentWords:
            del self.wordFreq[word]
        self.createWordIndexMapping()

    def removeLessFreqWords(self):
        emptyReviews = set()
        for review in self.allReviews:
            emptySentences = set()
            for sentence in review.sentences:
                deleteWords = set()
                for word in sentence.wordFreqDict.keys():
                    if word in self.lessFrequentWords:
                        deleteWords.add(word)
                for word in deleteWords:
                    del sentence.wordFreqDict[word]
                if not sentence.wordFreqDict:
                    emptySentences.add(sentence)
            review.sentences[:] = [x for x in review.sentences if x not in emptySentences]
            if not review.sentences:
                emptyReviews.add(review)  
        self.allReviews[:] = [x for x in self.allReviews if x not in emptyReviews]

# 
class Sentence(object):
    def __init__(self, wordList):
        self.wordFreqDict = FreqDist(wordList)#Dictionary of words in the sentence and corres. frequency
        self.assignedAspect = [] #list of aspects assigned to this sentence
    def __str__(self):
        return self.wordFreqDict.pformat(10000) + '##' + str(self.assignedAspect)

# 
class Review(object):
    def __init__(self):
        self.sentences = [] # list of objects of class Sentence
        self.reviewId = ""
        self.ratings = {} # true ratings provided by the user
        
    def __str__(self):
        retStr = ""
        for sentence in self.sentences: retStr += sentence.__str__() + '\n'
        retStr += "###"+self.reviewId+"###"+str(self.ratings)+"\n"
        return retStr

# 
class Table6_Modles():
    def __init__(self):
        self.aspect_columns = config.aspect_names
        self.pred_columns = ["pred_asp_%d"%(i) for i in range(len(self.aspect_columns))]
    
    def local_model(self,aspect_df):
        # aspect_df: roomid, overall_rating, aspect_rating
        aspect_r = aspect_df[self.aspect_columns].values
        pred_ary = []
        for idx in aspect_df.index:
            score = 1.0 * (aspect_df.ix[idx,"Overall"]) * config.aspect_max_rating / config.Overall_max_rating
            pred_ary.append([ score for i in range(len(self.pred_columns))])
        pred_ary = np.array(pred_ary)
        print("local model: "),; output_evaluate(aspect_r,pred_ary)

    def global_model(self,aspect_df):
        # aspect_df: roomid, overall_rating, aspect_rating
        aspect_r = aspect_df[self.aspect_columns].values
        pred_df = pd.DataFrame(np.zeros((len(aspect_r),len(aspect_r[0]))),columns=self.pred_columns)
        for i,column in enumerate(self.aspect_columns,0):
            pred_df[self.pred_columns[i]] = aspect_df[column].mean()
        pred_ary = pred_df[self.pred_columns].values
        print("global model: "),; output_evaluate(aspect_r,pred_ary)

    def SVR_O(self,aspect_df):
        aspect_r = aspect_df[self.aspect_columns].values
        pred_df = pd.DataFrame(np.zeros((len(aspect_r),len(aspect_r[0]))),columns=self.pred_columns)
        for column in self.pred_columns:
            pred_df[column] = aspect_df["Overall"]
        skf = StratifiedKFold( n_splits = 4, shuffle=True, random_state=2017 )
        pred_y = []; real_y = []
        trian_data = pred_df[self.pred_columns].values
        for i,(train_idx,vali_idx) in enumerate(skf.split(trian_data, np.zeros((len(aspect_r),))),1):
            tmp_pred = []
            for j in range(len(self.aspect_columns)):
                model = SVR(C = 100, kernel = 'rbf')
                model.fit(trian_data[train_idx],aspect_r[train_idx,j])
                tmp_pred.append( model.predict(trian_data[vali_idx]) )
            pred_y.extend( np.array(tmp_pred).T )
            real_y.extend( list(aspect_r[vali_idx]) )
        print("SVR-O: "),; output_evaluate(real_y,pred_y)

    def SVR_A(self,aspect_df):
        aspect_r = aspect_df[self.aspect_columns].values
        pred_df = pd.DataFrame(np.zeros((len(aspect_r),len(aspect_r[0]))),columns=self.pred_columns) 
        for column in self.pred_columns:
            pred_df[column] = aspect_df["Overall"]
        tmp_pred = []
        for j in range(len(self.aspect_columns)): 
            model = SVR(C = 100, kernel = 'rbf')
            model.fit(pred_df.values,aspect_r[:,j])
            tmp_pred.append( model.predict(pred_df.values) )
        pred_y = np.array(tmp_pred).T
        # print(pred_y.shape,aspect_r.shape)
        print("SVR-A: "),; output_evaluate(aspect_r,pred_y)





















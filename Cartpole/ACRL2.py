from pylab import *
import numpy as np
import random
import math
from tiles import *

class ACRL2():
	def __init__(self,gamma = 1, alphaSig = 0, alphaV = 0.1, alphaU = 0.01, lmbda = 0.75, n = 2048, logging = 0):
		print 'Gamma:', gamma
		print 'AlphaV:',alphaV
		print 'AlphaMean:', alphaU
		print 'AlphaSigma:',alphaSig

		self.logging = logging
		n = 4
		self.gamma = gamma		#for how much of the next value function to consider. Usually ~ 1 (TD-learning)
		    
		self.alphaV = alphaV    #learning rate for weights of value function
		self.alphaU = alphaU	#learning rate for weights of mean 
		self.alphaSig = alphaSig	#learning rate for weights of variance

		self.lmbda = lmbda    #for eligibility traces updates

		self.avgR = 0    	  #not really used so far in the code (higher the value, longer our episodes last)
		self.ev = np.zeros(n)  
		self.e_mu = np.zeros(n)
		self.e_sigma = np.zeros(n)

		self.w = np.zeros(n)   		#weights of value function          
		self.w_mu = np.zeros(n)  	#weights of mean 
		self.w_sigma = np.zeros(n)	#weights of sigma

		self.delta = 0.0
		self.R = 0.0 				#kinda pointless
		self.value = 0.0            #kinda pointless  
		self.nextValue = 0.0 		#kinda pointless

		self.compatibleFeatures_mu = np.zeros(n)
		self.compatibleFeatures_sigma = np.zeros(n)

		self.mean = 0.0
		self.sigma = 1.0

		self.action = 0.0 			#need to save the last action taken (useful while updateing eligibility traces of mean and sigma)
	
	def getAction(self,features):
		self.mean = 0.0           
		self.sigma = 0.0

		# print 'ACRL().getAction():features:', features
		for i, val in enumerate(features):
			self.mean += self.w_mu[i]*val
			self.sigma += self.w_sigma[i]*val
		self.sigma = exp(self.sigma)   
		if self.sigma == 0: 
			self.sigma = 1  	 	

		self.action = np.random.normal(self.mean,self.sigma)
		if self.logging: 
			print '--ACRL().getAction():Mean:',self.mean,' Sigma:',self.sigma, ' Action:',self.action,  1 if self.action > 0.7 else 0
		return 1 if abs(self.action) > 0.5 else 0

	def updates(self, reward, prev_state, new_state):
		self.nextValue = self.Value(new_state)   #need to use tile coding to convert the state-space into a higher dimensional space and have weights in that space
		self.value = self.Value(prev_state)

		if self.logging: print 'ACRL().updates(): Diff:', self.nextValue, self.value, self.gamma*self.nextValue - self.value
		self.delta = reward + self.gamma*self.nextValue - self.value

		self.update_EV(prev_state)   #to update eligibility traces, you need prev_state
		self.update_V()
		self.update_EU(prev_state)
		self.update_mu()
		self.update_Esigma(prev_state)
		self.update_sigma()

	def Value(self,features):
		val = 0.0
		for i,val in enumerate(features):
			val += self.w[i] * val
		return val 
	
	def update_EV(self, prev_state):
		self.ev = self.lmbda*self.ev + prev_state

	def update_V(self):
		self.w = self.w + self.alphaV*self.delta*self.ev

	def update_EU(self, prev_state):
		self.e_mu = self.lmbda*self.e_mu + (self.action-self.mean)* prev_state

	def update_mu(self):
		if self.logging: print 'ACRL().update_mu(): W_mu:',self.w_mu, ' Delta:',self.delta, ' Eligibility_Mu:',self.e_mu
		self.w_mu = self.w_mu + self.alphaU*self.delta*self.e_mu

	def update_Esigma(self, prev_state):
		self.e_sigma = self.lmbda*self.e_sigma + ((((self.action-self.mean)**2) * (self.sigma**2) ) - 1)*prev_state

	def update_sigma(self):
		self.w_sigma = self.w_sigma + self.alphaSig*self.delta*self.e_sigma
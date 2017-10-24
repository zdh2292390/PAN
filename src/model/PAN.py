# -*- coding: utf-8 -*-
import numpy as np
import sys
import datetime
sys.path.append('../../')
sys.path.append('../')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from create_prior_knowledge import create_prior
from embedding import Embedding

class Model(nn.Module):
	def __init__(self,type = "figer", encoder = "averaging", hier = False, feature = False, 
                    gaussian = False, margin = 1, negtive_size = 1, 
                    gaussian_dim = 20, regularize = False, minval = 1.0,
                    maxval = 5.0, bag_strategy = "one", bags = False):

		super(Model, self).__init__()
		# Argument Checking
		assert(encoder in ["averaging", "lstm", "attentive","type_att"])
		assert(type in ["figer", "gillick"])
		self.type = type
		self.encoder = encoder
		self.hier = hier
		self.feature = feature
		# Hyperparameters
		self.context_length = 10
		self.emb_dim = 300
		self.target_dim = 113 if type == "figer" else 89
		self.feature_size = 600000 if type == "figer" else 100000
		self.learning_rate = 0.001
		self.lstm_dim = 100
		self.att_dim  = 100 # dim of attention module
		self.feature_dim = 50 # dim of feature representation
		self.feature_input_dim = 70
		self.bags = bags
		self.bag_strategy = bag_strategy
		self.lstm_layers = 1
		if encoder == "averaging":
			self.rep_dim = self.emb_dim * 3
		else:
			self.rep_dim = self.lstm_dim * 2 + self.emb_dim
		if feature:
			self.rep_dim += self.feature_dim
		if self.encoder == "lstm":
			self.left_lstm = nn.LSTM(self.emb_dim,self.lstm_dim,self.lstm_layers)
			self.right_lstm = nn.LSTM(self.emb_dim,self.lstm_dim,self.lstm_layers)
		elif self.encoder == "attentive" or self.encoder == "type_att":
			self.left_lstm = nn.LSTM(self.emb_dim,self.lstm_dim,self.lstm_layers,bidirectional=True)
			self.right_lstm = nn.LSTM(self.emb_dim,self.lstm_dim,self.lstm_layers,bidirectional=True)
			# self.W_e = Variable(torch.randn(2*self.lstm_dim, self.att_dim),requires_grad=True)
			# self.W_a = Variable(torch.randn(self.att_dim,1),requires_grad=True)
			self.W_e = Parameter(torch.randn(2*self.lstm_dim, self.att_dim).uniform_(-0.01,0.01))
			self.W_a = Parameter(torch.randn(self.att_dim,1).uniform_(-0.01,0.01))
			self.W_e_type = Parameter(torch.randn(self.target_dim,2*self.lstm_dim, self.att_dim).uniform_(-0.01,0.01))
			self.W_a_type = Parameter(torch.randn(self.target_dim,self.att_dim,1).uniform_(-0.01,0.01))


		if self.feature:
			self.feature_embeddings = Embedding(self.feature_size, self.feature_dim)
		if hier:
			_d = "Wiki" if type == "figer" else "OntoNotes"
			S = create_prior("./resource/"+_d+"/label2id_"+type+".txt")
			assert(S.shape == (self.target_dim, self.target_dim))
			self.S = Variable(torch.Tensor(S))
			self.V = Parameter(torch.randn(self.target_dim,self.rep_dim).uniform_(-0.01,0.01))
		else:
			self.W = Parameter(torch.randn(self.rep_dim,self.target_dim).uniform_(-0.01,0.01))

		self.softmax = nn.Softmax()
		self.bce = nn.BCEWithLogitsLoss()

		if bags==True:
			self.A = Parameter(torch.randn(self.rep_dim).uniform_(-0.01,0.01))
			self.r = Parameter(torch.randn(self.target_dim,self.rep_dim).uniform_(-0.01,0.01))

		self.coarse_set = {}

	def attention(self,inputs):
		temp1 = [torch.tanh(torch.mm(inputs[i],self.W_e)) for i in range(len(inputs))]
		temp2 = [torch.mm(temp1[i],self.W_a) for i in range(len(inputs))]
		pre_activations = torch.cat(temp2,1)
		attentions = torch.split(tensor=self.softmax(pre_activations), split_size=1, dim=1)
		weighted_inputs = [torch.mul(inputs[i],attentions[i]) for i in range(len(inputs))]
		output = sum(weighted_inputs)
		return output, attentions

	def type_attention(self,inputs):
		outputs=[]
		attentions_ = []
		for n in range(self.target_dim):
			temp1 = [torch.tanh(torch.mm(inputs[i],self.W_e_type[n])) for i in range(len(inputs))]
			temp2 = [torch.mm(temp1[i],self.W_a_type[n]) for i in range(len(inputs))]
			pre_activations = torch.cat(temp2,1)
			attentions = torch.split(tensor=self.softmax(pre_activations), split_size=1, dim=1)
			weighted_inputs = [torch.mul(inputs[i],attentions[i]) for i in range(len(inputs))]
			output = sum(weighted_inputs)
			outputs.append(output)
			attentions_.append(attentions)

		return outputs, attentions_

	# def path_attention(self,inputs,labels):
		
	# 	rep_batch = []
	# 	for n in range(len(inputs[0])):
	# 		label = labels[n]
	# 		rep = inputs[:,n,:]
	# 		label = [i for i in range(self.target_dim) if label[i]==1 and i<4]

	# 		temp1 = [torch.tanh(torch.mm(rep[i],self.W_e)) for i in range(len(rep))]
	# 		temp2 = [torch.mm(temp1[i],self.W_a) for i in range(len(rep))]
	# 		pre_activations = torch.cat(temp2,1)
	# 		attentions = torch.split(tensor=self.softmax(pre_activations), split_size=1, dim=1)

	# 		weighted_inputs = [torch.mul(rep[i],attentions[i]) for i in range(len(rep))]
	# 		output = sum(weighted_inputs)

	# 		rep_batch.append(rep)
	# 	return output, attentions

	def forward(self,context_data,mention_representation,target,features,pro,len_of_each_bag=None,label_hierarchy=None,operation="add"):
		# if self.bags and self.bag_strategy=="one":
		# 	context_data,mention_representation,target,features = self.bag2instance(self.bag_strategy,context_data,mention_representation,target,features,scores,len_of_each_bag,label)

		batch_size = len(context_data)
		print "batch_size:"+str(batch_size)
		context=[]
		for i in range(self.context_length*2+1):
			context.append(context_data[:,i,:])
		dropout = nn.Dropout(p = pro)
		mention_representation_dropout = dropout(Variable(torch.Tensor(mention_representation)))
		left_context = Variable(torch.Tensor(context[:self.context_length]))
		right_context = Variable(torch.Tensor(list(reversed(context[self.context_length+1:]))))

		if self.encoder == "averaging":
			left_context_representation = torch.sum(left_context,0)
			right_context_representation = torch.sum(right_context,0)
			context_representation = torch.cat((left_context_representation,right_context_representation),1)
		elif self.encoder == "lstm":
			left_h0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation = torch.cat((left_output[-1],right_output[-1]),1)
		elif self.encoder == "attentive":
			left_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation, attentions = self.attention(left_output+right_output)
		elif self.encoder == "type_att":
			left_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation, attentions = self.type_attention(left_output+right_output)

		if self.feature:
			features = Variable(torch.from_numpy(features).long())
 			feature_representation = dropout(torch.sum(self.feature_embeddings(features),1))
			# representation = torch.cat([mention_representation_dropout,context_representation,feature_representation],1)
			# representation = torch.cat([context_representation,feature_representation],1)
			if self.encoder == "type_att":
				representations=[]
				# for n in range(4):
				# 	representations.append(torch.cat([mention_representation_dropout,context_representation[n],feature_representation],1))
				for n in range(self.target_dim):
					representations.append(torch.cat([mention_representation_dropout,context_representation[n],feature_representation],1))
			else:
				representation = torch.cat([mention_representation_dropout,context_representation,feature_representation],1)

		else:
			# representation = torch.cat([mention_representation_dropout,context_representation],1)
			# representation = context_representation
			if self.encoder == "type_att":
				representations=[]
				# for n in range(4):
				# 	representations.append(torch.cat([mention_representation_dropout,context_representation[n]],1))
				for n in range(self.target_dim):
					representations.append(torch.cat([mention_representation_dropout,context_representation[n]],1))
				
			else:
				representation = torch.cat([mention_representation_dropout,context_representation],1)

		if self.hier:
			self.W = torch.transpose(torch.mm(self.S,self.V),0,1)
		
		if self.bags:
			# loss = self.bag2instance(self.bag_strategy,representation,target,scores,len_of_each_bag)
			# logit = torch.mv(representation,self.W[:,label])
			# target = Variable(torch.Tensor(target))
			target_torch = Variable(torch.Tensor(target))
			if self.bag_strategy=="att":
				representation, target = self.bag_attention(representation,target_torch,len_of_each_bag,target,operation)
				logits = torch.mm(representation,self.W)
			#attention for each type	
			elif self.bag_strategy=="tatt":
				logits=[0 for i in range(self.target_dim)]
				for n in range(self.target_dim):
					rep, target = self.bag_type_attention(representation,target_torch,len_of_each_bag,target,n)
					logit = torch.mm(rep,self.W[:,n].contiguous().view(self.rep_dim,1)).contiguous().view(-1)
					logits[n] = logit
				logits = torch.stack(logits).transpose(0,1)
			elif self.bag_strategy=="patt":
				logits=[0 for i in range(self.target_dim)]
				for n in range(self.target_dim):
					rep, target = self.path_type_attention(representation,target_torch,len_of_each_bag,target,n,label_hierarchy)
					logit = torch.mm(rep,self.W[:,n].contiguous().view(self.rep_dim,1)).contiguous().view(-1)
					logits[n] = logit
				logits = torch.stack(logits).transpose(0,1)

			loss = self.bce(logits,target)
		else:
			target = Variable(torch.Tensor(target))
			if self.encoder == "type_att":
				logits=[0 for i in range(self.target_dim)]
				# for n in range(4):
				# 	for j in self.coarse_set[n]:
				# 		logit = torch.mm(representations[n],self.W[:,j].view(self.rep_dim,1)).view(-1)
				# 		logits[j] = logit
				for n in range(self.target_dim):
					logit = torch.mm(representations[n],self.W[:,n].view(self.rep_dim,1)).view(-1)
					logits[n] = logit

				logits = torch.stack(logits).transpose(0,1)
				loss = self.bce(logits,target)
			else:
				logit = torch.mm(representation,self.W)
				loss = self.bce(logit,target)
		return loss

	def predict(self,context_data,mention_representation,features,pro):
		batch_size = len(context_data)
		print "batch_size:"+str(batch_size)
		context=[]
		for i in range(self.context_length*2+1):
			context.append(context_data[:,i,:]) 
		dropout = nn.Dropout(p = pro)
		mention_representation_dropout = dropout(Variable(torch.Tensor(mention_representation)))
		left_context = Variable(torch.Tensor(context[:self.context_length]))
		right_context = Variable(torch.Tensor(list(reversed(context[self.context_length+1:]))))

		if self.encoder == "averaging":
			left_context_representation = torch.sum(left_context,0)
			right_context_representation = torch.sum(right_context,0)
			context_representation = torch.cat((left_context_representation,right_context_representation),1)
		elif self.encoder == "lstm":
			left_h0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation = torch.cat((left_output[-1],right_output[-1]),1)
		elif self.encoder == "attentive":
			left_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation, attentions = self.attention(left_output+right_output)
		elif self.encoder == "type_att":
			left_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_h0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			right_c0 = Variable(torch.randn(2*self.lstm_layers, batch_size, self.lstm_dim))
			left_output,_ = self.left_lstm(left_context,(left_h0,left_c0))
			right_output,_ = self.right_lstm(right_context,(right_h0,right_c0))
			context_representation, attentions = self.type_attention(left_output+right_output)

		if self.feature:
			features = Variable(torch.from_numpy(features).long())
 			feature_representation = dropout(torch.sum(self.feature_embeddings(features),1))
 			if self.encoder == "type_att":
				representations=[]
				# for n in range(4):
				# 	representations.append(torch.cat([mention_representation_dropout,context_representation[n],feature_representation],1))
				for n in range(self.target_dim):
					representations.append(torch.cat([mention_representation_dropout,context_representation[n],feature_representation],1))
			else:
				representation = torch.cat([mention_representation_dropout,context_representation,feature_representation],1)
		
		else:
			if self.encoder == "type_att":
				representations=[]
				# for n in range(4):
				# 	representations.append(torch.cat([mention_representation_dropout,context_representation[n]],1))
				for n in range(self.target_dim):
					representations.append(torch.cat([mention_representation_dropout,context_representation[n]],1))
			else:
				representation = torch.cat([mention_representation_dropout,context_representation],1)

		if self.hier:
			self.W = torch.transpose(torch.mm(self.S,self.V),0,1)

		if self.encoder == "type_att":
			logits=[0 for i in range(self.target_dim)]
			# for n in range(4):
			# 	for j in self.coarse_set[n]:
			# 		logit = torch.mm(representations[n],self.W[:,j].view(self.rep_dim,1)).view(-1)
			# 		logits[j] = logit
			for n in range(self.target_dim):
				logit = torch.mm(representations[n],self.W[:,n].view(self.rep_dim,1)).view(-1)
				logits[n] = logit

			logits = torch.stack(logits).transpose(0,1)
		else:
			logits = torch.mm(representation,self.W)

		scores = torch.sigmoid(logits)
		return scores.data.numpy()



	def bag_attention(self,representation, target, len_of_each_bag, nptarget, operation="mul"):
		start=0
		att_rep = []
		att_target = []
		for i in range(len(len_of_each_bag)):
			end = start+len_of_each_bag[i]
			bag_rep = representation[start:end]
			bag_target = target[start]
			labels = [j for j in range(self.target_dim) if nptarget[start][j]==1]
			if operation=="add":
				r = 0
				for l in labels:
					r = r + self.r[l]
			elif operation=="mul":
				r = 1
				for l in labels:
					r = r * self.r[l]

			alpha = F.softmax(torch.matmul(torch.mul(bag_rep,self.A),r.view(self.rep_dim,1)).view(len_of_each_bag[i])).view(1,len_of_each_bag[i])
			att_rep.append(torch.matmul(alpha,bag_rep).view(self.rep_dim))
			# att_rep.append(torch.mean(bag_rep,0))
			att_target.append(bag_target)
			start = end

		att_rep = torch.stack(att_rep)
		att_target = torch.stack(att_target)
		
		return [att_rep,att_target]

	def bag_type_attention(self,representation, target, len_of_each_bag, nptarget, label):
		start=0
		att_rep = []
		att_target = []
		for i in range(len(len_of_each_bag)):
			end = start+len_of_each_bag[i]
			bag_rep = representation[start:end]
			bag_target = target[start]
			alpha = F.softmax(torch.matmul(torch.mul(bag_rep,self.A),self.r[label].view(self.rep_dim,1)).view(len_of_each_bag[i])).view(1,len_of_each_bag[i])
			att_rep.append(torch.matmul(alpha,bag_rep).view(self.rep_dim))
			# att_rep.append(torch.mean(bag_rep,0))
			att_target.append(bag_target)
			start = end

		att_rep = torch.stack(att_rep)
		att_target = torch.stack(att_target)
		
		return [att_rep,att_target]

	def bag_type_attention(self,representation, target, len_of_each_bag, nptarget, label):
		start=0
		att_rep = []
		att_target = []
		for i in range(len(len_of_each_bag)):
			end = start+len_of_each_bag[i]
			bag_rep = representation[start:end]
			bag_target = target[start]
			if bag_target.data.numpy()[label]==1:
				alpha = F.softmax(torch.matmul(torch.mul(bag_rep,self.A),self.r[label].view(self.rep_dim,1)).view(len_of_each_bag[i])).view(1,len_of_each_bag[i])
				att_rep.append(torch.matmul(alpha,bag_rep).view(self.rep_dim))
				# att_rep.append(torch.mean(bag_rep,0))
			else:
				r = np.random.randint(0,len_of_each_bag[i])
				att_rep.append(bag_rep[r])
			att_target.append(bag_target)
			start = end

		att_rep = torch.stack(att_rep)
		att_target = torch.stack(att_target)
		
		return [att_rep,att_target]

	def path_type_attention(self,representation, target, len_of_each_bag, nptarget, label, label_hierarchy):
		start=0
		att_rep = []
		att_target = []
		ancestors = []
		label_tmp = label
		while len(label_hierarchy[label_tmp]['father'])!=0:
			label_tmp = label_hierarchy[label_tmp]['father'][0]
			ancestors.append(label_tmp)

		for i in range(len(len_of_each_bag)):
			end = start+len_of_each_bag[i]
			bag_rep = representation[start:end]
			bag_target = target[start]

			if bag_target.data.numpy()[label]==1:
				r = self.r[label]
				for l in ancestors:
					r = r + self.r[l]
				alpha = F.softmax(torch.matmul(torch.mul(bag_rep,self.A),r.view(self.rep_dim,1)).view(len_of_each_bag[i])).view(1,len_of_each_bag[i])
				att_rep.append(torch.matmul(alpha,bag_rep).view(self.rep_dim))
			else:
				r = np.random.randint(0,len_of_each_bag[i])
				att_rep.append(bag_rep[r])
			att_target.append(bag_target)
			start = end

		att_rep = torch.stack(att_rep)
		att_target = torch.stack(att_target)
		
		return [att_rep,att_target]

	# def path_type_attention(self,representation, target, len_of_each_bag, nptarget, label, label_hierarchy):
	# 	start=0
	# 	att_rep = []
	# 	att_target = []
	# 	ancestors = []
	# 	label_tmp = label
	# 	while len(label_hierarchy[label_tmp]['father'])!=0:
	# 		label_tmp = label_hierarchy[label_tmp]['father'][0]
	# 		ancestors.append(label_tmp)

	# 	for i in range(len(len_of_each_bag)):
	# 		end = start+len_of_each_bag[i]
	# 		bag_rep = representation[start:end]
	# 		bag_target = target[start]
	# 		r = self.r[label]
	# 		for l in ancestors:
	# 			r = r + self.r[l]
	# 		alpha = F.softmax(torch.matmul(torch.mul(bag_rep,self.A),r.view(self.rep_dim,1)).view(len_of_each_bag[i])).view(1,len_of_each_bag[i])
	# 		att_rep.append(torch.matmul(alpha,bag_rep).view(self.rep_dim))
	# 		att_target.append(bag_target)
	# 		start = end

	# 	att_rep = torch.stack(att_rep)
	# 	att_target = torch.stack(att_target)
		
	# 	return [att_rep,att_target]

	def coarse_type_set(self,id2label):
		for n in range(4):
			self.coarse_set[n]=[]
			for i in id2label:
				if id2label[n] in id2label[i]:
					self.coarse_set[n].append(i)
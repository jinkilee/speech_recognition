import torch
import numpy as np
from torch import nn
from torch.autograd import Variable  
import torch.nn.functional as F

def CreateOnehotVariable( input_x, encoding_dim=63):
	if type(input_x) is Variable:
		input_x = input_x.data 
	input_type = type(input_x)
	batch_size = input_x.size(0)
	time_steps = input_x.size(1)
	input_x = input_x.unsqueeze(2).type(torch.LongTensor)
	onehot_x = Variable(torch.LongTensor(
				batch_size, 
				time_steps, 
				encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
	
	return onehot_x

def TimeDistributed(input_module, input_x):
	batch_size = input_x.size(0)
	time_steps = input_x.size(1)
	reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
	output_x = input_module(reshaped_x)
	return output_x.view(batch_size,time_steps,-1)

class pBLSTMLayer(nn.Module):
	def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
		super(pBLSTMLayer, self).__init__()
		self.rnn_unit = getattr(nn,rnn_unit.upper())

		# feature dimension will be doubled since time resolution reduction
		self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
								   dropout=dropout_rate,batch_first=True)
	
	def forward(self,input_x):
		batch_size = input_x.size(0)
		timestep = input_x.size(1)
		feature_dim = input_x.size(2)
		# Reduce time resolution
		input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
		# Bidirectional RNN
		output,hidden = self.BLSTM(input_x)
		return output,hidden

class LAS(nn.Module):
	def __init__(self,
			input_size,
			listener_hidden_size,
			speller_hidden_size,
			output_class_dim,
			max_label_len,
			use_mlp_in_attention=True,
			mlp_dim_in_attention=64,
			rnn_unit='LSTM',
			use_gpu=False,
			listener_layer=3,
			speller_rnn_layer=1,
			mlp_dim_in_activation='relu',
			multi_head=1,
			decode_mode=1):

		super(LAS, self).__init__()
		self.listener = Listener(
			input_size, 
			listener_hidden_size, 
			listener_layer, 
			rnn_unit, 
			use_gpu)
		self.speller = Speller(
        output_class_dim=output_class_dim,
			speller_hidden_dim=speller_hidden_size,
			rnn_unit=rnn_unit,
			speller_rnn_layer=speller_rnn_layer,
			use_gpu=use_gpu,
			max_label_len=max_label_len,
			use_mlp_in_attention=use_mlp_in_attention,
			mlp_dim_in_attention=mlp_dim_in_attention,
			mlp_activate_in_attention=mlp_dim_in_activation,
			listener_hidden_dim=listener_hidden_size,
			multi_head=multi_head,
			decode_mode=decode_mode)

	def forward(self, x):
		h = self.listener(x)
		raw_pred_seq, attention_record = self.speller(h)
		return raw_pred_seq, attention_record



# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
	def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
		super(Listener, self).__init__()
		# Listener RNN layer
		self.listener_layer = listener_layer
		assert self.listener_layer>=1,'Listener should have at least 1 layer'
		
		self.pLSTM_layer0 = pBLSTMLayer(
			input_feature_dim,
			listener_hidden_dim, 
			rnn_unit=rnn_unit, 
			dropout_rate=dropout_rate)

		for i in range(1,self.listener_layer):
			setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

		self.use_gpu = use_gpu
		if self.use_gpu:
			self = self.cuda()

	def forward(self,input_x):
		output,_  = self.pLSTM_layer0(input_x)
		for i in range(1,self.listener_layer):
			output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
		
		return output


# Speller specified in the paper
class Speller(nn.Module):
	def __init__(self, output_class_dim,  speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
				 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, listener_hidden_dim,
				 multi_head, decode_mode, **kwargs):
		super(Speller, self).__init__()
		self.rnn_unit = getattr(nn,rnn_unit.upper())
		self.max_label_len = max_label_len
		self.decode_mode = decode_mode
		self.use_gpu = use_gpu
		self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
		self.label_dim = output_class_dim
		self.rnn_layer = self.rnn_unit(
			output_class_dim+speller_hidden_dim,
			speller_hidden_dim,
			num_layers=speller_rnn_layer,
			batch_first=True)
		self.attention = Attention(
			mlp_preprocess_input=use_mlp_in_attention,
			preprocess_mlp_dim=mlp_dim_in_attention,
			activate=mlp_activate_in_attention,
			input_feature_dim=2*listener_hidden_dim,
			multi_head=multi_head)
		self.character_distribution = nn.Linear(speller_hidden_dim*2,output_class_dim)
		self.softmax = nn.LogSoftmax(dim=-1)
		if self.use_gpu:
			self = self.cuda()

	# Stepwise operation of each sequence
	def forward_step(self,input_word, last_hidden_state,listener_feature):
		rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
		attention_score, context = self.attention(rnn_output,listener_feature)
		concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
		raw_pred = self.softmax(self.character_distribution(concat_feature))

		return raw_pred, hidden_state, context, attention_score

	def forward(self, listener_feature, ground_truth=None, teacher_force_rate = 0.9):
		if ground_truth is None:
			teacher_force_rate = 0
		teacher_force = True if np.random.random_sample() < teacher_force_rate else False

		batch_size = listener_feature.size()[0]

		output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
		if self.use_gpu:
			output_word = output_word.cuda()
		rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)

		hidden_state = None
		raw_pred_seq = []
		output_seq = []
		attention_record = []

		if (ground_truth is None) or (not teacher_force):
			max_step = self.max_label_len
		else:
			max_step = ground_truth.size()[1]

		for step in range(max_step):
			raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
			raw_pred_seq.append(raw_pred)
			attention_record.append(attention_score)
			# Teacher force - use ground truth as next step's input
			if teacher_force:
				output_word = ground_truth[:,step:step+1,:].type(self.float_type)
			else:
				# Case 0. raw output as input
				if self.decode_mode == 0:
					output_word = raw_pred.unsqueeze(1)
				# Case 1. Pick character with max probability
				elif self.decode_mode == 1:
					output_word = torch.zeros_like(raw_pred)
					for idx,i in enumerate(raw_pred.topk(1)[1]):
						output_word[idx,int(i)] = 1
					output_word = output_word.unsqueeze(1)			 
				# Case 2. Sample categotical label from raw prediction
				else:
					sampled_word = Categorical(raw_pred).sample()
					output_word = torch.zeros_like(raw_pred)
					for idx,i in enumerate(sampled_word):
						output_word[idx,int(i)] = 1
					output_word = output_word.unsqueeze(1)
				
			rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)

		return raw_pred_seq, attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state					  with shape [batch size, 1, decoder hidden dimension]
#		 Compressed feature from Listner	with shape [batch size, T, listener feature dimension]
# Output: Attention score					with shape [batch size, T (attention score of each time step)]
#		 Context vector					 with shape [batch size,  listener feature dimension]
#		 (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):  

	def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, mode='dot', input_feature_dim=512,
				multi_head=1):
		super(Attention,self).__init__()
		self.mode = mode.lower()
		self.mlp_preprocess_input = mlp_preprocess_input
		self.multi_head = multi_head
		self.softmax = nn.Softmax(dim=-1)
		if mlp_preprocess_input:
			self.preprocess_mlp_dim  = preprocess_mlp_dim
			self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim*multi_head)
			self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
			if self.multi_head > 1:
				self.dim_reduce = nn.Linear(input_feature_dim*multi_head,input_feature_dim)
			if activate != 'None':
				self.activate = getattr(F, activate)
			else:
				self.activate = None

	def forward(self, decoder_state, listener_feature):
		if self.mlp_preprocess_input:
			if self.activate:
				comp_decoder_state = self.activate(self.phi(decoder_state))
				comp_listener_feature = self.activate(TimeDistributed(self.psi,listener_feature))
			else:
				comp_decoder_state = self.phi(decoder_state)
				comp_listener_feature = TimeDistributed(self.psi,listener_feature)
		else:
			comp_decoder_state = decoder_state
			comp_listener_feature = listener_feature

		if self.mode == 'dot':
			if self.multi_head == 1:
				energy = torch.bmm(comp_decoder_state,comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
				attention_score = [self.softmax(energy)]
				context = torch.sum(listener_feature*attention_score[0].unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)
			else:
				attention_score =  [ self.softmax(torch.bmm(att_querry,comp_listener_feature.transpose(1, 2)).squeeze(dim=1))\
									for att_querry in torch.split(comp_decoder_state, self.preprocess_mlp_dim, dim=-1)]
				projected_src = [torch.sum(listener_feature*att_s.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1) \
								for att_s in attention_score]
				context = self.dim_reduce(torch.cat(projected_src,dim=-1))
		else:
			# TODO: other attention implementations
			pass
		
		

		return attention_score,context

# partly adapted from the https://github.com/brightmart/text_classification/tree/master/a03_TextRNN

# last updated: 25 March 2019

# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class BiGRU:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,lambda_sim=0.00001,lambda_sub=0,initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,multi_label_flag=True): #initializer=tf.random_normal_initializer(stddev=0.1)
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_sentences = 1
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=embed_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5) # using assign to half the learning_rate
        self.initializer=initializer
        self.multi_label_flag = multi_label_flag
        self.clip_gradients=clip_gradients
        self.lambda_sim=lambda_sim
        self.lambda_sub=lambda_sub
        
        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y") # for single label # y [None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        
        self.label_sim_matrix = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sim_mat")
        self.label_sub_matrix = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sub_mat")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            if self.lambda_sim == 0:
                if self.lambda_sub == 0:
                    # none
                    self.loss_val = self.loss_multilabel() # without any semantic regularisers, no L_sim or L_sub
                else:
                    # using L_sub only
                    self.loss_val = self.loss_multilabel_onto_new_sub_per_batch(self.label_sub_matrix); # j,k per batch - used in the NAACL paper
                    #self.loss_val = self.loss_multilabel_onto_new_sub_per_doc(self.label_sub_matrix); # j,k per document
            else:
                if self.lambda_sub == 0:
                    # using L_sim only
                    #pair_diff_squared on s_d
                    self.loss_val = self.loss_multilabel_onto_new_sim_per_batch(self.label_sim_matrix) # j,k per batch - used in the NAACL paper
                    #self.loss_val = self.loss_multilabel_onto_new_sim_per_doc(self.label_sim_matrix) # j,k per document
                    
                else:
                    # L_sim+L_sub
                    self.loss_val = self.loss_multilabel_onto_new_simsub_per_batch(self.label_sim_matrix,self.label_sub_matrix) # j,k per batch - used in the NAACL paper
                    #self.loss_val = self.loss_multilabel_onto_new_simsub_per_doc(self.label_sim_matrix,self.label_sub_matrix) # j,k per document
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        
        # output evaluation results on training data
        sig_output = tf.sigmoid(self.logits)
        if not self.multi_label_flag:
            self.predictions = tf.argmax(sig_output, axis=1, name="predictions")  # shape:[None,]
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                          self.input_y)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
            self.precision = 0
            self.recall = 0
            #self.f_measure = 0
        else:
            self.predictions = tf.round(sig_output) #y = sign(x) = -1 if x < 0; 0 if x == 0 or tf.is_nan(x); 1 if x > 0.
            #self.predictions = tf.cast(tf.greater(self.sig_logits,0.25),tf.float32)
            temp = tf.cast(tf.equal(self.predictions,self.input_y_multilabel), tf.float32)
            print('temp',temp)
            tp = tf.reduce_sum(tf.multiply(temp,self.predictions), axis=1)  # [128,1]
            p = tf.reduce_sum(self.predictions, axis=1) + 1e-10 # [128,1]
            t = tf.reduce_sum(self.input_y_multilabel, axis=1) # [128,1]
            union = tf.reduce_sum(tf.cast(tf.greater(self.predictions + self.input_y_multilabel,0),tf.float32), axis=1) # [128,1]
            self.accuracy = tf.reduce_mean(tf.div(tp,union))
            self.precision = tf.reduce_mean(tf.div(tp,p))
            self.recall = tf.reduce_mean(tf.div(tp,t))
        
        self.training_loss = tf.summary.scalar("train_loss_per_batch",self.loss_val)
        self.training_loss_per_epoch = tf.summary.scalar("train_loss_per_epoch",self.loss_val)
        self.validation_loss = tf.summary.scalar("validation_loss_per_batch",self.loss_val)
        self.validation_loss_per_epoch = tf.summary.scalar("validation_loss_per_epoch",self.loss_val)
        self.writer = tf.summary.FileWriter("./logs")
        
    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]
            
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])
    
    #this is the original lstm implementation in https://github.com/brightmart/text_classification/blob/master/a03_TextRNN/p8_TextRNN_model.py
    def inference_lstm(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        #2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32) #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>",outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        #3. concat output
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        #self.output_rnn_last=tf.reduce_mean(output_rnn,axis=1) #[batch_size,hidden_size*2] # this is average pooling
        self.output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] # this uses the last hidden state as the representation.
        print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        #4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits
    
    # using gru instead of lstm
    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) 
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])
        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
                             #self.hidden_state is a list.
        print('self.hidden_state', len(self.hidden_state), self.hidden_state[0].get_shape())                      
        self.output_rnn_last = self.hidden_state[-1] # using last hidden state
        #self.output_rnn_last = self.hidden_state[0] # using first hidden state
        print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        #4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits
        
    # loss for single-label classification    
    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss
    
    # loss for multi-label classification
    def loss_multilabel(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda #12 loss
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses
        return loss
    
    # L_sim only: j,k per batch
    def loss_multilabel_onto_new_sim_per_batch(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d. 
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True) # input_y_multilabel is a matrix \in R^{|D|,|T|}
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sim_matrix = tf.multiply(co_label_mat_batch,label_sim_matrix) # only considering the label similarity of labels in the label set for this document (here is a batch).
            
            # sim-loss after sigmoid L_sim = sim(T_j,T_k)|s_dj-s_dk|^2
            sig_output = tf.sigmoid(self.logits) # self.logit is the matrix S \in R^{|D|,|T|}
            vec_square = tf.multiply(sig_output,sig_output) # element-wise multiplication 
            vec_square = tf.reduce_sum(vec_square,0) # an array of num_classes values {sum_d l_dj^2}_j
            vec_mid = tf.matmul(tf.transpose(sig_output),sig_output)
            vec_rows=tf.ones([tf.size(vec_square),1])*vec_square # copy the vector by it self to shape a square
            vec_columns=tf.transpose(vec_rows)
            vec_diff=vec_rows-2*vec_mid+vec_columns # (li-lj)^2=li^2-2lilj+lj^2 # vec_diff is now a matrix = {sum_d (l_di-l_dj)^2}_i,j
            vec_diff=tf.multiply(vec_diff,label_sim_matrix) #sim(T_i,T_j)*(li-lj)^2 # element-wise # using the label_sim_matrix
            #vec_diff=tf.multiply(vec_diff,co_label_mat_batch) # using only tag co-occurrence 
            vec_final=tf.reduce_sum(vec_diff)/2 # vec_diff is symmetric
            #vec_final=tf.reduce_sum(vec_diff)/2/self.num_classes/self.num_classes # vec_diff is symmetric
            self.sim_loss=(vec_final/self.batch_size)*self.lambda_sim
            
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
    
	# L_sim only: j,k per document
    def loss_multilabel_onto_new_sim_per_doc(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            sig_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sim_loss = 0
            for i in range(len(sig_list)): # loop over d
                logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |s_dj-s_dk|^2
                pair_diff_squared_d = tf.square(tf.transpose(s_d_true) - s_d_true)
                #gather the Sim_jk from Sim
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                #ind_flat_lower = tf.tile(label_index,[label_len])
                #ind_mat = tf.reshape(ind_flat_lower,[label_len,label_len])
                #ind_flat_upper = tf.reshape(tf.transpose(ind_mat),[-1])
                #ind_squ = tf.transpose(tf.stack([ind_flat_upper,ind_flat_lower]))
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_squared_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
		
    # L_sim and L_sub - per doc
    # label_sub_matrix: sub(T_j,T_k) \in {0,1} means whether T_j is a hyponym of T_k.
    def loss_multilabel_onto_new_simsub_per_doc(self, label_sim_matrix, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            sig_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sim_loss = 0
            self.sub_loss = 0
            for i in range(len(sig_list)): # loop over d
                logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |s_dj-s_dk|^2
                pair_diff_squared_d = tf.square(tf.transpose(s_d_true) - s_d_true)
                #calculate R(s_dj)(1-R(s_dk))
                pred_d_true = tf.round(s_d_true)
                pair_sub_d = tf.matmul(tf.transpose(pred_d_true),1-pred_d_true)
                
                #gather the Sim_jk from Sim and the Sub_jk from Sub
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                label_sub_matrix_d = tf.reshape(tf.gather_nd(label_sub_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_squared_d))
                self.sub_loss = self.sub_loss + tf.reduce_sum(tf.multiply(label_sub_matrix_d,pair_sub_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss=(self.sub_loss/self.batch_size)*self.lambda_sub/2.0
            
            loss = self.loss_ce + self.l2_losses + self.sim_loss + self.sub_loss
        return loss
        
    # L_sim and L_sub - per batch, used in the NAACL paper
    # label_sub_matrix: sub(T_j,T_k) \in {0,1} means whether T_j is a hypernym of T_k.
    def loss_multilabel_onto_new_simsub_per_batch(self, label_sim_matrix, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True)
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sim_matrix = tf.multiply(co_label_mat_batch,label_sim_matrix) # only considering the label similarity of labels in the label set for this document (batch of documents).
            label_sub_matrix = tf.multiply(co_label_mat_batch,label_sub_matrix)
            
            # the sim-loss after sigmoid
            sig_output = tf.sigmoid(self.logits)
            vec_square = tf.multiply(sig_output,sig_output)
            vec_square = tf.reduce_sum(vec_square,0) # an array of num_classes values {sum_d l_di}_i
            vec_mid = tf.matmul(tf.transpose(sig_output),sig_output)
            vec_rows=tf.ones([tf.size(vec_square),1])*vec_square
            vec_columns=tf.transpose(vec_rows)
            vec_diff=vec_rows-2*vec_mid+vec_columns # (li-lj)^2=li^2-2lilj+lj^2 # vec_diff is now a matrix = {sum_d (l_di-l_dj)^2}_i,j
            vec_diff=tf.multiply(vec_diff,label_sim_matrix) #sim(T_i,T_j)*(li-lj)^2 # element-wise # using the label_sim_matrix
            #vec_diff=tf.multiply(vec_diff,co_label_mat_batch) # using only tag co-occurrence 
            vec_final=tf.reduce_sum(vec_diff)/2 # vec_diff is symmetric
            #vec_final=tf.reduce_sum(vec_diff)/2/self.num_classes/self.num_classes # vec_diff is symmetric
            self.sim_loss=(vec_final/self.batch_size)*self.lambda_sim
            
            # the sub-loss after sigmoid 
            pred = tf.round(sig_output)
            pred_mat = tf.matmul(tf.transpose(pred),1-pred)
            sub_loss = tf.multiply(pred_mat,label_sub_matrix)
            self.sub_loss = self.lambda_sub * tf.reduce_sum(sub_loss) / 2. / self.batch_size
            
            loss = self.loss_ce + self.l2_losses + self.sim_loss + self.sub_loss
        return loss
    
    # L_sub only - per batch - used in the NAACL paper
    def loss_multilabel_onto_new_sub_per_batch(self, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            ## sub_loss: matrix multiplication: only using the label relations in the label set, treating same in each batch.
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True)
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sub_matrix = tf.multiply(co_label_mat_batch,label_sub_matrix)
            
            # the sub-loss after sigmoid
            sig_output = tf.sigmoid(self.logits)
            pred = tf.round(sig_output)
            pred_mat = tf.matmul(tf.transpose(pred),1-pred)
            sub_loss = tf.multiply(pred_mat,label_sub_matrix)
            self.sub_loss = self.lambda_sub * tf.reduce_sum(sub_loss) / 2. / self.batch_size
            
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sub_loss
        return loss
    
    # L_sub only - per document
    def loss_multilabel_onto_new_sub_per_doc(self, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            ## sub_loss: matrix multiplication: only using the label relations in the label set, treating same in each batch.
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            sig_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sub_loss = 0
            for i in range(len(sig_list)): # loop over d
                logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate R(s_dj)(1-R(s_dk))
                pred_d_true = tf.round(s_d_true)
                pair_sub_d = tf.matmul(tf.transpose(pred_d_true),1-pred_d_true)
                #gather the Sub_jk from Sub
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sub_matrix_d = tf.reshape(tf.gather_nd(label_sub_matrix,ind_squ),[label_len,label_len])
                
                self.sub_loss = self.sub_loss + tf.reduce_sum(tf.multiply(label_sub_matrix_d,pair_sub_d))
            
            self.sub_loss=(self.sub_loss/self.batch_size)*self.lambda_sub/2.0
            
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sub_loss
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True) #exponential_decay
        #train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients) #using adam here. # gradient cliping is also applied.
        return train_op
        
    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,
                                                                self.U_z) + self.b_z)  # z_t:[batch_size*num_sentences,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,
                                                                self.U_r) + self.b_r)  # r_t:[batch_size*num_sentences,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t
    
    # forward gru for first level: word levels
    def gru_forward_word_level(self, embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
                                           # Now the sequence_length is the sentence_length
        #print('after splitting in gru', len(embedded_words_splitted), embedded_words_splitted[0].get_shape())                                   
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=embedded_words_squeeze[0].get_shape().dims[0]
        h_t = tf.ones((self.batch_size * self.num_sentences,
                       self.hidden_size))  #TODO self.hidden_size h_t =int(tf.get_shape(embedded_words_squeeze[0])[0]) # tf.ones([self.batch_size*self.num_sentences, self.hidden_size]) # [batch_size*num_sentences,embed_size]
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size*num_sentences,embed_size]
            h_t = self.gru_single_step_word_level(Xt,h_t)  # [batch_size*num_sentences,embed_size]<------Xt:[batch_size*num_sentences,embed_size];h_t:[batch_size*num_sentences,embed_size]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: backward hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze.reverse()  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=int(tf.get_shape(embedded_words_squeeze[0])[0]) #h_t = tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse()
        return h_t_backward_list
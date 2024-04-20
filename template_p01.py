#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def softmax(vector):
    '''
    vector: np.array of shape (n, m)

    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


# In[3]:


def multiplicative_attention_score(decoder_hidden_state, weights_matrix, encoder_hidden_states):
  attention_scores = np.dot(decoder_hidden_state.T, weights_matrix).dot(encoder_hidden_states)
  return attention_scores


# In[4]:


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    attention_scores = multiplicative_attention_score(decoder_hidden_state=decoder_hidden_state, weights_matrix=W_mult, encoder_hidden_states=encoder_hidden_states)
    attention_vector = softmax(attention_scores).dot(encoder_hidden_states.T).T

    return attention_vector


# In[5]:


def additive_attention_score(encoder_hidden_states, W_add_enc, W_add_dec, decoder_hidden_state):
  attentions_scores = np.tanh(np.dot(W_add_enc, encoder_hidden_states) + np.dot(W_add_dec, decoder_hidden_state))
  return attentions_scores


# In[6]:


def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    attentions_scores = np.dot(v_add.T, additive_attention_score(encoder_hidden_states, W_add_enc, W_add_dec, decoder_hidden_state))
    attention_vector = softmax(attentions_scores).dot(encoder_hidden_states.T).T
    return attention_vector


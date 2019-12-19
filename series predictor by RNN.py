# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:51:04 2019

@author: Asus
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
 
   
def load_series(filename, series_idx=1):
        try:
            with open(timeseries) as csvfile:
                csvreader = csv.reader(csvfile)
            
                data = [float(row[series_idx]) for row in csvreader
                                            if len(row) > 0]
                normalized_data = (data - np.mean(data)) / np.std(data) 
            return normalized_data
        except IOError:
            return None


def split_data(data, percent_train=0.80):
        num_rows = len(data) * percent_train
        return data[:num_rows], data[num_rows:] 
    
    
def test(self, sess, test_x):
    tf.get_variable_scope().reuse_variables()
    self.saver.restore(sess, './model.ckpt')
    output = sess.run(self.model(), feed_dict={self.x: test_x})
    return output

    if __name__ == '__main__':
      seq_size = 5
      predictor = SeriesPredictor(
            input_dim=1, #A
            seq_size=seq_size, #B
            hidden_dim=100) #C
#D
    data = data_loader.load_series('international-airline-passengers.csv')
    train_data, actual_vals = data_loader.split_data(data)
    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1): #E
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])
    test_x, test_y = [], [] #F
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])
    predictor.train(train_x, train_y, test_x, test_y) #G
#H
    with tf.Session() as sess:
        predicted_vals = predictor.test(sess, test_x)[:,0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, 'predictions.png')
        prev_seq = train_x[-1]
        predicted_vals = []
    for i in range(20):
        next_seq = predictor.test(sess, [prev_seq])
        predicted_vals.append(next_seq[-1])
        prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        
        plot_results(train_data, predicted_vals, actual_vals, 'hallucinations.png')
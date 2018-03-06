from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from data import *
import pp

assert len(argv) == 2
problem_label = argv[1]

if problem_label == "3.1.1":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	x = [10 * (i + 1) for i in range(6)]
	n_exp = 3
	for dim in [50, 75, 100, 150]:
		print "[Info] # Hidden Node =", dim
		model = BernoulliRBM(n_components = dim)
		model.set_params(learning_rate = 0.02, random_state = 1)
		RBM_error = []
		AE_error = []
		
		for n_iter in range(6):
			train_n_iter = (n_iter + 1) * 10
			model.set_params(n_iter = train_n_iter, verbose = False)
			print "Training RBM with n_iter =", train_n_iter, "and n_hidden =", dim, "..."
			model.fit(train_data)
			error_count = 0.
			for i in range(n_exp):
				error_count += np.sum(np.sum(np.abs(model.gibbs(test_data) - test_data))) / 784. / 2000. / n_exp 
			RBM_error.append(error_count)
			print "RBM error with n_iter =", train_n_iter, ":", error_count
		print "RMB Error with dim =", dim, ":", RBM_error
		plt.plot(x, RBM_error, label = "RMB {}".format(dim))
		
		print "Training AE with n_iter =", 10, "and n_hidden =", dim, "..."
		model = MLPClassifier(hidden_layer_sizes = (dim, ), random_state = 1, learning_rate_init = 0.003, verbose = False, max_iter = 10, warm_start = True)
		model.fit(train_data, train_data)
		AE_error.append(np.sum(np.sum(np.abs(model.predict(test_data) - test_data))) / 784. / 2000.)
		print "AE error with n_iter =", 10, ":", AE_error[-1]
		
		
		for i in range(5):
			print "Training AE with n_iter =", (i + 2) * 10, "and n_hidden =", dim, "..."
			for j in range(10):
				model.fit(train_data, train_data)
			AE_error.append(np.sum(np.sum(np.abs(model.predict(test_data) - test_data))) / 784. / 2000.)
			print "AE error with n_iter =", (i + 2) * 10, ":", AE_error[-1]
		print "AE Error with dim =", dim, ":", AE_error
		plt.plot(x, AE_error, label = "AE {}".format(dim))
		# image.show_image(model.predict(train_data[0:1]))
		
		
	plt.legend()
	plt.show()

if problem_label == "3.1.2":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	for dim in [50, 100]:
		# RBM
		print "Processing RBM with dim = {}".format(dim)
		model = BernoulliRBM(n_components = dim, learning_rate = 0.02, random_state = 1, n_iter = 60, verbose = False)
		model.fit(train_data)
		for i in range(dim):
			plt.imshow(np.reshape(model.components_[i], (28, 28)).tolist())
			plt.savefig("results/3.1.2/RBM/{}/{}.png".format(dim, i + 1))
		print model.components_.shape
		
		# AE
		print "Processing AE with dim = {}".format(dim)
		model = MLPClassifier(hidden_layer_sizes = (dim, ), random_state = 1, learning_rate_init = 0.003, verbose = False, max_iter = 60)
		model.fit(train_data, train_data)
		print len(model.coefs_)
		print model.coefs_[1].shape
		for i in range(dim):
			plt.imshow(np.reshape(model.coefs_[1][i], (28, 28)).tolist())
			plt.savefig("results/3.1.2/AE/{}/{}.png".format(dim, i + 1))		
		
	
	
	pass


from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model, metrics
from sklearn.utils import extmath
from sklearn.neural_network._base import *
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from data import *

assert len(argv) == 2
problem_label = argv[1]

if problem_label == "3.1.1":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	x = [10 * (i + 1) for i in range(6)]
	n_exp = 3
	for dim in [50, 75, 100, 150]:
		print("[Info] # Hidden Node =", dim)
		model = BernoulliRBM(n_components = dim)
		model.set_params(learning_rate = 0.02, random_state = 1)
		RBM_error = []
		AE_error = []
		
		for n_iter in range(6):
			train_n_iter = (n_iter + 1) * 10
			model.set_params(n_iter = train_n_iter, verbose = False)
			print("Training RBM with n_iter =", train_n_iter, "and n_hidden =", dim, "...")
			model.fit(train_data)
			error_count = 0.
			for i in range(n_exp):
				error_count += np.sum(np.sum(np.abs(model.gibbs(test_data) - test_data))) / 784. / 2000. / n_exp 
			RBM_error.append(error_count)
			print("RBM error with n_iter =", train_n_iter, ":", error_count)
		print("RMB Error with dim =", dim, ":", RBM_error)
		plt.plot(x, RBM_error, label = "RMB {}".format(dim))
		
		print("Training AE with n_iter =", 10, "and n_hidden =", dim, "...")
		model = MLPClassifier(hidden_layer_sizes = (dim, ), random_state = 1, learning_rate_init = 0.003, verbose = False, max_iter = 10, warm_start = True)
		model.fit(train_data, train_data)
		AE_error.append(np.sum(np.sum(np.abs(model.predict(test_data) - test_data))) / 784. / 2000.)
		print("AE error with n_iter =", 10, ":", AE_error[-1])
		
		
		for i in range(5):
			print("Training AE with n_iter =", (i + 2) * 10, "and n_hidden =", dim, "...")
			for j in range(10):
				model.fit(train_data, train_data)
			AE_error.append(np.sum(np.sum(np.abs(model.predict(test_data) - test_data))) / 784. / 2000.)
			print("AE error with n_iter =", (i + 2) * 10, ":", AE_error[-1])
		print("AE Error with dim =", dim, ":", AE_error)
		plt.plot(x, AE_error, label = "AE {}".format(dim))
		# image.show_image(model.predict(train_data[0:1]))
		
		
	plt.legend()
	plt.show()

if problem_label == "3.1.2":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	for dim in [50, 100]:
		# RBM
		print("Processing RBM with dim = {}".format(dim))
		model = BernoulliRBM(n_components = dim, learning_rate = 0.005, random_state = 1, n_iter = 200, verbose = True)
		model.fit(train_data)
		for i in range(dim):
			plt.imshow(np.reshape(model.components_[i], (28, 28)).tolist())
			plt.savefig("results/3.1.2/RBM/{}/{}.png".format(dim, i + 1))
		print(model.components_.shape)

		# AE
		print("Processing AE with dim = {}".format(dim))
		model = MLPClassifier(hidden_layer_sizes = (dim, ), random_state = 1, learning_rate_init = 0.003, verbose = False, max_iter = 60)
		model.fit(train_data, train_data)
		print(len(model.coefs_))
		print(model.coefs_[1].shape)
		for i in range(dim):
			plt.imshow(np.reshape(model.coefs_[1][i], (28, 28)).tolist())
			plt.savefig("results/3.1.2/AE/{}/{}.png".format(dim, i + 1))		

if problem_label == "3.2.1.1":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	dim = [150, 100, 50]
	midVal = train_data.copy()
	# layerOutput = []
	models = []
	for layer in range(len(dim)):
		model = BernoulliRBM(n_components=dim[layer], learning_rate=0.005, random_state=1, n_iter=100, verbose=True)
		model.fit(midVal)
		models.append(model)
		midVal = model.transform(midVal)
		# layerOutput.append(midVal.copy)

	# logistic_classifier = []
	classifier = linear_model.LogisticRegression(C=100.0)
	classifier.fit(train_data, train_label)
	print("Layer", 0, ":\n",
		  metrics.classification_report(test_label, classifier.predict(test_data)))

	midVal = test_data.copy()

	# images = [[],[],[],[]]

	for layer in range(len(dim)):
		classifier = linear_model.LogisticRegression(C=100.0)
		midVal = models[layer].transform(midVal)
		count = [0 for i in range(5)]
		for ii in range(len(test_data)):
			for jj in [0, 1, 2, 3, 4]:
				if (test_label[ii]==jj and count[jj]<10):
					image = midVal[ii].reshape((10, dim[layer]//10))
					plt.imshow(image)
					plt.savefig("results/3.2.1.1/RBM_layer{}_{}_{}_{}".format(layer, dim[layer], jj, count[jj]))
					count[jj] += 1
		classifier.fit(midVal, test_label)
		print("Layer", layer+1, "-", dim[layer], ":\n",
			  metrics.classification_report(test_label, classifier.predict(midVal)))

if problem_label == "3.2.1.2":
	train_data, train_label = data.get_train_data()
	test_data, test_label = data.get_test_data()
	dim = [150, 100, 50]
	midVal = train_data.copy()
	# layerOutput = []
	models = []
	for layer in range(len(dim)):
		model = MLPRegressor(hidden_layer_sizes=(dim[layer],), random_state = 1, learning_rate_init = 0.003, verbose = True, max_iter = 100)
		model.fit(midVal, midVal)
		models.append(model)
		hidden_activation = ACTIVATIONS[model.activation]
		midVal = hidden_activation(extmath.safe_sparse_dot(midVal, model.coefs_[0])+model.intercepts_[0]).copy()

	# logistic_classifier = []
	classifier = linear_model.LogisticRegression(C=100.0)
	classifier.fit(train_data, train_label)
	print("Layer", 0, ":\n",
		  metrics.classification_report(test_label, classifier.predict(test_data)))

	midVal = test_data.copy()
	for layer in range(len(dim)):
		classifier = linear_model.LogisticRegression(C=100.0)
		hidden_activation = ACTIVATIONS[models[layer].activation]
		midVal = hidden_activation(extmath.safe_sparse_dot(midVal, models[layer].coefs_[0])+models[layer].intercepts_[0]).copy()
		count = [0 for i in range(5)]
		for ii in range(len(test_data)):
			for jj in [0, 1, 2, 3, 4]:
				if (test_label[ii] == jj and count[jj] < 10):
					image = midVal[ii].reshape((10, dim[layer] // 10))
					plt.imshow(image)
					plt.savefig("results/3.2.1.2/AE_layer{}_{}_{}_{}".format(layer, dim[layer], jj, count[jj]))
					count[jj] += 1
		classifier.fit(midVal, test_label)
		print("Layer", layer + 1, "-", dim[layer], ":\n",
			  metrics.classification_report(test_label, classifier.predict(midVal)))



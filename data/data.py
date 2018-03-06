import image

def get_train_data():
	train_data = []
	train_label = []
	index = 0
	with open("binMNIST_data/bindigit_trn.csv") as f:
		lines = f.readlines()
		for line in lines:
			train_data.append([])
			info = line.split(",")
			for i in range(784):
				train_data[-1].append(int(info[i]))
	with open("binMNIST_data/targetdigit_trn.csv") as f:
		lines = f.readlines()
		for line in lines:
			train_label.append(int(line))
	return train_data, train_label

def get_test_data():
	test_data = []
	test_label = []
	index = 0
	with open("binMNIST_data/bindigit_tst.csv") as f:
		lines = f.readlines()
		for line in lines:
			test_data.append([])
			info = line.split(",")
			for i in range(784):
				test_data[-1].append(int(info[i]))
	with open("binMNIST_data/targetdigit_tst.csv") as f:
		lines = f.readlines()
		for line in lines:
			test_label.append(int(line))
	return test_data, test_label

if __name__ == "__main__":
	x, y = get_train_data()
	x1, y1 = get_test_data()
	
	image.show_image(x[0])
	print y[0]

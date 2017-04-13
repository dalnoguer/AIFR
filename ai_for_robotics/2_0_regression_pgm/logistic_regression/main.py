import numpy as np
import matplotlib.pyplot as plt
import copy

def logistic_function(w, x):

	denominator = 1 + np.exp(np.dot(-w,x.transpose()))
	return 1/denominator

def compute_gradient(x,y,w,alpha):

	logistic = logistic_function(w,x)
	h_y = logistic - y.reshape(1,y.shape[0])
	gradient = np.dot(h_y,x)
	w_tmp = copy.deepcopy(w)
	w_tmp[-1] = 0
	gradient = gradient + alpha*w_tmp

	return gradient

def compute_hessian(x,y,w,alpha):

	diag_h = np.diag(logistic_function(w,x))
	diag_1_h = np.diag(logistic_function(w,-x))
	diagonal = np.dot(diag_h,diag_1_h)
	hessian = np.dot(np.dot(x.transpose(),diagonal),x)
	identity = np.identity(w.shape[1])
	identity[-1,-1] = 0
	hessian = hessian + alpha*identity

	return hessian


input_data = np.genfromtxt(open("XtrainIMG.txt"))  #features: columns, samples: rows
output_data = np.genfromtxt(open("Ytrain.txt"))  # Classification (1 for open eye 0 for closed eye) in the rows

n_samples = input_data.shape[0]
n_features = input_data.shape[1]


ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch][:,None]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:][:,None]

x_0 = np.ones(training_input.shape[0])
training_input = np.column_stack((training_input,x_0))
x_0 = np.ones(validation_input.shape[0])
validation_input = np.column_stack((validation_input,x_0))

w = np.zeros([1,n_features+1])
iteration = 0
iteration_max = 10000
alpha = 10
while iteration<iteration_max:

	gradient = compute_gradient(training_input,training_output,w,alpha)
	hessian = compute_hessian(training_input,training_output,w,alpha)
	w = w - np.dot(np.linalg.inv(hessian),gradient.transpose()[:,0])
	iteration += 1
	print iteration

#validation
h = logistic_function(w,validation_input)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()

print 'wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set'


# classify test data for evaluation
test_input = np.genfromtxt(open("XtestIMG.txt"))
x_0 = np.ones(test_input.shape[0])
test_input = np.column_stack((test_input,x_0))

h = logistic_function(w,test_input)
test_output = np.round(h).transpose()
np.savetxt('results.txt', test_output)

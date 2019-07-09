import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("dataset.csv", index_col = 0)
x = np.array(dataset['x'])
y = np.array(dataset['y'])

def cost(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


def gradient_descent(learning_rate, x, y, iterations):
	#initialize theta
	t0 = 0
	t1 = 0

	#number of examples
	m = x.shape[0]
	decay_rate = 0.99
	#total error
	J = cost(m, t0, t1, x, y)
	cache0, cache1, eps = 0,0,0.000001
	loss = np.empty(iterations)
	count = [i for i in range(1, iterations+1)]

	for it in range(iterations):
		#Calculating the gradients
		grad0 = 1/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
		grad1 = 1/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
		#Updating the parameters
		cache0 = decay_rate * cache0 + (1 - decay_rate) * grad0**2
		cache1 = decay_rate * cache1 + (1 - decay_rate) * grad1**2
		t0 = t0 - (learning_rate * grad0)/(np.sqrt(cache0 + eps))
		t1 = t1 - (learning_rate * grad1)/(np.sqrt(cache1 + eps))
		loss[it] = cost(m, t0, t1, x, y)
	
	return count, loss, t0, t1


#Implementation

alpha = 0.01
max_iter = 10000

count, loss, theta0, theta1 = gradient_descent(alpha, x, y, max_iter)

print('theta0 = ' + str(theta0))
print('theta1 = ' + str(theta1))

plt.figure(0)
plt.scatter(x, y, c = 'red')
plt.plot(x, theta0 + theta1 * x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()

plt.figure(1)
plt.plot(count, loss)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()

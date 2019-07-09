import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("data.csv", index_col = 0)
x = np.array(dataset['x'])
y = np.array(dataset['y'])

def costFunction(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


def gd(lr, x, y, iterations):
	#initialize theta
	theeta0 = 0
	theeta1 = 0

	#number of examples
	m = x.shape[0]
	decay_rate = 0.99
	#total error
	J = costFunction(m, theeta0, theeta1, x, y)
	cache0, cache1, eps = 0,0,0.000001
	m0,m1,v0,v1 = 0,0,0,0 
	beta1 = 0.99
	beta2 = 0.9999
	loss = np.empty(iterations)
	count = [i for i in range(1, iterations+1)]

	for it in range(1, iterations+1):
		grad0 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
		grad1 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
		
		m0 = beta1*m0 + (1-beta1)*grad0
		mtheeta0 = m0 / (1-beta1**it)
		v0 = beta2*v0 + (1-beta2)*(grad0**2)
		vtheeta0 = v0 / (1-beta2**it)
		
		m1 = beta1*m1 + (1-beta1)*grad1
		mtheeta1 = m1 / (1-beta1**it)
		v1 = beta2*v1 + (1-beta2)*(grad1**2)
		vtheeta1 = v1 / (1-beta2**it)

		theeta0 = theeta0 - (lr * mtheeta0)/(np.sqrt(vtheeta0 + eps))
		theeta1 = theeta1 - (lr * mtheeta1)/(np.sqrt(vtheeta1 + eps))
		loss[it-1] = costFunction(m, theeta0, theeta1, x, y)

	return count, loss, theeta0, theeta1

max_iter = 10000
alpha = 1
count, loss, theta0, theta1 = gd(alpha, x, y, max_iter)

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

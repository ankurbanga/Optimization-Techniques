import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("dataset.csv", index_col = 0)
x = np.array(dataset['x'])
y = np.array(dataset['y'])

def cost(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


def gradient_descent(lr, x, y, iterations):
	#initialize theta
	t0 = 0
	t1 = 0

	#number of examples
	m = x.shape[0]
	decay_rate = 0.99
	#total error
	J = cost(m, t0, t1, x, y)
	cache0, cache1, eps = 0,0,0.00000001
	m0,mt0,m1,mt1,v0,vt0,v1,vt1 = 0,0,0,0,0,0,0,0 
	beta1 = 0.9
	beta2 = 0.999

	loss = np.empty(iterations)
	count = [i for i in range(1, iterations+1)]

	for it in range(1, iterations+1):
		#Calculating the gradients
		grad0 = 1/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
		grad1 = 1/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
		
		#processing the parameters
		

		m0 = beta1*m0 + (1-beta1)*grad0
		mt0 = m0 / (1-beta1**it)
		v0 = beta2*v0 + (1-beta2)*(grad0**2)
		vt0 = v0 / (1-beta2**it)
		u0 = max(beta2*v0, np.sqrt(v0))


		m1 = beta1*m1 + (1-beta1)*grad1
		mt1 = m1 / (1-beta1**it)
		v1 = beta2*v1 + (1-beta2)*(grad1**2)
		vt1 = v1 / (1-beta2**it)
		u1 = max(beta2*v1, np.sqrt(v1))


		#updating the parameters
		t0 = t0 - (lr * mt0)/u0
		t1 = t1 - (lr * mt1)/u1
		loss[it-1] = cost(m, t0, t1, x, y)
		#if it%1000==0:
		#	print(loss[it])
	return count, loss, t0, t1


#Implementation

max_iter = 10000
alpha = 1
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

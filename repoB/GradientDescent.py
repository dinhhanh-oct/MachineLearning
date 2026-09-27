import numpy as np

def myGD(grad, x0, eta, max_iter=100):
    x = [x0]
    for it in range(max_iter):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

# Bài 1: f(x) = x^2 - 2,  f'(x) = 2x
def cost1(x): return x**2 - 2
def grad1(x): return 2*x

# Bài 2: g(x) = (1/3)x^3 - x,  g'(x) = x^2 - 1
def cost2(x): return (1/3)*x**3 - x
def grad2(x): return x**2 - 1

print("Bài 1:")
for x0 in [-5, 5]:
    x, it = myGD(grad1, x0, .1)
    print('x0 = %d -> x = %f, cost = %f, after %d iterations' % (x0, x[-1], cost1(x[-1]), it))

print("Bài 2:")
for x0 in [0.5, 5]:
    x, it = myGD(grad2, x0, .1)
    print('x0 = %.1f -> x = %f, cost = %f, after %d iterations' % (x0, x[-1], cost2(x[-1]), it))

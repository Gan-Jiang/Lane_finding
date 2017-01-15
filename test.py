import numpy as np
import scipy.optimize

def f(y):
    p1,p2,p3,p4 = y[:4]
    #t1 = (1-p1)*(1-p2)*(1-p3)*(1-p4)-0.1
    t2 = p1*(1-p2)*(1-p3)*(1-p4)+p2*(1-p1)*(1-p3)*(1-p4)+p3*(1-p1)*(1-p2)*(1-p4)+p4*(1-p1)*(1-p2)*(1-p3)-0.2
    t3 = p1*p2*(1-p3)*(1-p4)+ p1*p3*(1-p2)*(1-p4)+ p1*p4*(1-p2)*(1-p3)+ p2*p3*(1-p1)*(1-p4)+ p2*p4*(1-p1)*(1-p3)+ p3*p4*(1-p1)*(1-p2)-0.2
    t4 = p1*p2*p3*(1-p4)+p1*p2*p4*(1-p3)+p1*p3*p4*(1-p2)+p2*p3*p4*(1-p1)-0.4
    t5 = p1*p2*p3*p4-0.1
    return [t2,t3,t4,t5]

x0 = np.array([0.95,0.8,0.6,0.2])

sol = scipy.optimize.root(f, x0, method='hybr')
sol2 = scipy.optimize.root(f, sol.x, method='lm')
print(f(sol2.x))
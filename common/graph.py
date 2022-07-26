from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
name = "Correct"
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(-1,8, 10000)
y1 = norm.rvs(loc=5,scale=1,size=10000)
y2 = norm.rvs(loc=2,scale=1,size=10000)
y3 = y1+y2
w = np.linspace(0,1,10000)
z1 = norm.ppf(w,loc=5,scale=1)
z2 = norm.ppf(w,loc=2, scale=1)
#z3 = np.quantile(y3,w)
z3 = norm.ppf(w,loc=3.5,scale=0.4)
z4 = (z1+z2)/2+z3
#z3 = z1+z2
"""ax.plot(w, z1,'r-', lw=1, alpha=0.6, label='Quantile function (z1)')
ax.plot(w, z2,'b', lw=1, alpha=0.6, label='Quantile function (z2)')
ax.plot(w, z3,'g', lw=1, alpha=0.6, label='State_bias (z3)')
ax.plot(w, z4,'orange', lw=1, alpha=0.6, label='Quantile mixture ((z1+z2)/2+z3)')
plt.legend()
plt.title("{} Addition".format(name))
plt.savefig('../result/graph/{}.png'.format(name))
plt.show()"""

pdf1 = norm.cdf(x,loc=5, scale=1)
pdf2 = norm.cdf(x,loc=2, scale=1)
pdf3 = (pdf1+pdf2)/2
ax.plot(pdf1,x,'r-', lw=1, alpha=0.6, label='Quantile function N(2,1)')
ax.plot(pdf2,x,'b', lw=1, alpha=0.6, label='Quantile function N(5,1)')
ax.plot(pdf3,x,'g', lw=1, alpha=0.6, label='Quantile mixture')
plt.legend()
plt.title("{} Addition".format(name))
plt.savefig('../result/graph/{}.png'.format(name))
plt.show()
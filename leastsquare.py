import numpy as np 

x = [ 2 , 4 , 6 , 8]
y = [ 81, 93, 91, 97]

mx = np.mean(x) 
my = np.mean(y)

divisor = sum([(i-mx)**2 for i in x])
## subtract mean x from each x and square it

def top(x,mx,y,my):
    d = 0
    for i in range(len(x)):
        d+=(x[i]-mx) * (y[i]-my)
    return d

dividend = top(x,mx,y,my)

a = dividend/divisor
b = my - (mx*a)
print("기울기 a = ", a)
print("y 절편 b=",  b)


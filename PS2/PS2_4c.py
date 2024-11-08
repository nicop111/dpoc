import numpy as np

#Initialisation
J1 = 0
J2 = 0
J1_old = np.inf
J2_old = np.inf
my1 = "advertise"
my2 = "research"

#Parameter
alpha = 0.99
it = 0

#Value iteration
while np.abs(J1 - J1_old) > 1e-6 or np.abs(J2 - J2_old) > 1e-6:
    it += 1
    J1_old = J1
    J2_old = J2
    J1 = np.max([4+alpha*(4/5*J1+1/5*J2), 6+alpha*(1/2*J1+1/2*J2)])
    J2 = np.max([-5+alpha*(7/10*J1+3/10*J2), -3+alpha*(2/5*J1+3/5*J2)])

#Evaluate optimal actions
if (4+alpha*(4/5*J1+1/5*J2)) > (6+alpha*(1/2*J1+1/2*J2)):
    my1 = "research"
else:
    my1 = "dont research"

if (-5+alpha*(7/10*J1+3/10*J2)) > (-3+alpha*(2/5*J1+3/5*J2)):
    my2 = "advertise"
else:
    my2 = "dont advertise"

print("Value iteration converged after " + str(it) + " iterations.")
print("J1 = ", J1)
print("J2 = ", J2)
print("my1 = " + my1)
print("my2 = " + my2)

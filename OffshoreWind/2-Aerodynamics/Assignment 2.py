import numpy as np
import matplotlib.pyplot as plt

a1=np.linspace(0,1,1000)
a2=np.linspace(0,1,1000)

Cp=np.zeros((len(a1),len(a2)))


for i in range (len(a1)) : 
    for j in range (len(a2)):
        Cp[i,j]=4*a1[i]*(1-a1[i])**2+4*a2[j]*(1-a2[j])**2*(1-2*a1[i])**3

Cp_max = Cp.max()

# Indices where the maximum occurs
indices = np.argwhere(Cp == Cp_max)[0]

print("Max value for 2 turbines:", Cp_max)
print("Indices:", indices)
print("a1=",a1[indices[0]], "a2=",a2[indices[1]])

#For 3 turbines
a1=np.linspace(0,1,100)
a2=np.linspace(0,1,100)
a3=np.linspace(0,1,100)

Cp_3t=np.zeros((len(a1),len(a2),len(a3)))


for i in range (len(a1)) : 
    for j in range (len(a2)):
        for k in range(len(a3)):
            Cp_3t[i,j,k]=4*a1[i]*(1-a1[i])**2 + 4*a2[j]*(1-a2[j])**2*(1-2*a1[i])**3 + 4*a3[k]*(1-a3[k])**2*(1-2*a1[i])**3*(1-2*a2[j])**3

Cp_max_3t = Cp_3t.max()

# Indices where the maximum occurs
indices = np.argwhere(Cp_3t == Cp_max_3t)[0]

print("Max value for 3 turbines:", Cp_max_3t)
print("Indices:", indices)
print("a1=",a1[indices[0]], "a2=",a2[indices[1]],"a3=",a3[indices[2]])

#For 4 turbines
a1=np.linspace(0,1,50)
a2=np.linspace(0,1,50)
a3=np.linspace(0,1,50)
a4=np.linspace(0,1,50)

Cp_4t=np.zeros((len(a1),len(a2),len(a3),len(a4)))


for i in range (len(a1)) : 
    for j in range (len(a2)):
        for k in range(len(a3)):
            for l in range(len(a4)):
                Cp_4t[i,j,k,l]=4*a1[i]*(1-a1[i])**2 + 4*a2[j]*(1-a2[j])**2*(1-2*a1[i])**3 + 4*a3[k]*(1-a3[k])**2*(1-2*a1[i])**3*(1-2*a2[j])**3 + 4*a4[l]*(1-a4[l])**2*(1-2*a1[i])**3*(1-2*a2[j])**3*(1-2*a3[k])**3

Cp_max_4t = Cp_4t.max()

# Indices where the maximum occurs
indices = np.argwhere(Cp_4t == Cp_max_4t)[0]

print("Max value for 4 turbines:", Cp_max_4t)
print("Indices:", indices)
print("a1=",a1[indices[0]], "a2=",a2[indices[1]],"a3=",a3[indices[2]],"a4=",a4[indices[3]])

#plot 
Cp_1t=0.593
Cp_list=[Cp_1t,Cp_max,Cp_max_3t,Cp_max_4t]
nb_tur=[1,2,3,4]


plt.show()
# Plot the graph with a smoother line and dots at each point
plt.plot(nb_tur, Cp_list, linestyle='-', marker='o', color='b', label='Cp vs Number of Turbines')

# Add labels and a legend
plt.xlabel('Number of Turbines')
plt.ylabel('Cp')
plt.title('Power Coefficient vs Number of Turbines')
plt.grid(True)
plt.legend()

# Display the plot
plt.show()
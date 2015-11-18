import numpy as np
import numpy.linalg as lingal
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import math
import skimage as ski
import skimage.io
from mpl_toolkits.mplot3d import Axes3D


# parameters
lx=50  # length of domain in the x direction
ly=20  # length of domain in the y direction
w=np.array([4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,
    1.0/36.0,1.0/36.0,1.0/36.0]) # weights for directions
cx=np.array([0,1,0,-1,0,1,-1,-1,1]) # direction vector for the x direction
cy=np.array([0,0,1,0,-1,1,1,-1,-1]) # direction vector for the y direction
tau=1
g=10**-4
cs=1/math.sqrt(3)
timestep=1000 # time steps for the simulation
timeplot=1 # plotimeing interval

# initializing variables
rho=np.ones((lx,ly)) # initializing the rho
u=np.zeros((lx,ly)) # initializing the horizontal velocities
v=np.zeros((lx,ly)) # initializing the vertical velocities
f=np.zeros((9,lx,ly)) # initializing f
conn=[0,3,4,1,2,7,8,5,6] #index of respective connectors

#this initializes the f function to be the weight ttimes the initial desnity
for j in range(9):
        f[j,:,:]=w[j]*rho
ff=f
#beging time incrementation loop, all of the code goes into this loop and analysis
#is done over time incrementation
for time in range(timestep):
    #in this part we define the macroscopic behaviour, where the physics dictates how
    #the particle distribution is migrated over the time incrementation
    for i in range(lx):
        for k in range(ly):
            rho[i,k]=0
            
            for j in range(9):
                rho[i,k]=rho[i,k]+f[j,i,k]
                u[i,k]=u[i,k]+f[j,i,k]*cx[j]
                v[i,k]=v[i,k]+f[j,i,k]*cy[j]
                
            u[i,k]=u[i,k]/rho[i,k]
            v[i,k]=v[i,k]/rho[i,k]

    #here we define the collision array and is related as described in the problem
    #statement from above
    for i in range(lx):
        for k in range(ly):
            u[i,k]=u[i,k]+g*tau/rho[i,k]
            for j in range(9):
                cc=(1/cs**2)*(cx[j]*u[i,k]+cy[j]*v[i,k])
                feq=rho[i,k]*w[j]*(1.0+cc+cc**2+v[i,k]**2)/(2*cs**2)
                f[j,i,k]=f[j,i,k]+(1/tau)*(feq-f[j,i,k])

    #rendefines for the next time step
    ff=f
    #boundary conditions
    #north periodicity
    for i in range(lx):
        for j in range(9):
            ff[j,i,ly-1]=f[conn[j],i,ly-1]
            ff[j,i,0]=f[conn[j],i,0]

	# in this loop we move the particles and essentially do the equivalent of a 
	# circshift in MATLAB
    for j in range(9):
        f[j,:,:]=np.roll(np.roll(ff[j,:,:],cx[j],axis=0),cy[j],axis=1)
        
	#this part is for the plotting and only takes a few plot instances over time
    plt.figure()
    if not time%timeplot:
        #print the current time step of the simulation
        print "Time ",time," of ",timestep," completed" 
        u_plot=np.array(np.transpose(u))
        allx=np.array(range(lx+1))
        ally=np.array(range(ly+1))
        
        #create the 2D color plot
        fig, ax = plt.subplots()
        plt.xlabel(r'x'); plt.ylabel(r'y')
        heatmap = ax.pcolor(allx, ally, u_plot,vmin=0,vmax=0.0003)
        colorbar = plt.colorbar(heatmap)
        colorbar.set_label(r'value of velocity')
        plt.gcf().set_size_inches(17, 7)
        plt.draw()
        plt.pause(0.01)

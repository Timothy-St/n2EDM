import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import struct 
from scipy import interpolate


def series(t, a, b, T=2*np.pi):
    """
    takes: t scalar or array, a n-dim array /!\, b (n-1)-dim array /!\
    returns: fourier series
    """
    t = t * np.pi /180    
    f = 0
    for n in range(len(a)):
        f += a[n]*np.cos(n*t*2*np.pi/T) 
        if n>0:
            f += b[n-1]*np.sin(n*t*2*np.pi/T)
            
    return f

def seriesFit(y0, t, N, sigma=0, T=2*np.pi):
    """
    takes: y0 array (values to be fitted), t array (fit points), N order of the fit, variance of the measure points
    returns: 2*N+1 size array of fit parameters [a_0, a_1, ..., a_N, b_1, ..., b_N], along with covariance of the parameters
    """
    
    u = np.zeros(2*N+1)
    U = np.zeros((2*N+1, 2*N+1))
    I = len(t)
    t = t * np.pi /180
    
    for i in range(I):
        for n in range(2*N+1):
            
            if n<=N:
                u[n] += y0[i]*np.cos(n*t[i]*2*np.pi/T)
                for m in range(2*N+1):
                    if m<=N:
                        U[n][m] += np.cos(m*t[i]*2*np.pi/T)*np.cos(n*t[i]*2*np.pi/T)
                    else:
                        U[n][m] += np.sin((m-N)*t[i]*2*np.pi/T)*np.cos(n*t[i]*2*np.pi/T)
                    
            else:
                u[n] += y0[i]*np.sin((n-N)*t[i]*2*np.pi/T)
                for m in range(2*N+1):
                    if m<N:
                        U[n][m] += np.cos(m*t[i]*2*np.pi/T)*np.sin((n-N)*t[i]*2*np.pi/T)
                    else:
                        U[n][m] += np.sin((m-N)*t[i]*2*np.pi/T)*np.sin((n-N)*t[i]*2*np.pi/T)
    
    Uinv = np.linalg.inv(U)

    par = np.dot(Uinv, u)
    cov = sigma**2*np.array([Uinv[i][i] for i in range(2*N+1)])
    
    return par, cov

def Cn(ab, y, n, T):
    """
    takes: ab (string, 'a' or 'b'), y data (array), n order of coef (int), T period (float)
    returns: Fourier coef (float)
    """
    
    phi = np.linspace(0, 2*np.pi, len(y))
    
    if ab=='a':
        C = (2/T)*scipy.integrate.simps(y * np.cos(2*np.pi*n*phi/T), phi)
    if ab=='b':
        C = (2/T)*scipy.integrate.simps(y * np.sin(2*np.pi*n*phi/T), phi)
        
    return C

def seriesDo(y, x, N, T):
    
    S = 0.5*Cn('a', y, 0, T)
    
    for i in range(1, N+1):
        S +=  Cn('a', y, i, T)*np.cos(2*np.pi*i*x/T) + Cn('b', y, i, T)*np.sin(2*np.pi*i*x/T)
        
    return S


def mkValues_Bin(filename):
    """
    takes: filepath from .bin files
    returns: values
    """
    # can get this from n2EDM header:
    struct_fmt = '=Q12dQ' # Q means unsigned 64 bit int, 4Q is four of them

    #some calculations for reading 'line by line'
    str_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    def read_chunks(f, length):
     while True:
      data = f.read(length)
      if not data: break
      yield data

    with open(filename, "rb") as f:
     results = [struct_unpack(chunk) for chunk in read_chunks(f, str_len)]

    #need to generate format string from n2EDM header
    rows =[]
    for resline in results:
        rows.append(resline)

    values = np.zeros((len(rows),10))

    # extract array in the right from:
    # We want to go from (time, rho, rhoSD, phi, phiSD, z, zSD, Brho, BrhoSD, Bphi, BphiSD, Bz, BzSD)
    # to                 (time, rho, phi, z, Brho, Bphi, Bz, BrhoSD, BphiSD, BzSD)
    indices= np.array([0,1,3,5,7,9,11,8,10,12]) 
    rows = np.array(rows)
    for k,i in zip(range(0,10),indices):
        values[:,k] = rows[:,i]
        
    #values = np.transpose([[float(cell) for cell in rows[i]] for i in range(2, len(rows))])
    values[:,0]= values[:,0]-values[0][0]
    return  values.T

"""________Functions for the binary files________"""


def rho_mkRing_Bin(values,start,Drho):
    """
    takes: position and field values, start index of a ring, ring axis (1:rho, 2:phi, 3:z, 4:t)
    returns: index of the next ring, position values, field values for that ring.
    
    Data is saved into one ring as long as rho remains constant. 
    Wrongly created rings are going to be filterd out due to their short running time in mk_Cyl_Bin.
    """
    delta = 7
    incr_const = 50
    
    r = []
    phi = []
    z = []
    
    Bphi = [] # fluxgate Bx
    Bz = [] # fluxgate By
    Br = [] # fluxgate Bz
    
    t= []
    
    i = start
    length = len(values[0])

    while i< length-1 and Drho[i]<delta:
        r.append(values[1][i]*0.1) # von mm in cm  
        phi.append((values[2][i] % (360)) )# in deg
        z.append(values[3][i]*0.1) # in cm
        Bphi.append(values[5][i]*1000) # in nT   #registered data is in muT
        Bz.append(values[6][i]*1000) # in nT 
        Br.append(values[4][i]*1000) # in nT 
        t.append(values[0][i])
        i +=1
    
    newStart = i 

    R = [r, phi, z,t]
    B = [Br, Bphi, Bz]
    
    return newStart, R, B
    

def mkCyl_Bin (values):
    """
    takes: position and field values.
    returns: position and field values for all rings along the rho axis. R=(rho,phi,z,t)
    """
    R = []
    B = []
    
    Drho = abs(values[1,1:]-values[1,:-1])
    
    i = 1
    while i<(len(values[0])-2):
        i, r, b = rho_mkRing_Bin(values, i+1,Drho)
        R.append(r)
        B.append(b)
    
    R_filter = []
    B_filter = []
    for i in range(len(R)):
        if len(R[i][0])> 300:
            R_filter.append(R[i])
            B_filter.append(B[i])
    
    return R_filter, B_filter

def mkValues_Bin(filename):
    """
    takes: filepath from .bin files
    returns: values: (time, rho, phi, z, Brho, Bphi, Bz)
    """
    # can get this from n2EDM header:
    struct_fmt = '=Q16dQ' # Q means unsigned 64 bit int, 4Q is four of them
    
    # B-flied factor used for fluxgate calibration (muT)
    B_factor = 1

    #some calculations for reading 'line by line'
    str_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    def read_chunks(f, length):
     while True:
      data = f.read(length)
      if not data: break
      yield data

    with open(filename, "rb") as f:
     results = [struct_unpack(chunk) for chunk in read_chunks(f, str_len)]

    #need to generate format string from n2EDM header
    rows =[]
    for resline in results:
        rows.append(resline)

    values = np.zeros((len(rows),7))

    # extract array in the right from:
    # We want to go from (time, Mphi, Cphi, Ephi, CEphi, Mrho, Crho, Erho, CErho, Mz, Cz, Ez, ECz, AIN1, AIN2, AIN3, AIN4)  
    #                    # M = Motor; C = Calculated; E = Encoder; We use CE
    # to                 (time, rho, phi, z, Brho, Bphi, Bz)
    indices= np.array([0,8,4,12,13,15,14]) 
    rows = np.array(rows)
    for k,i in zip(range(0,7),indices):
        values[:,k] = rows[:,i]
    
    #Calculating the B-field:
    values[:,4] = values[:,4]*B_factor
    values[:,5] = values[:,5]*B_factor
    values[:,6] = values[:,6]*B_factor
    
    values[:,0] = values[:,0]-values[0][0]
    return  values.T

def Simple_Plotting(R,B, ring, Bvar, arg = 'phi', color = 'tab:blue'):
    assert(arg == 'phi' or arg == 'z')
    Y = np.array(B[ring][Bvar])
    
    if(arg == 'phi'):
        X = np.array(R[ring][1])
        plt.xlabel(r"$\varphi$ (°)", size=15)
        plt.xlim(-5, 365)
        plt.xticks(np.arange(0, 361, 30))
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Ring scan at z={:.0f} cm and $\\rho$={:.0f} cm".format(R[ring][2][0], R[ring][0][0]))
    else :
        X = np.array(R[ring][2])
        plt.xlabel(r"z (cm)", size=15)
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Axis scan at rho={:.0f} cm".format(R[ring][0][0]))
    
    
    if Bvar==0:
        plt.ylabel(r"$B_\rho$ (nT)", size=15)
    if Bvar==1:
        plt.ylabel(r"$B_\varphi$ (nT)", size=15)
    if Bvar==2:
        plt.ylabel(r"$B_z$ (nT)", size=15)

    #plt.title('source:{}'.format(file1))
    plt.grid()
    plt.legend()

    
def avergeing_point(R,Bfield,init,dif):
    """ 
    takes: position and field values, index from where to average, array of differences along averaging direction.
    returns: Averaged point.
    (Variances are not forwared at the moment)
    """
    i = init
    r_av,phi_av, z_av = 0,0,0
    Br_av, Bphi_av, Bz_av = 0,0,0
    while(i < len(R[0])-1 and dif[i]<0.05 ):
        r_av = r_av + R[0][i]
        phi_av = phi_av + R[1][i]
        z_av = z_av + R[2][i]
        Br_av = Br_av + Bfield[0][i]
        Bphi_av = Bphi_av + Bfield[1][i]
        Bz_av = Bz_av + Bfield[2][i]
        i = i +1
    
    N = (i-init) 
    if (N ==0):
        N = 1
    r_av = r_av / N
    phi_av = phi_av/ N
    z_av = z_av / N    
    Br_av = Br_av /N
    Bphi_av = Bphi_av /N
    Bz_av = Bz_av /N
    
    r_variance = np.sqrt(((R[0][init:i]-r_av)**2).sum() / N )
    phi_variance = np.sqrt(((R[1][init:i]-phi_av)**2).sum() / N )
    z_variance = np.sqrt(((R[2][init:i]-z_av)**2).sum() / N )
    Br_variance = np.sqrt(((Bfield[0][init:i]-Br_av)**2).sum() / N )
    Bphi_variance = np.sqrt(((Bfield[1][init:i]-Bphi_av)**2).sum() / N )
    Bz_variance = np.sqrt(((Bfield[2][init:i]-Bz_av)**2).sum() / N )
    
    B_av = [Br_av,Bphi_av,Bz_av]
    B_var = [Br_variance,Bphi_variance,Bz_variance]
    
    return r_av,phi_av,z_av, Br_av ,Bphi_av , Bz_av, i

def averager_ring(R, B, ring_no, mode = 'phi'):
    """ Averages points for in one ring.
    """
    assert(mode == 'phi' or 'z')
    if(mode == 'phi'):
        d = 1
    elif (mode == 'z'):
        d = 2
    d_ring = R[ring_no][d]  #ring of relevant step coordinate for discretisation 
    Bfield = B[ring_no]
    dif = abs(np.array(d_ring[1:]) - np.array(d_ring[:-1]))
    
    r_new = []
    phi_new = []
    z_new = []
    
    Br_av = []
    Bphi_av = []
    Bz_av = []
    
    p_std = []
    prev = d_ring[0]

    i = 0
    while i in range(len(d_ring)-1):
        if dif[i]<0.05:
            r,phi,z,Br ,Bphi ,Bz , i = avergeing_point(R[ring_no],Bfield,i,dif)
            r_new.append(r)
            phi_new.append(phi)
            z_new.append(z)
            Br_av.append(Br)
            Bphi_av.append(Bphi)
            Bz_av.append(Bz)
            #p_std.append(var)
        i = i+1
    R = [r_new, phi_new,z_new]
    B = [Br_av, Bphi_av, Bz_av]
    
    return R,B
    

def average_discrete_data(R,B, mode = 'phi'):
    """ Averages points of a discrete measurement sequence.
    mode: significant measurment direction (either 'phi' or 'z').
    R postion values sorted in rings, B mag. field values.
    """
   
    assert(mode == 'phi' or 'z')
    
    R_new = []
    B_new = []
    rlen = len(R)
    for ring_no in range(len(R)):
        R_av,B_av = averager_ring(R,B,ring_no, mode)
        R_new.append(R_av)
        B_new.append(B_av)
        
    return R_new, B_new

def Plotting_Sum(R1,B1,R2,B2, ring, Bvar, save = False, no_points = 180, arg = 'phi', color = 'tab:blue'):
    """
    Plots B1+B2 with interpolarisation. To be used for continous data maps.
    no_points: number of points created in the plot.
    """
    assert(arg == 'phi' or arg == 'z')
    
    Y_1 = np.array(B1[ring][Bvar])
    Y_2 = np.array(B2[ring][Bvar])
    
    if(arg == 'phi'):
        X1 = np.array(R1[ring][1])
        X2 = np.array(R2[ring][1])
        up_b= min(X1.max(),X2.max())
        low_b = max(X1.min(),X2.min())
        X = np.linspace(low_b,up_b,no_points)
        Y1_i = interpolate.interp1d(X1,Y_1)
        Y2_i = interpolate.interp1d(X2,Y_2)
        Y1=Y1_i(X)
        Y2=Y2_i(X)       
        Y = Y1+Y2
        plt.xlabel(r"$\varphi$ (°)", size=15)
        plt.xlim(-5, 365)
        plt.xticks(np.arange(0, 361, 30))
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Ring scan at z={:.0f} cm and $\\rho$={:.0f} cm".format(R1[ring][2][0], R1[ring][0][0]))
    else :
        X1 = np.array(R1[ring][2])
        X2 = np.array(R2[ring][2])
        up_b= min(X1.max(),X2.max())
        low_b = max(X1.min(),X2.min())
        if (no_points == 'step'):
            no_points = len(X1)
        X = np.linspace(low_b,up_b,no_points)
        Y1_i = interpolate.interp1d(X1,Y_1)
        Y2_i = interpolate.interp1d(X2,Y_2)
        Y1=Y1_i(X)
        Y2=Y2_i(X)       
        Y = Y1+Y2
        plt.xlabel(r"z (cm)", size=15)
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Axis scan at rho={:.0f} cm".format(R1[ring][0][0]))
    
    
    if Bvar==0:
        plt.ylabel(r"$B_\rho$ (nT)", size=15)
    if Bvar==1:
        plt.ylabel(r"$B_\varphi$ (nT)", size=15)
    if Bvar==2:
        plt.ylabel(r"$B_z$ (nT)", size=15)
        
    plt.grid()
    plt.legend()
    
def Plotting_Diff(R1,B1,R2,B2, ring, Bvar, save = False, no_points = 180, arg = 'phi', color = 'tab:blue'):
    """Plots B1 - B2 with interpolarisation. To be used for continous data maps.
    no_points: number of points created in the plot.
    """
    assert(arg == 'phi' or arg == 'z')
    Y_1 = np.array(B1[ring][Bvar])
    Y_2 = np.array(B2[ring][Bvar])
    
    
    
    if(arg == 'phi'):
        X1 = np.array(R1[ring][1])
        X2 = np.array(R2[ring][1])
        up_b= min(X1.max(),X2.max())
        low_b = max(X1.min(),X2.min())
        X = np.linspace(low_b,up_b,no_points)
        Y1_i = interpolate.interp1d(X1,Y_1)
        Y2_i = interpolate.interp1d(X2,Y_2)
        Y1=Y1_i(X)
        Y2=Y2_i(X) 
        Y = Y2-Y1
        plt.xlabel(r"$\varphi$ (°)", size=15)
        plt.xlim(-5, 365)
        plt.xticks(np.arange(0, 361, 30))
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Difference: Ring scan at z={:.0f} cm and $\\rho$={:.0f} cm".format(R1[ring][2][0], R1[ring][0][0]))
    else :
        X1 = np.array(R1[ring][2])
        X2 = np.array(R2[ring][2])
        up_b= min(X1.max(),X2.max())
        low_b = max(X1.min(),X2.min())
        X = np.linspace(low_b,up_b,no_points)
        Y1_i = interpolate.interp1d(X1,Y_1)
        Y2_i = interpolate.interp1d(X2,Y_2)
        Y1=Y1_i(X)
        Y2=Y2_i(X)       
        Y = Y2-Y1
        plt.xlabel(r"z (cm)", size=15)
        plt.plot(X, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Axis scan at rho={:.0f} cm".format(R1[ring][0][0]))
    
    
    if Bvar==0:
        plt.ylabel(r"$B_\rho$ (nT)", size=15)
    if Bvar==1:
        plt.ylabel(r"$B_\varphi$ (nT)", size=15)
    if Bvar==2:
        plt.ylabel(r"$B_z$ (nT)", size=15)

    plt.grid()
    plt.legend()

    
def Plotting_Sum_Step(R1,B1,R2,B2, ring, Bvar,arg = 'phi', color = 'tab:blue'):
    """Plots for step data maps the sum."""
    assert(arg == 'phi' or arg == 'z')
    
    Y1 = np.array(B1[ring][Bvar])
    Y2 = np.array(B2[ring][Bvar])
    
    if(arg == 'phi'):
        X1 = np.array(R1[ring][1])
        X2 = np.array(R2[ring][1])
        Y = Y1+Y2
        plt.xlabel(r"$\varphi$ (°)", size=15)
        plt.xlim(-5, 365)
        plt.xticks(np.arange(0, 361, 30))
        plt.plot(X1, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Ring scan at z={:.0f} cm and $\\rho$={:.0f} cm".format(R1[ring][2][0], R1[ring][0][0]))
    else :
        X1 = np.array(R1[ring][2])
        X2 = np.array(R2[ring][2])
        Y = Y1+Y2
        plt.xlabel(r"z (cm)", size=15)
        plt.plot(X1, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Axis scan at rho={:.0f} cm".format(R1[ring][0][0]))
    
    
    if Bvar==0:
        plt.ylabel(r"$B_\rho$ (nT)", size=15)
    if Bvar==1:
        plt.ylabel(r"$B_\varphi$ (nT)", size=15)
    if Bvar==2:
        plt.ylabel(r"$B_z$ (nT)", size=15)

    plt.grid()
    plt.legend()    
    
    
def Plotting_Diff_Step(R1,B1,R2,B2, ring, Bvar,arg = 'phi', color = 'tab:blue'):
    """Plots for step data maps the difference."""
    assert(arg == 'phi' or arg == 'z')
    
    Y1 = np.array(B1[ring][Bvar])
    Y2 = np.array(B2[ring][Bvar])
    
    if(arg == 'phi'):
        X1 = np.array(R1[ring][1])
        X2 = np.array(R2[ring][1])
        Y = Y1-Y2
        plt.xlabel(r"$\varphi$ (°)", size=15)
        plt.xlim(-5, 365)
        plt.xticks(np.arange(0, 361, 30))
        plt.plot(X1, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Ring scan at z={:.0f} cm and $\\rho$={:.0f} cm".format(R1[ring][2][0], R1[ring][0][0]))
    else :
        X1 = np.array(R1[ring][2])
        X2 = np.array(R2[ring][2])
        Y = Y1-Y2
        plt.xlabel(r"z (cm)", size=15)
        plt.plot(X1, Y, color= color, marker='.', linestyle="none", markersize=8, label="Sum: Axis scan at rho={:.0f} cm".format(R1[ring][0][0]))
    
    
    if Bvar==0:
        plt.ylabel(r"$B_\rho$ (nT)", size=15)
    if Bvar==1:
        plt.ylabel(r"$B_\varphi$ (nT)", size=15)
    if Bvar==2:
        plt.ylabel(r"$B_z$ (nT)", size=15)

    plt.grid()
    plt.legend()
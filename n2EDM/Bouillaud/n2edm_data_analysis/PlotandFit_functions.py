import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def mkRing(values, start, ringaxis):
    """
    takes: position and field values, start index of a ring, ring axis (1:rho, 2:phi, 3:z)
    returns: end index of the ring, position values, field values for that ring.
    """
    r = []
    phi = []
    z = []
    
    Bphi = [] # fluxgate Bx
    Bz = [] # fluxgate By
    Br = [] # fluxgate Bz
                                                                                  
    i = start
    
    if ringaxis==2:
        
        while values[ringaxis][i]<360:
            
            r.append(values[1][i]*0.1) # in cm
            phi.append(values[2][i]*np.pi/180) # in rad
            z.append(values[3][i]*0.1 - 57.5) # in cm
            Bphi.append(values[4][i]*1000) # in pT
            Bz.append(values[6][i]*1000) # in pT 
            Br.append(values[8][i]*1000) # in pT 

            i += 1
            if i>(len(values[0])-2):
                break 
        
    else:
        
        while values[ringaxis][i+1]==values[ringaxis][i]:

            r.append(values[1][i]*0.1) # in cm
            phi.append(values[2][i]*np.pi/180) # in rad
            z.append(values[3][i]*0.1 - 57.5) # in cm
            Bphi.append(values[4][i]*1000) # in pT
            Bz.append(values[6][i]*1000) # in pT 
            Br.append(values[8][i]*1000) # in pT 

            i += 1
            if i>(len(values[0])-2):
                break 
    
    R = [r, phi, z]
    B = [Br, Bphi, Bz]
    
    return i, R, B

def mkRing2(values, start, ringaxis):
    """
    takes: position and field values, start index of a ring, ring axis (1:rho, 2:phi, 3:z)
    returns: end index of the ring, position values, field values for that ring.
    """
    r = []
    phi = []
    z = []
    
    Bphi = [] # fluxgate Bx
    Bz = [] # fluxgate By
    Br = [] # fluxgate Bz
                                                                                  
    i = start
    
    if ringaxis==2:
        
        while values[ringaxis][i]<360:
            
            r.append(values[1][i]*0.1) # in cm
            phi.append(values[2][i]*np.pi/180) # in rad
            z.append(values[3][i]*0.1 - 57.5) # in cm
            Bphi.append(values[4][i]*1000) # in pT
            Bz.append(values[6][i]*1000) # in pT 
            Br.append(values[8][i]*1000) # in pT 

            i += 1
            if i>(len(values[0])-2):
                r.append(values[1][i]*0.1) # in cm
                phi.append(values[2][i]*np.pi/180) # in rad
                z.append(values[3][i]*0.1 - 57.5) # in cm
                Bphi.append(values[4][i]*1000) # in pT
                Bz.append(values[6][i]*1000) # in pT 
                Br.append(values[8][i]*1000) # in pT 
                break 
        
    else:
        
        while values[ringaxis][i+1]==values[ringaxis][i]:

            r.append(values[1][i]*0.1) # in cm
            phi.append(values[2][i]*np.pi/180) # in rad
            z.append(values[3][i]*0.1 - 57.5) # in cm
            Bphi.append(values[4][i]*1000) # in pT
            Bz.append(values[6][i]*1000) # in pT 
            Br.append(values[8][i]*1000) # in pT 

            i += 1
            if i>(len(values[0])-2):
                r.append(values[1][i]*0.1) # in cm
                phi.append(values[2][i]*np.pi/180) # in rad
                z.append(values[3][i]*0.1 - 57.5) # in cm
                Bphi.append(values[4][i]*1000) # in pT
                Bz.append(values[6][i]*1000) # in pT 
                Br.append(values[8][i]*1000) # in pT 
                break 
    
    R = [r, phi, z]
    B = [Br, Bphi, Bz]
    
    return i, R, B

def mkCyl(values, ringaxis):
    """
    takes: position and field values.
    returns: position and field values for all rings along a vertical scan.
    """
    R = []
    B = []
    
    i = -1
    while i<(len(values[0])-2):
        i, r, b = mkRing(values, i+1, ringaxis)
        R.append(r)
        B.append(b)
    
    return R, B

def mkValues(file):
    """
    takes: filepath
    returns: values
    """
    rows = []
    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
            
    values = np.transpose([[float(cell) for cell in rows[i]] for i in range(2, len(rows))])
    
    return values

def mk2Rings(values, start):
    """
    takes: position and field values, start index of a ring, ring axis (1:rho, 2:phi, 3:z)
    returns: end index of the ring, position values, field values for that ring.
    """
    n = len(values[0])//2

    R = np.zeros((2, 3, n))
    B = np.zeros((2, 3, n))    
    
    for i in range(start, n):
        
        R[0][0][i] = values[1][2*i]*0.1
        R[0][1][i] = values[2][2*i]*np.pi/180
        R[0][2][i] = values[3][2*i]*0.1 - 57.5
        B[0][1][i] = values[4][2*i]*1000 # B_phi
        B[0][2][i] = values[6][2*i]*1000 # B_z
        B[0][0][i] = values[8][2*i]*1000 # B_r

        R[1][0][i] = values[1][2*i+1]*0.1
        R[1][1][i] = values[2][2*i+1]*np.pi/180
        R[1][2][i] = values[3][2*i+1]*0.1 - 57.5
        B[1][1][i] = values[4][2*i+1]*1000
        B[1][2][i] = values[6][2*i+1]*1000
        B[1][0][i] = values[8][2*i+1]*1000

    return R, B
    
def series(t, a, b, T=2*np.pi):
    """
    takes: t scalar or array, a n-dim array /!\, b (n-1)-dim array /!\
    returns: fourier series
    """
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
        
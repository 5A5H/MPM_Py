import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

class Body:
    def __init__(self):
        self.NoParticles = 0
        self.x_P = np.array([])
        self.m_P = np.array([])
        self.v_P = np.array([])
        self.v_P_bar = np.array([])
        self.F_P = np.array([])
        self.L_P = np.array([])
        self.S_P = np.array([])
        self.Emod = 0
        self.nu = 0

    def add_Particle(self, XP, MP):
        self.NoParticles += 1
        self.m_P.resize(self.NoParticles)
        self.m_P[-1] = MP
        self.x_P.resize(self.NoParticles,2)
        self.x_P[-1] = np.array([XP[0], XP[1]])
        self.v_P.resize(self.NoParticles,2)
        self.v_P[-1] = np.zeros(2)
        self.v_P_bar.resize(self.NoParticles,2)
        self.v_P_bar[-1] = np.zeros(2)
        self.F_P.resize(self.NoParticles,2,2)
        self.F_P[:] = np.eye(2)
        self.L_P.resize(self.NoParticles,2,2)
        self.S_P.resize(self.NoParticles,2,2)

    def Sig(self,F):
        lam = (self.Emod*self.nu)/((1+self.nu)*(1-2*self.nu))
        mue = self.Emod/(2*(1+self.nu))
        eps = np.array([[F[0,0]-1,0.5*(F[0,1]+F[1,0]),0],[0.5*(F[0,1]+F[1,0]),F[1,1]-1,0],[0,0,0]])
        sig = lam * (eps[0,0]+eps[1,1]+eps[2,2]) * np.eye(3) + 2 * mue * eps
        return sig[:2,:2]

    def plot(self, ax, **options):
        Scale = options.get("Scale", 1.0) * options.get("S", 1.0)
        ax.scatter(
            self.x_P[:,0], self.x_P[:,1],
            s = self.m_P * Scale,
            c = np.array([[0.0,0.0,1.0,0.5]])
            )
        
        if options.get("Velocity",False) == True:
            ax.quiver(
                self.x_P[:,0], self.x_P[:,1],
                self.v_P[:,0] * Scale, self.v_P[:,1] * Scale,
            )
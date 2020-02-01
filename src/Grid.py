import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

class Grid:
    def __init__(self, XI, Elmt):
        self.NoNodes = len(XI)
        self.NoElements = len(Elmt)
        self.X_I  = XI
        self.Elmt = Elmt
        self.reset()
        self.EBC_Container = np.array([],dtype=np.int16)

    def reset(self):
        self.m_I  = np.zeros(self.NoNodes)
        self.f_I  = np.zeros(self.NoNodes*2)
        self.mv_I = np.zeros(self.NoNodes*2)
        self.a_I  = np.zeros(self.NoNodes*2)
        self.v_I  = np.zeros(self.NoNodes*2)
        self.u_I  = np.zeros(self.NoNodes*2)
        self.mv_I.resize(self.NoNodes,2)
        self.f_I.resize(self.NoNodes,2)
        self.a_I.resize(self.NoNodes,2)
        self.v_I.resize(self.NoNodes,2)
        self.u_I.resize(self.NoNodes,2)

    def addEBC(self, NodeIndex, DofIndex, Value):
        currentlen = len(self.EBC_Container)
        self.EBC_Container.resize(currentlen+1,3)
        self.EBC_Container[-1] = np.array([NodeIndex, DofIndex, Value],dtype=np.int16)

    def plot(self, ax, **options):
        Scale = options.get("Scale", 1.0) * options.get("S", 1.0)
        
        if options.get("DeformedMesh",False) == True:
            for i in range(len(self.Elmt)): 
                ax.add_patch(
                    Polygon(
                        self.X_I[self.Elmt[i]], True,
                        fc=(0.0,0.0,0.0,0.0),  
                        ec=(0.9,0.9,0.9) , lw=1.5
                        )
                    )
                ax.add_patch(
                    Polygon(
                        (self.X_I[self.Elmt[i]]+self.u_I[self.Elmt[i]]*Scale), True,
                        fc=(0.0,0.0,0.0,0.0),  
                        ec=(0.0,0.0,0.0) , lw=1.5
                        )
                    )
        else:
            for i in range(len(self.Elmt)): 
                ax.add_patch(
                    Polygon(
                        self.X_I[self.Elmt[i]], True,
                        fc=(0.0,0.0,0.0,0.0),  
                        ec=(0.0,0.0,0.0) , lw=1.5
                        )
                    )

        if options.get("NodeMarks",False) == True:
            ax.plot(
                self.X_I[:,0], self.X_I[:,1], 
                color='black', marker='o', lw=0
                )

        if options.get("Momentum",False) == True:
            ax.quiver(
                self.X_I[:,0], self.X_I[:,1],
                self.mv_I[:,0] * Scale, self.mv_I[:,1] * Scale,
            )
        if options.get("Velocity",False) == True:
            ax.quiver(
                self.X_I[:,0], self.X_I[:,1],
                self.v_I[:,0] * Scale, self.v_I[:,1] * Scale,
            )
        if options.get("Acceleration",False) == True:
            ax.quiver(
                self.X_I[:,0], self.X_I[:,1],
                self.a_I[:,0] * Scale, self.a_I[:,1] * Scale,
            )
        if options.get("Forces",False) == True:
            ax.quiver(
                self.X_I[:,0], self.X_I[:,1],
                self.f_I[:,0] * Scale, self.f_I[:,1] * Scale,
            )
    
    def SHP(self, XP):
        for E in Elmt:
            if self.PointInsideQuadQ(self.X_I[E], XP):
                xi, eta = 0.0, 0.0
                for iter in range(10):
                        NI = (1/4) * np.array(
                            [(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
                        dNI = (1/4) * np.array([
                                                [-(1-eta),(1-eta),(1+eta),-(1+eta)],
                                                [-(1-xi),-(1+xi),(1+xi),(1-xi)]])
                        XP_i = NI.dot(self.X_I[E])
                        J = dNI.dot(self.X_I[E])
                        R = XP_i - XP
                        if np.linalg.norm(R) <= 1e-8: break
                        delta_Xi = - np.linalg.inv(J).dot(R)
                        xi  += delta_Xi[0]
                        eta += delta_Xi[1]
                NI = (1/4) * np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
                dNI = (1/4) * np.array([[-(1-eta),(1-eta),(1+eta),-(1+eta)],
                                        [-(1-xi),-(1+xi),(1+xi),(1-xi)]])
                J = dNI.dot(self.X_I[E])
                dNIdx = np.linalg.inv(J).dot(dNI)
                return E,NI,dNIdx

    def PointInsideQuadQ(self, XI, XP):
        M = XI.mean(axis=0)
        NodesToMid = M - XI
        NodesToXP = XP - XI
        QuadTangents = np.array([XI[1]-XI[0],XI[2]-XI[1],XI[3]-XI[2],XI[0]-XI[3]])
        OutwardPointingVectors = np.array([QuadTangents[:,1],-QuadTangents[:,0]]).T

        for i in range(4):
            if OutwardPointingVectors[i].dot(NodesToXP[i]) >= 10e-8:
                return False
    
        return True
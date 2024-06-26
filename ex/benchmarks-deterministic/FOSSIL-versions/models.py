#Edited by Ben Wooding for purpose of comparisons with PRoTECT

#Copy these lines into the models.py file in FOSSIL to be used for the case studies for comparison

class DCMotor2(control.DynamicalModel):
    n_vars = 2
    tau = 0.01
    R = 1
    L = 0.01
    J = 0.01
    Kdc = 0.01
    b = 1

    def f_torch(self, v):
        x1, x2 = v[:, 0], v[:, 1]
        return [x1 + self.tau*(-self.R/self.L*x1 - self.Kdc/self.L*x2), x2 + self.tau*(self.Kdc/self.J*x1 - self.b/self.J*x2)]

    def f_smt(self, v):
        x1, x2 = v
        return [x1 + self.tau*(-self.R/self.L*x1 - self.Kdc/self.L*x2), x2 + self.tau*(self.Kdc/self.J*x1 - self.b/self.J*x2)]

class Jet2(control.DynamicalModel):
    n_vars = 2

    def f_torch(self, v):
        x1, x2 = v[:, 0], v[:, 1]
        return [-x2-1.5*x1**2-0.5*x1**3, x1]

    def f_smt(self, v):
        x1, x2 = v
        return [-x2-1.5*x1**2-0.5*x1**3, x1]

        
class RoomTemp1d(control.DynamicalModel):
    n_vars = 1

    tau = 5 * 60  # discretise param
    temp_e = 15  # external temp

    def f_torch(self, v):
        x = v[:, 0]
        return [
            x + self.tau * (self.temp_e - x)
        ]

    def f_smt(self, v):
        x = v[0]
        return [
            x + self.tau * (self.temp_e - x)
        ]

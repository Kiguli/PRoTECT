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
        
class HighOrd8B(control.DynamicalModel):
    n_vars = 8

    def f_torch(self, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = (
            v[:, 0],
            v[:, 1],
            v[:, 2],
            v[:, 3],
            v[:, 4],
            v[:, 5],
            v[:, 6],
            v[:, 7],
        )
        return [
            x1 - 50*x2,
            x2 -50*x3,
            x3 -50*x4,
            x4 -50*x5,
            x5 -50*x6,
            x6 -50*x7,
            x7 - 50*x0,
            -20 * x7
            - 170 * x6
            - 800 * x5
            - 2273 * x4
            - 3980 * x3
            - 4180 * x2
            - 2400 * x1
            - 576 * x0,
        ]

    def f_smt(self, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return [
            x1 -50*x2,
            x2 -50*x3,
            x3 -50*x4,
            x4 -50*x5,
            x5 -50*x6,
            x6 -50*x7,
            x7 -50*x0,
            -20 * x7
            - 170 * x6
            - 800 * x5
            - 2273 * x4
            - 3980 * x3
            - 4180 * x2
            - 2400 * x1
            - 576 * x0,
        ]
        
class HighOrd6B(control.DynamicalModel):
    n_vars = 6

    def f_torch(self, v):
        x0, x1, x2, x3, x4, x5 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return [
            x1-100*x2,
            x2,
            x3-100*x4,
            x4,
            x5-100*x0,
            -800 * x5 - 2273 * x4 - 3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0,
        ]

    def f_smt(self, v):
        x0, x1, x2, x3, x4, x5 = v
        return [
            x1-100*x2,
            x2,
            x3-100*x4,
            x4,
            x5-100*x0,
            -800 * x5 - 2273 * x4 - 3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0,
        ]


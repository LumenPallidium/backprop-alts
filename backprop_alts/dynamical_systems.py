import torch
from torch.nn import functional as F

def runge_kutta(f, initial_conditions, n_steps, dt = 10e-5):
    """Runge-Kutta 4th order integration of a dynamical system, in torch.
    
    Parameters
    ----------
    f : callable
        The dynamical system, a function of time and state i.e. f(t, x).
    initial_conditions : torch.Tensor
        The initial conditions of the dynamical system.
    n_steps : int
        The number of steps to integrate for.
    dt : float
        The time step.
    """

    y0 = initial_conditions.to(torch.float64)
    values = [y0.unsqueeze(0)]

    t_i = 0

    for i in range(n_steps):
        t_next = t_i + dt
        y_i = values[i][0]
        dtd2 = 0.5 * dt
        f1 = f(t_i, y_i)
        f2 = f(t_i + dtd2, y_i + dtd2 * f1)
        f3 = f(t_i + dtd2, y_i + dtd2 * f2)
        f4 = f(t_next, y_i + dt * f3)
        dy = 1/6 * dt * (f1 + 2 * (f2 + f3) +f4)
        y_next = y_i + dy
        y_next = y_next
        t_i += dt

        values.append(y_next.unsqueeze(0))

    return torch.cat(values, dim = 0)


class DynamicalSystemSampler:
    def __init__(self, dynamical_system, n_steps, dt=10e-5, dim = 4):
        self.dynamical_system = dynamical_system
        self.n_steps = n_steps
        self.dt = dt
        self.dim = dim

    def sample(self):
        initial_conditions = torch.randn(self.dim)
        return runge_kutta(self.dynamical_system, initial_conditions, self.n_steps, self.dt)

class Lorenz4D(torch.nn.Module):
    """Lorenz system with 4D state space, see 
    https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-017-1280-5"""
    def __init__(self, a = 5 , b = 20, c = 1, d = 0.1, e = 0.1, f = 20.6, g = 1):
        super().__init__()
        self.a, self.b, self.c, self.d, self.e, self.f, self.g = a, b, c, d, e, f, g

    def forward(self, t, x):
        x, y, z, w = x
        dxdt = self.a * (y - x) - self.f * w
        dydt = x * z - self.g * y
        dzdt = self.b - x * y - self.c * z
        dwdt = self.d * y - self.e * w
        return torch.tensor([dxdt, dydt, dzdt, dwdt], dtype=torch.float64)

if __name__ == "__main__":
    # test the dynamical system sampler
    import plotly.express as px
    import pandas as pd

    # test the dynamical system sampler with two trajectories
    lorenz = Lorenz4D()
    sampler = DynamicalSystemSampler(lorenz, 10000, 10e-3, 4)
    traj1 = sampler.sample()
    traj2 = sampler.sample()

    # plot the result
    df1 = pd.DataFrame(traj1, columns = ["x", "y", "z", "w"])
    df2 = pd.DataFrame(traj2, columns = ["x", "y", "z", "w"])

    df1["sample"] = 1
    df2["sample"] = 2

    df = pd.concat([df1, df2], axis = 0)

    fig = px.line_3d(df, x = "x", y = "y", z = "z", color = "sample")


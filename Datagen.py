from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'red'
plt.rcParams['axes.linewidth'] = 2

def prbs(cf):
    cf = cf +  np.random.normal(0,5)
    return cf

def Reactor(t,x,*args):
    Qf, V, cf, k1, k2, k_2 = args
    ca, cb, cc = x
    dcadt = Qf*(cf - ca)/V - k1*ca
    dcbdt = -Qf*cb/V + k1*ca - 3*(k2*cb**2 - k_2*cc)
    dccdt = -Qf*cc/V + k2*cb**2 - k_2*cc
    return dcadt, dcbdt, dccdt

t_start = 0
t_end = 200
num_step_ups = 5
t_jumps = t_end/num_step_ups
A,B,C, ta, cfa = [],[],[],[],[]
initial_conditions = [0.3, 0.2, 0.1]
np.random.seed(100)
parameters = [[0.6, 15, 0.3, 0.2, 0.5, 0.1]]
parameters = [list(inner_list) for inner_list in parameters for _ in range(5)]
for i in range(len(parameters)-1):
    parameters[i+1][2] = np.random.uniform(0,1.5)
for i, params in enumerate(parameters):
    t_start = t_jumps * i
    t_end = t_jumps * (i + 1)
    t = np.linspace(t_start, t_end, 100)
    sol = solve_ivp(Reactor, [t_start, t_end], initial_conditions, method="RK45", t_eval=t, args=params)
    initial_conditions = sol.y[:, -1]
    A.append(sol.y[0])
    B.append(sol.y[1])
    C.append(sol.y[2])
    ta.append(t)

A = np.concatenate(A)
B = np.concatenate(B)
C = np.concatenate(C)
A = A + np.random.normal(0,0.01,len(A))
B = B + np.random.normal(0,0.01,len(B))
C = C + np.random.normal(0,0.01,len(C))
t = np.concatenate(ta)

for i in range(len(parameters)):
    cfa.append([parameters[i][2]]*100)
cfa = np.concatenate(cfa)
fig, axs = plt.subplots(4, 1, figsize=(9, 10))
axs[0].plot(t, A, 'b')
axs[0].set_ylabel(r"$C_A$")
axs[1].plot(t, B, 'b')
axs[1].set_ylabel(r"$C_B$")
axs[2].plot(t, C, 'b')
axs[2].set_ylabel(r"$C_C$")
axs[3].plot(t, cfa, 'b')
axs[3].set_ylabel(r"$C_{Af}$")
axs[3].set_xlabel("Time (min)")
plt.tight_layout()
plt.savefig("Datagen.pdf")

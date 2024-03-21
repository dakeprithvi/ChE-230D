from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import jax
import jax.numpy as jnp
from jax import grad, jit, lax
from jax.experimental.ode import odeint
from jax.example_libraries.optimizers import adam
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 2
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


def neural_net(params, x, kp):
    xz = x
    Qf, V, cf = kp
    for W, b in params:
        xz = jnp.tanh(jnp.dot(W, xz) + b)
    dcadt = Qf*(cf - x[0])/V - xz[0]
    dcbdt = -Qf*(x[1])/V + xz[0] - xz[1]
    return jnp.array([dcadt, dcbdt])

def ode_nn(params, x, cf, kp, z):
    xnew = jnp.concatenate([x, jnp.array([cf]), z])
    return neural_net(params, xnew, kp)

def update_z(z, new_x):
    return jnp.roll(z, -2, axis=-1).at[-2:].set(new_x)

@jit
def RK4_ode(y0, z0, t, params, kp):
    dt = t[1] - t[0]
    def body_fun(carry, ts_i):
        y_prev, z, i = carry
        Qf, V, cf = kp
        cfi = cf[i]
        current_kp = [Qf, V, cfi]
        new_x_half = y_prev + (dt / 2) * ode_nn(params, y_prev, cfi, current_kp, z)
        z_updated_half = update_z(z, new_x_half)
        k1 = ode_nn(params, y_prev,cfi, current_kp, z)
        k2 = ode_nn(params, y_prev + 0.5 * dt * k1,cfi, current_kp, z_updated_half)
        k3 = ode_nn(params, y_prev + 0.5 * dt * k2,cfi,  current_kp, z_updated_half)
        new_x_full = y_prev + dt * k3
        z_updated_full = update_z(z, new_x_full)
        k4 = ode_nn(params, y_prev + dt * k3,cfi, current_kp, z_updated_full)
        y_next = y_prev + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        z_next = update_z(z, y_next)
        return (y_next, z_next, i + 1), y_next
    _, y = lax.scan(body_fun, (y0, z0, 0), jnp.zeros(len(t) - 1))
    return jnp.vstack([y0[None, :], y])


def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return jnp.maximum(q * e, (q - 1) * e)

@jit
def loss_fn(params, x0, z0, t_span, true_solution, kp, q):
    learned_solution = RK4_ode(x0,z0, t_span, params, kp)
    return jnp.sum(quantile_loss(q, true_solution, learned_solution))
    
num_steps = 100*5
t_span_true = jnp.linspace(0., 200, num_steps)
x0_true = jnp.array([0.3, 0.2])
z0_true = jnp.array([0.,0.])
true_solution = [A,B]
true_solution_j = jnp.array(true_solution)
true_solution_jax = true_solution_j.T
kp = [0.6, 15, cfa]
layer_sizes = [3 + z0_true.size, 8, 8, 2]

def init_network_params(layer_sizes, rng_key):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_key, b_key = jax.random.split(rng_key)
        W = 1e-2 * jax.random.normal(W_key, (n_out, n_in))
        b = 1e-2 * jax.random.normal(b_key, (n_out,))
        params.append((W, b))
    return params

rng = jax.random.PRNGKey(0)
params = init_network_params(layer_sizes, rng)

grad_loss = grad(loss_fn)
opt_init, opt_update, get_params = adam(1e-3)
opt_state = opt_init(params)
loss = []
for i in range(2000):
    grads = grad_loss(get_params(opt_state), x0_true, z0_true, t_span_true, true_solution_jax, kp, 0.025)
    opt_state = opt_update(i, grads, opt_state)
    if i % 100 == 0:
        loss_value = loss_fn(get_params(opt_state), x0_true, z0_true,  t_span_true, true_solution_jax, kp, 0.025)
        print(f"Iteration {i}, Loss: {loss_value:.4f}")
        loss.append(loss_value)

params_opt1 = get_params(opt_state)
print("Optimized parameters:", params_opt1)


grad_loss = grad(loss_fn)
opt_init, opt_update, get_params = adam(1e-3)
opt_state = opt_init(params)
loss = []
for i in range(2000):
    grads = grad_loss(get_params(opt_state), x0_true, z0_true, t_span_true, true_solution_jax, kp, 0.975)
    opt_state = opt_update(i, grads, opt_state)
    if i % 100 == 0:
        loss_value = loss_fn(get_params(opt_state), x0_true, z0_true,  t_span_true, true_solution_jax, kp, 0.975)
        print(f"Iteration {i}, Loss: {loss_value:.4f}")
        loss.append(loss_value)
        
params_opt2 = get_params(opt_state)
print("Optimized parameters:", params_opt2)

def neural_net(params, x, kp):
    xz = x
    Qf, V, cf = kp
    for W, b in params:
        xz = jnp.tanh(jnp.dot(W, xz) + b)
    dcadt = Qf*(cf - x[0])/V - xz[0]
    dcbdt = -Qf*(x[1])/V + xz[0] - xz[1]
    return jnp.array([dcadt, dcbdt])

def ode_nn(params, x, cf, kp, z):
    xnew = jnp.concatenate([x, jnp.array([cf]), z])
    return neural_net(params, xnew, kp)

def update_z(z, new_x):
    return jnp.roll(z, -2, axis=-1).at[-2:].set(new_x)

@jit
def RK4_ode(y0, z0, t, params, kp):
    dt = t[1] - t[0]
    def body_fun(carry, ts_i):
        y_prev, z, i = carry
        Qf, V, cf = kp
        cfi = cf[i]
        current_kp = [Qf, V, cfi]
        new_x_half = y_prev + (dt / 2) * ode_nn(params, y_prev, cfi, current_kp, z)
        z_updated_half = update_z(z, new_x_half)
        k1 = ode_nn(params, y_prev,cfi, current_kp, z)
        k2 = ode_nn(params, y_prev + 0.5 * dt * k1,cfi, current_kp, z_updated_half)
        k3 = ode_nn(params, y_prev + 0.5 * dt * k2,cfi,  current_kp, z_updated_half)
        new_x_full = y_prev + dt * k3
        z_updated_full = update_z(z, new_x_full)
        k4 = ode_nn(params, y_prev + dt * k3,cfi, current_kp, z_updated_full)
        y_next = y_prev + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        z_next = update_z(z, y_next)
        return (y_next, z_next, i + 1), y_next
    _, y = lax.scan(body_fun, (y0, z0, 0), jnp.zeros(len(t) - 1))
    return jnp.vstack([y0[None, :], y])


@jit
def loss_fn(params, x0,z0, t_span, true_solution, kp):
    learned_solution = RK4_ode(x0,z0, t_span, params, kp)
    return jnp.sum((learned_solution - true_solution) ** 2)

num_steps = 100*5
t_span_true = jnp.linspace(0., 200, num_steps)
x0_true = jnp.array([0.3, 0.2])
z0_true = jnp.array([0.,0.])
true_solution = [A,B]
true_solution_j = jnp.array(true_solution)
true_solution_jax = true_solution_j.T
kp = [0.6, 15, cfa]
layer_sizes = [3 + z0_true.size, 8, 8, 2]

def init_network_params(layer_sizes, rng_key):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_key, b_key = jax.random.split(rng_key)
        W = 1e-2 * jax.random.normal(W_key, (n_out, n_in))
        b = 1e-2 * jax.random.normal(b_key, (n_out,))
        params.append((W, b))
    return params

rng = jax.random.PRNGKey(0)
params = init_network_params(layer_sizes, rng)

grad_loss = grad(loss_fn)
opt_init, opt_update, get_params = adam(1e-3)
opt_state = opt_init(params)
loss = []
for i in range(2000):
    grads = grad_loss(get_params(opt_state), x0_true, z0_true, t_span_true, true_solution_jax, kp)
    opt_state = opt_update(i, grads, opt_state)
    if i % 100 == 0:
        loss_value = loss_fn(get_params(opt_state), x0_true, z0_true,  t_span_true, true_solution_jax, kp)
        print(f"Iteration {i}, Loss: {loss_value:.4f}")
        loss.append(loss_value)

params_opt = get_params(opt_state)
print("Optimized parameters:", params_opt)

t_start = 0
t_end = 200
num_step_ups = 5
t_jumps = t_end/num_step_ups
A,B,C, ta, cfa = [],[],[],[],[]
initial_conditions = [0.3, 0.2, 0.1]
np.random.seed(500)
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


pdf = PdfPages('Uncertainty.pdf')
true_solution = [A,B]
true_solution_j = jnp.array(true_solution)
true_solution_jax = true_solution_j.T
kp = [0.6, 15, cfa]
x0_true = jnp.array([0.3,0.2])
fig, axs = plt.subplots(3, 1, figsize=(9, 10)) # 2 rows, 1 column
fitted_solution = RK4_ode(x0_true, z0_true, t_span_true, params_opt, kp)
f1 = RK4_ode(x0_true, z0_true, t_span_true, params_opt1, kp)
f2 = RK4_ode(x0_true, z0_true, t_span_true, params_opt2, kp)
# Plot A_true and A on the first subplot
axs[0].plot(np.linspace(0,200,len(true_solution_j[0])),true_solution_j[0], 'r', label="Plant")
axs[0].plot(np.linspace(0,200,len(fitted_solution.T[0])),fitted_solution.T[0],'k',label="Struc")
axs[0].fill_between(np.linspace(0,200,len(f1.T[0])), f1.T[0], f2.T[0], color='gray', alpha=0.5)
axs[0].set_ylabel(r"$c_A$")
axs[0].legend()

# Plot B_true (assuming this is the same as the true B) and B on the second subplot
axs[1].plot(np.linspace(0,200,len(true_solution_j[1])), true_solution_j[1], 'g', label="Plant")
axs[1].plot(np.linspace(0,200,len(fitted_solution.T[1])), fitted_solution.T[1],'k',label="Struc")
axs[1].fill_between(np.linspace(0,200,len(f1.T[1])), f1.T[1], f2.T[1], color='gray', alpha=0.5)
axs[1].set_ylabel(r"$c_B$")
axs[1].legend()

axs[2].plot(np.linspace(0,200,len(cfa)), cfa, 'b', label = r"$c_{Af}$")
axs[2].set_xlabel("Time (min)")
axs[2].set_ylabel(r"$c_{Af}$")


#axs[2].plot(C, 'k', label = r"$c_{Af}$")
#axs[2].set_xlabel("Time (min)")
#axs[2].set_ylabel(r"$c_{C}$")
plt.tight_layout()

pdf.savefig(fig) 
plt.close(fig)

pdf.close()

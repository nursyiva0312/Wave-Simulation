#simulation
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
def app():
    st.title("2D Shallow Water Equations Simulation")

    # ==================================================================================
    # ================================ Parameter stuff =================================
    # ==================================================================================
    # --------------- Physical parameters ---------------

    L_x = st.sidebar.slider("Length of domain in x-direction (L_x)", value=1E+6)
    L_y = st.sidebar.slider("Length of domain in y-direction (L_y)", value=1E+6)
    g = st.sidebar.number_input("Acceleration of gravity (g)", value=9.81)
    H = st.sidebar.number_input("Depth of fluid (H)", value=100)
    f_0 = 0
    beta = st.sidebar.number_input("Gradient of coriolis parameter (beta)", value=2E-11)
    rho_0 = st.sidebar.number_input("Density of fluid (rho_0)", value=1024.0)
    tau_0 = st.sidebar.number_input("Amplitude of wind stress (tau_0)", value=0.1)
    use_coriolis = st.sidebar.checkbox("Use Coriolis force", value=True)
    use_friction = st.sidebar.checkbox("Use bottom friction", value=False)
    use_wind = st.sidebar.checkbox("Use wind stress", value=False)
    use_beta = st.sidebar.checkbox("Use variation in Coriolis (beta)", value=True)
    use_source = st.sidebar.checkbox("Use mass source into the domain", value=False)
    use_sink = st.sidebar.checkbox("Use mass sink out of the domain", value=False)

    # --------------- Computational parameters ---------------
    N_x = st.sidebar.number_input("Number of grid points in x-direction (N_x)", value=150, step=1)
    N_y = st.sidebar.number_input("Number of grid points in y-direction (N_y)", value=150, step=1)
    max_time_step = st.sidebar.number_input("Total number of time steps in simulation", value=5000, step=1)

    # Derived parameters
    dx = L_x / (N_x - 1)
    dy = L_y / (N_y - 1)
    dt = 0.1 * min(dx, dy) / np.sqrt(g * H)
    x = np.linspace(-L_x / 2, L_x / 2, N_x)
    y = np.linspace(-L_y / 2, L_y / 2, N_y)
    X, Y = np.meshgrid(x, y)
    X = np.transpose(X)
    Y = np.transpose(Y)

    # Display parameters
    st.write("dx: {:.2f} km".format(dx / 1000))
    st.write("dy: {:.2f} km".format(dy / 1000))
    st.write("dt: {:.2f} s".format(dt))

    # Allocate arrays and initial conditions
    u_n = np.zeros((N_x, N_y))
    u_np1 = np.zeros((N_x, N_y))
    v_n = np.zeros((N_x, N_y))
    v_np1 = np.zeros((N_x, N_y))
    eta_n = np.zeros((N_x, N_y))
    eta_np1 = np.zeros((N_x, N_y))
    h_e = np.zeros((N_x, N_y))
    h_w = np.zeros((N_x, N_y))
    h_n = np.zeros((N_x, N_y))
    h_s = np.zeros((N_x, N_y))
    uhwe = np.zeros((N_x, N_y))
    vhns = np.zeros((N_x, N_y))

    # Initial conditions
    eta_n = np.exp(-((X - L_x / 2.7) ** 2 / (2 * (0.05E+6) ** 2) + (Y - L_y / 4) ** 2 / (2 * (0.05E+6) ** 2)))

    # Simulation
    eta_list = []
    u_list = []
    v_list = []
    time_step = 1

    if st.button("Run Simulation"):
        progress_bar = st.progress(0)
        t_0 = time.perf_counter()

        while time_step < max_time_step:
            u_np1[:-1, :] = u_n[:-1, :] - g * dt / dx * (eta_n[1:, :] - eta_n[:-1, :])
            v_np1[:, :-1] = v_n[:, :-1] - g * dt / dy * (eta_n[:, 1:] - eta_n[:, :-1])

            if use_friction:
                kappa_0 = 1 / (5 * 24 * 3600)
                kappa = np.ones((N_x, N_y)) * kappa_0
                u_np1[:-1, :] -= dt * kappa[:-1, :] * u_n[:-1, :]
                v_np1[:, :-1] -= dt * kappa[:, :-1] * v_n[:, :-1]

            if use_wind:
                tau_x = -tau_0 * np.cos(np.pi * y / L_y)
                tau_y = np.zeros_like(tau_x)
                
                u_np1[:-1, :] += dt * tau_x / (rho_0 * H)
                #v_np1[:, :-1] += dt * tau_y / (rho_0 * H)

            if use_coriolis:
                f = f_0 + beta * y if use_beta else f_0
                alpha = dt * f
                beta_c = alpha ** 2 / 4
                
                u_np1[:, :] = (u_np1[:, :] - beta_c * u_n[:, :] + alpha * v_n[:, :]) / (1 + beta_c)
                v_np1[:, :] = (v_np1[:, :] - beta_c * v_n[:, :] - alpha * u_n[:, :]) / (1 + beta_c)

            v_np1[:, -1] = 0.0
            u_np1[-1, :] = 0.0

            h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
            h_e[-1, :] = eta_n[-1, :] + H
            h_w[0, :] = eta_n[0, :] + H
            h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
            h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
            h_n[:, -1] = eta_n[:, -1] + H
            h_s[:, 0] = eta_n[:, 0] + H
            h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

            uhwe[0, :] = u_np1[0, :] * h_e[0, :]
            uhwe[1:, :] = u_np1[1:, :] * h_e[1:, :] - u_np1[:-1, :] * h_w[1:, :]
            vhns[:, 0] = v_np1[:, 0] * h_n[:, 0]
            vhns[:, 1:] = v_np1[:, 1:] * h_n[:, 1:] - v_np1[:, :-1] * h_s[:, 1:]

            eta_np1[:, :] = eta_n[:, :] - dt * (uhwe[:, :] / dx + vhns[:, :] / dy)

            if use_source:
                sigma = np.zeros((N_x, N_y))
                sigma = 0.0001 * np.exp(-((X - L_x / 2) ** 2 / (2 * (1E+5) ** 2) + (Y - L_y / 2) ** 2 / (2 * (1E+5) ** 2)))
                eta_np1[:, :] += dt * sigma

            if use_sink:
                w = np.ones((N_x, N_y)) * sigma.sum() / (N_x * N_y)
                eta_np1[:, :] -= dt * w

            u_n = np.copy(u_np1)
            v_n = np.copy(v_np1)
            eta_n = np.copy(eta_np1)

            time_step += 1

            if time_step % 100 == 0:
                eta_list.append(eta_n.copy())
                u_list.append(u_n.copy())
                v_list.append(v_n.copy())
                progress_bar.progress(time_step / max_time_step)

        st.success("Simulation complete!")
        st.write("Execution time: {:.2f} seconds".format(time.perf_counter() - t_0))


        with st.spinner('Waiting for animation.. Please wait..'):
            fig = plt.figure(figsize = (6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, eta_list[-1], cmap='ocean')
            plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
            plt.ylabel("y [km]", fontname = "serif", fontsize = 16)


            # Create animation
            def update(frame):
                ax.clear()
                ax.plot_surface(X, Y, eta_list[frame],cmap='ocean')
                ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
                frame*20/3600), fontname = "serif", fontsize = 14, y=1.04)
                ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
                ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
                ax.set_xlim(X.min(), X.max())
                ax.set_ylim(Y.min(), Y.max())
                ax.set_zlim(np.min(eta_list), np.max(eta_list))

            ani = animation.FuncAnimation(fig, update, frames=len(eta_list), repeat=False)
            ani.save("shallow_water_simulation.gif", writer='pillow')



            fig, ax = plt.subplots(figsize = (5, 5), facecolor = "white")
            plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 12)
            plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
            plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
            q_int = 3
            Q = ax.quiver(X[::q_int, ::q_int]/1000.0, Y[::q_int, ::q_int]/1000.0, u_list[0][::q_int,::q_int], v_list[0][::q_int,::q_int],
                scale=0.2, scale_units='inches')

            # Update function for quiver animation.
            def update_quiver(num):
                u = u_list[num]
                v = v_list[num]
                ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
                    num*20*dt/3600), fontname = "serif", fontsize = 19)
                Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
                return Q,

            ani = animation.FuncAnimation(fig, update_quiver,
                frames = len(u_list), interval = 10, blit = False)
            fig.tight_layout()
            ani.save('velocity.gif', writer='pillow')

            col1, col2 = st.columns(2) 
        

            with open("shallow_water_simulation.gif", "rb") as file:
                col1.image(file.read())

            with open("velocity.gif", "rb") as file:
                col2.image(file.read())
app()

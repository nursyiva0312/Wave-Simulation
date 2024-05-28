import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the PINN model architecture
def create_shallow_water_pinn():
    inputs = tf.keras.layers.Input(shape=(3,))
    x, y, t = tf.unstack(inputs, axis=1)
    input_tensor = tf.stack([x, y, t], axis=1)
    
    u = tf.keras.layers.Dense(50, activation='tanh')(input_tensor)
    u = tf.keras.layers.Dense(1)(u)
    
    v = tf.keras.layers.Dense(50, activation='tanh')(input_tensor)
    v = tf.keras.layers.Dense(1)(v)
    
    eta = tf.keras.layers.Dense(50, activation='tanh')(input_tensor)
    eta = tf.keras.layers.Dense(1)(eta)
    
    outputs = tf.stack([u, v, eta], axis=1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

@tf.function
def compute_loss(inputs, targets, pinn_model, g, tau_0, rho_0, H, kappa, f_0, beta, use_friction, use_wind, use_coriolis, use_beta, use_source, use_sink):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        predictions = pinn_model(inputs)
        predictions = tf.cast(predictions, tf.float64)
        u, v, eta = tf.unstack(predictions, axis=1)

        du_dx = tape.gradient(u, inputs)[:, 0]
        du_dy = tape.gradient(u, inputs)[:, 1]
        dv_dx = tape.gradient(v, inputs)[:, 0]
        dv_dy = tape.gradient(v, inputs)[:, 1]
        d_eta_dx = tape.gradient(eta, inputs)[:, 0]
        d_eta_dy = tape.gradient(eta, inputs)[:, 1]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        predictions = pinn_model(inputs)
        predictions = tf.cast(predictions, tf.float64)
        u, v, eta = tf.unstack(predictions, axis=1)
        du_dt = tape.gradient(u, inputs)[:, 2]
        dv_dt = tape.gradient(v, inputs)[:, 2]
        d_eta_dt = tape.gradient(eta, inputs)[:, 2]

    du_dx = tf.zeros_like(inputs[:, 0]) if du_dx is None else du_dx
    du_dy = tf.zeros_like(inputs[:, 1]) if du_dy is None else du_dy
    dv_dx = tf.zeros_like(inputs[:, 0]) if dv_dx is None else dv_dx
    dv_dy = tf.zeros_like(inputs[:, 1]) if dv_dy is None else dv_dy
    d_eta_dx = tf.zeros_like(inputs[:, 0]) if d_eta_dx is None else d_eta_dx
    d_eta_dy = tf.zeros_like(inputs[:, 1]) if d_eta_dy is None else d_eta_dy
    du_dt = tf.zeros_like(inputs[:, 2]) if du_dt is None else du_dt
    dv_dt = tf.zeros_like(inputs[:, 2]) if dv_dt is None else dv_dt
    d_eta_dt = tf.zeros_like(inputs[:, 2]) if d_eta_dt is None else d_eta_dt

    u_true, v_true, eta_true = tf.unstack(targets, axis=1)

    H = tf.convert_to_tensor(H, dtype=tf.float64)
    g = tf.convert_to_tensor(g, dtype=tf.float64)
    tau_0 = tf.convert_to_tensor(tau_0, dtype=tf.float64)
    rho_0 = tf.convert_to_tensor(rho_0, dtype=tf.float64)
    kappa = tf.convert_to_tensor(kappa, dtype=tf.float64)
    f_0 = tf.convert_to_tensor(f_0, dtype=tf.float64)
    beta = tf.convert_to_tensor(beta, dtype=tf.float64)

    x, y, t = tf.unstack(inputs, axis=1)
    f = f_0 + beta * y if use_beta else f_0 * tf.ones_like(y)

    if use_friction:
        kappa_0 = 1 / (5 * 24 * 3600)
        kappa = tf.ones_like(inputs[:, 0]) * kappa_0

    if use_wind:
        tau_x = -tau_0 * tf.cos(np.pi * y / L_y)
        tau_y = tf.zeros_like(tau_x)
    else:
        tau_x = tf.zeros_like(y)
        tau_y = tf.zeros_like(y)

    residual_u = du_dt - f*v + g * d_eta_dx - tau_x / (rho_0 * H) - kappa * u
    residual_v = dv_dt + f*u + g * d_eta_dy - tau_y / (rho_0 * H) - kappa * v
    residual_eta = d_eta_dt + (du_dx + dv_dy) * (eta + H)

    loss = tf.reduce_mean(tf.square(residual_u) + tf.square(residual_v) + tf.square(residual_eta))

    return loss

def train_pinn(inputs, targets, pinn_model, optimizer, num_epochs, g, tau_0, rho_0, H, kappa, f_0, beta, use_friction, use_wind, use_coriolis, use_beta, use_source, use_sink):
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(inputs, targets, pinn_model, g, tau_0, rho_0, H, kappa, f_0, beta, use_friction, use_wind, use_coriolis, use_beta, use_source, use_sink)
        gradients = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))
        if epoch % 100 == 0:
            st.write(f'Epoch {epoch}, Loss: {loss.numpy()}')
    pinn_model.save('pinn_model.h5')

def plot_solution_at_time(pinn_model, time, L_x, L_y):
    x_vals = np.linspace(-L_x/2, L_x/2, 100)
    y_vals = np.linspace(-L_y/2, L_y/2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    T = np.full_like(X, time)

    inputs = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
    predictions = pinn_model(inputs)
    u, v, eta = tf.unstack(predictions, axis=1)

    U = u.numpy().reshape(X.shape)
    V = v.numpy().reshape(X.shape)
    ETA = eta.numpy().reshape(X.shape)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    c1 = ax[0].contourf(X, Y, U, levels=50, cmap='viridis')
    c2 = ax[1].contourf(X, Y, V, levels=50, cmap='viridis')
    c3 = ax[2].contourf(X, Y, ETA, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax[0])
    fig.colorbar(c2, ax=ax[1])
    fig.colorbar(c3, ax=ax[2])
    ax[0].set_title('u velocity')
    ax[1].set_title('v velocity')
    ax[2].set_title('Surface elevation eta')
    plt.tight_layout()
    st.pyplot(fig)

def app():
    st.title("Shallow Water Model PINN-ECMBO Training")

    # Define columns
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Parameters"):
            L_x = st.number_input('L_x', value=1E+6)
            L_y = st.number_input('L_y', value=1E+6)
            g = st.number_input('g', value=9.81)
            H = st.number_input('H', value=100)
            f_0 = st.number_input('f_0', value=1E-4)
            beta = st.number_input('beta', value=2E-11)
            rho_0 = st.number_input('rho_0', value=1024.0)
            tau_0 = st.number_input('tau_0', value=0.1)

    with col2:
        with st.expander("Options"):
            use_coriolis = st.checkbox('Coriolis', value=True)
            use_friction = st.checkbox('Friction', value=False)
            use_wind = st.checkbox('Wind', value=False)
            use_beta = st.checkbox('Beta', value=True)
            use_source = st.checkbox('Source', value=False)
            use_sink = st.checkbox('Sink', value=False)
            num_epochs = st.number_input('Number of Epochs', value=500, step=100)

    if use_friction:
        kappa = 1 / (5 * 24 * 3600)
    else:
        kappa = 0.0

    if st.button('Start Training'):
        num_samples = 1000
        x_train = np.random.uniform(low=-L_x/2, high=L_x/2, size=num_samples)
        y_train = np.random.uniform(low=-L_y/2, high=L_y/2, size=num_samples)
        t_train = np.random.uniform(low=0, high=1E+6, size=num_samples)
        inputs = np.stack([x_train, y_train, t_train], axis=1)

        # Generate random targets (replace this with your own target generation process)
        u_train = np.random.uniform(low=-1, high=1, size=num_samples)
        v_train = np.random.uniform(low=-1, high=1, size=num_samples)
        eta_train = np.random.uniform(low=-1, high=1, size=num_samples)
        targets = np.stack([u_train, v_train, eta_train], axis=1)

        pinn_model = create_shallow_water_pinn()
        optimizer = tf.keras.optimizers.Adam()

        train_pinn(inputs, targets, pinn_model, optimizer, num_epochs, g, tau_0, rho_0, H, kappa, f_0, beta, use_friction, use_wind, use_coriolis, use_beta, use_source, use_sink)

        st.write("Training complete!")


# Run the app
app()

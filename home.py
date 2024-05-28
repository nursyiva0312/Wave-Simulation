import streamlit as st
from PIL import Image

def app():
    st.markdown(
        """
        <style>
        /* General settings */
        body {
            background-color: #f4f4f9;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }

        /* Title and headers */
        h1 {
            color: #2c3e50;
        }

        h2, h3, h4 {
            color: #34495e;
        }

        /* Sidebar settings */
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: #ffffff;
        }

        /* Radio button settings */
        .sidebar .radio label {
            color: #ecf0f1;
        }

        /* Simulation section styling */
        .simulation-section {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        /* Add some padding to markdown content */
        .markdown-text-container {
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("2D Shallow Water Equations")
    st.image("Shallow_water_waves.gif", caption="Waves", width=500)
    #st.image("Shallow_water_model_diagram.png", caption="Waves", width=500)
    st.markdown('<div class="markdown-text-container">', unsafe_allow_html=True)
    st.markdown("""
    The section describes a numerical model for solving the 2D shallow water equations using Physics Informed Neural Networks with Empowered Cat and Mouse based Optimizer (PINN-ECMBO). 
    The shallow water equations are a set of partial differential equations that govern the behavior of water waves in shallow regions, 
    where the depth of the water is much smaller compared to the wavelength of the waves. 
    
    The equations describe the conservation of mass and momentum in the fluid, and they take into account various factors such as gravitational forces, 
    Coriolis forces due to the rotation of the Earth, wind stress, frictional effects, and sources/sinks of mass.
    Here are the equations of motion for the 2D shallow water equations:

    **Momentum Equations:**

    $$
    \\frac{\\partial u}{\\partial t} - fv = -g \\frac{\\partial \\eta}{\\partial x} + \\frac{\\tau_x}{\\rho_0 H} - \\kappa u
    $$

    $$
    \\frac{\\partial v}{\\partial t} + fu = -g \\frac{\\partial \\eta}{\\partial y} + \\frac{\\tau_y}{\\rho_0 H} - \\kappa v
    $$

    **Continuity Equation:**

    $$
    \\frac{\\partial \\eta}{\\partial t} + \\frac{\\partial}{\\partial x} \left((\\eta + H)u\\right) + \\frac{\\partial}{\\partial y} \left((\\eta + H)v\\right) = \\sigma - w
    $$

    where:
    - $u$ and $v$ are the horizontal velocity components,
    - $\\eta$ is the surface elevation,
    - $f$ is the Coriolis parameter,
    - $g$ is the acceleration due to gravity,
    - $\\tau_x$ and $\\tau_y$ are the wind stress components,
    - $\\rho_0$ is the density of the fluid,
    - $H$ is the resting depth of the fluid,
    - $\\kappa$ is the friction coefficient,
    - $\\sigma$ is the mass source term, and
    - $w$ is the mass sink term.
    """)

    st.markdown("""Here's an explanation of the key components and terms in the problem description:
    """)

    st.subheader("1. Momentum Equations:")
    st.markdown("""
    - The momentum equations describe the evolution of horizontal velocity components $u$ and $v$ over time. 
    They are influenced by Coriolis forces ($f$), gravitational forces ($g$), wind stress ($τ_x$ and $τ_y$), friction ($κ$), 
    and the spatial gradient of the surface elevation ($η$).
    - The Coriolis force $f$ depends on the Coriolis parameter $f_0$ and the meridional gradient of the Earth's angular velocity ($β$).
    - The momentum equations are discretized using a forward-in-time centered-in-space scheme.
    """)

    st.subheader("2. Continuity Equation:")
    st.markdown("""
    - The continuity equation describes the evolution of the surface elevation ($η$) over time. 
    It accounts for changes in $η$ due to advection by the horizontal velocity components $u$ and $v$, as well as sources/sinks of mass ($σ$ and $w$).
    - The continuity equation is discretized using a forward difference for the time derivative and an upwind scheme for the nonlinear terms.
    """)

    st.subheader("3. Stability Conditions:")
    st.markdown("""
    - The stability of the numerical model is ensured by satisfying the CFL (Courant-Friedrichs-Lewy) condition, 
    which requires that the time step $dt$ is smaller than or equal to the minimum of the grid spacings $dx$ and $dy$ 
    divided by the square root of the product of gravitational acceleration ($g$) and the resting depth of the fluid ($H$).
    $$
    \\Delta t \\leq \\frac{\\min(\\Delta x, \\Delta y)}{\\sqrt{gH}}
    $$

    and

    $$
    \\alpha \\ll 1 \\quad \\text{(if Coriolis is used)}
    $$
    """)

    st.markdown("""
    Overall, the numerical model described in the script provides a comprehensive framework for simulating the dynamics of shallow water waves, 
    accounting for various physical processes and ensuring numerical stability through appropriate discretization and parameter choices.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    app()
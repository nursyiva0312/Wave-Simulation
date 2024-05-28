import streamlit as st
from streamlit_option_menu import option_menu

import PINN_ECMBO, home, simulation

# Function to create the Streamlit menu
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Training Process PINN-ECMBO", "Prediction and Simulation"],  # required
                icons=["house", "search-heart", "search"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Training Process PINN-ECMBO", "Prediction and Simulation"],  # required
            icons=["house", "search-heart", "document"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 3. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Training Process PINN-ECMBO", "Prediction and Simulation"],  # required
            icons=["house", "search-heart", "document"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

# Get the selected menu item
selected = streamlit_menu(example=1)

# Handle the menu selection
if selected == "Home":
    home.app()

if selected == "Training Process PINN-ECMBO":
    PINN_ECMBO.app()

if selected == "Prediction and Simulation":
    simulation.app()

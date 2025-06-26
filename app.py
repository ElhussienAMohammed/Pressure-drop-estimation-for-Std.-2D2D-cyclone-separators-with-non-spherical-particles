import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Fix: define the class used in the original pickle
class SimpleEnsemble:
    def __init__(self, model1, model2, weight1=0.7, weight2=0.3):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return self.weight1 * pred1 + self.weight2 * pred2

# Now load the model
model = joblib.load("fold4_ensemble_model.pkl")
transformer = joblib.load("target_transformer.pkl")
selected_features = joblib.load("selected_features.pkl")


# Constants
AIR_DENSITY = 1.225
D_H = 66.7  # mm (fixed)
eps = 1e-10

# Title
st.title("🔬 Cyclone Pressure Drop Estimator")
st.markdown("Enter your particle properties to estimate cyclone pressure drop for spherical particles.")
st.markdown("Calculation with 99.92% accuracy.")
# Inputs
st.markdown("All calculations where estimated based on cyclone main diameter **Dc = 200 mm** ")
# --- Inputs ---
st.markdown("**1️⃣ Φ (sphericity):** Range `0.5 ≤ Φ ≤ 1`")
phi = st.number_input("Particle Shape Φ", min_value=0.5, max_value=1.0, value=0.5)

st.markdown("**2️⃣ ρₛ (particle density kg/m³):** Range `700 ≤ ρₛ ≤ 3320`")
rho_s = st.number_input("Particle Density ρₛ", min_value=700.0, max_value=3320.0, value=1500.0)

st.markdown("**3️⃣ dₚ (particle diameter μm):** Range `0.1 ≤ dₚ ≤ 10`")
d_p = st.number_input("Particle Diameter dₚ", min_value=0.1, max_value=10.0, value=1.0)

st.markdown("**4️⃣ αₚ (volume fraction):** Range `1e-6 ≤ αₚ ≤ 1e-3`")
alpha_p = st.number_input("Volume Fraction αₚ", min_value=1e-6, max_value=1e-4, format="%.6f", value=1e-5)


# Computed variables
X_input = phi
R = rho_s / AIR_DENSITY
H = d_p / (D_H * 1000)
alpha = alpha_p

# Create feature-engineered input
def engineer_features(X, H, R, alpha):
    df = pd.DataFrame({
        "x": [X], "H": [H], "R": [R], "alpha": [alpha]
    })
    df["log_H"] = np.log1p(df["H"] + eps)
    df["log_R"] = np.log1p(df["R"] + eps)
    df["log_alpha"] = np.log1p(df["alpha"] + eps)
    df["x_squared"] = df["x"] ** 2
    df["x_cubed"] = df["x"] ** 3
    df["x_fourth"] = df["x"] ** 4
    df["sqrt_x"] = np.sqrt(df["x"])
    df["inv_x"] = 1 / (df["x"] + eps)
    df["inv_x_squared"] = 1 / ((df["x"] + eps) ** 2)
    df["x_log_H"] = df["x"] * df["log_H"]
    df["x_log_R"] = df["x"] * df["log_R"]
    df["x_log_alpha"] = df["x"] * df["log_alpha"]
    df["log_H_log_R"] = df["log_H"] * df["log_R"]
    df["log_H_log_alpha"] = df["log_H"] * df["log_alpha"]
    df["log_R_log_alpha"] = df["log_R"] * df["log_alpha"]
    df["x_log_H_log_R"] = df["x"] * df["log_H"] * df["log_R"]
    df["x_log_H_log_alpha"] = df["x"] * df["log_H"] * df["log_alpha"]
    df["x_log_R_log_alpha"] = df["x"] * df["log_R"] * df["log_alpha"]
    df["log_H_log_R_log_alpha"] = df["log_H"] * df["log_R"] * df["log_alpha"]
    df["H_to_R"] = df["H"] / (df["R"] + eps)
    df["H_to_alpha"] = df["H"] / (df["alpha"] + eps)
    df["R_to_alpha"] = df["R"] / (df["alpha"] + eps)
    df["R_to_H"] = df["R"] / (df["H"] + eps)
    df["alpha_to_H"] = df["alpha"] / (df["H"] + eps)
    df["log_H_to_alpha"] = np.log1p(df["H_to_alpha"] + eps)
    df["log_R_to_alpha"] = np.log1p(df["R_to_alpha"] + eps)
    df["log_H_to_R"] = np.log1p(df["H_to_R"] + eps)
    df["log_R_to_H"] = np.log1p(df["R_to_H"] + eps)
    df["dist_from_x1"] = np.abs(df["x"] - 1.0)
    df["exp_dist"] = np.exp(-df["dist_from_x1"] * 5)
    df["exp_dist_2"] = np.exp(-df["dist_from_x1"] * 10)
    df["pressure_proxy"] = df["R"] * df["alpha"] / (df["H"] + eps)
    df["log_pressure_proxy"] = np.log1p(df["pressure_proxy"] + eps)
    df["boundary_transition"] = 1 / (1 + np.exp(-20 * (df["x"] - 0.9)))
    df["boundary_pressure"] = df["boundary_transition"] * df["pressure_proxy"]
    df["reynolds_proxy"] = df["R"] * df["H"] / (df["alpha"] + eps)
    df["boundary_reynolds"] = df["boundary_transition"] * df["reynolds_proxy"]
    return df[selected_features]

# Predict
input_data = engineer_features(X_input, H, R, alpha)
pred_trans = model.predict(input_data)
pred = transformer.inverse_transform(pred_trans.reshape(-1, 1)).ravel()[0]
if np.isclose(X_input, 1.0):
    pred = 1.0
# Display Result
st.markdown("## ✅ Your cyclone pressure drop equals y of the perfect spherical particle  :")
st.success(f"**y = {pred:.6f}**")

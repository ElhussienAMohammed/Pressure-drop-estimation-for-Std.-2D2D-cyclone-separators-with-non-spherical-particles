import streamlit as st
st.title("🔍 Cyclone Efficiency Estimator")

# Test joblib import
try:
    import joblib
    st.success("✅ joblib imported successfully.")
except Exception as e:
    st.error(f"❌ joblib import failed: {e}")

# Test model file existence
import os
for f in ["fold4_ensemble_model.pkl", "target_transformer.pkl", "selected_features.pkl"]:
    if not os.path.exists(f):
        st.error(f"❌ File missing: {f}")
    else:
        st.success(f"✅ File exists: {f}")

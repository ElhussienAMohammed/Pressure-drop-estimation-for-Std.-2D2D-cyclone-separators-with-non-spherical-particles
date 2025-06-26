# 🔬 Pressure Drop Estimation for STD-2D2D Cyclone Separators with Non-Spherical Particles

This repository hosts a Streamlit-powered web app and pretrained machine learning model for estimating **cyclone pressure drop efficiency (y)** for non-spherical particles.

## 📌 How It Works

The app takes four user inputs:
1. **Particle Shape Factor (Φ)**: 0.5 ≤ Φ ≤ 1.0  
2. **Particle Density (ρₛ)** in kg/m³: 700 ≤ ρₛ ≤ 3320  
3. **Particle Diameter (dₚ)** in μm: 0.1 ≤ dₚ ≤ 10  
4. **Volume Fraction (αₚ)**: 1e-5 ≤ αₚ ≤ 1e-3

These are transformed into physical and statistical features used by a robust **Ensemble ML model (KernelRidge + GradientBoosting)** trained on a well-engineered dataset.

When Φ = 1, the boundary condition y = 1 is enforced.

## 🌐 Try the Web App

🚀 [Launch the App on Streamlit](https://eit4uaakdvjb3qtgatbfr9.streamlit.app/)

## 📦 Files

- `app.py`: Streamlit UI and backend logic
- `fold4_ensemble_model.pkl`: Trained ensemble model (best fold R² = 0.9933)
- `target_transformer.pkl`: Transformer for target normalization
- `selected_features.pkl`: Feature subset used during modeling
- `requirements.txt`: Required Python packages

## 📜 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for more.

## 📖 Citation

This work is citable via Zenodo DOI (coming soon). Please check the next Zenodo release or badge.

---

📌 Developed by [Elhussien]  
📬 Questions? Contact [eng-elhussin.abuali@alexu.edu.eg ]

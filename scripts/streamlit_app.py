# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from matplotlib import cm

# -----------------------
# CONFIG / PATHS
# -----------------------
MODEL_PATH = r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\models\hyperthermia_temp_model.pkl"
TARGET_MIN = 41.0
TARGET_MAX = 45.0
THERMAL_INERTIA = 0.2   # how quickly the real temperature moves toward model-predicted value

# -----------------------
# UTILITIES
# -----------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

def build_input_df(SAR, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region):
    return pd.DataFrame([{
        "SAR": float(SAR),
        "E_field": float(E_field),
        "Frequency_Index": int(Frequency_Index),
        "Depth": float(Depth),
        "AMC_Presence": AMC_Presence,
        "Tissue_Region": Tissue_Region
    }])

def predict_temp(model, SAR, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region):
    df = build_input_df(SAR, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region)
    return float(model.predict(df)[0])

def compute_cem43_increment(temp, duration_seconds):
    R = 0.5 if temp >= 43.0 else 0.25
    return duration_seconds * (R ** (43.0 - temp))

# -----------------------
# SIMPLE Q-LEARNING AGENT (LIGHTWEIGHT)
# -----------------------
class SimpleSAR_QAgent:
    def __init__(self, sar_values, temp_bins, alpha=0.4, gamma=0.9, epsilon=0.2):
        self.sar_values = np.array(sar_values)
        self.state_bins = temp_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = len(temp_bins) - 1
        self.n_actions = len(sar_values)
        self.Q = np.zeros((self.n_states, self.n_actions))

    def state_from_temp(self, temp):
        s = np.digitize([temp], bins=self.state_bins)[0] - 1
        return int(max(0, min(self.n_states - 1, s)))

    def choose_action(self, temp):
        s = self.state_from_temp(temp)
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = int(np.argmax(self.Q[s]))
        return a, float(self.sar_values[a])

    def learn(self, s, a, reward, s_next):
        best_next = np.max(self.Q[s_next])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (reward + self.gamma * best_next)

    def policy_sar(self, temp):
        s = self.state_from_temp(temp)
        return float(self.sar_values[int(np.argmax(self.Q[s]))])

# Light RL trainer (keeps runtime short)
def rl_train_policy(model, base_inputs, sar_values, temp_bins, episodes=120, max_steps=30):
    agent = SimpleSAR_QAgent(sar_values=sar_values, temp_bins=temp_bins, alpha=0.35, gamma=0.9, epsilon=0.25)
    for ep in range(episodes):
        # start with small random perturbation around base predicted temp
        temp = max(30.0, predict_temp(model, **base_inputs) + np.random.randn() * 0.5)
        for _ in range(max_steps):
            s = agent.state_from_temp(temp)
            a_idx, sar_choice = agent.choose_action(temp)
            new_pred = predict_temp(model, sar_choice, base_inputs["E_field"], base_inputs["Frequency_Index"],
                                    base_inputs["Depth"], base_inputs["AMC_Presence"], base_inputs["Tissue_Region"])
            temp_next = temp + THERMAL_INERTIA * (new_pred - temp)
            reward = 0.0
            if TARGET_MIN <= temp_next <= TARGET_MAX:
                reward += 8.0
            reward -= abs(((TARGET_MIN + TARGET_MAX) / 2) - temp_next) * 0.15
            reward -= sar_choice * 0.03  # penalize high SAR
            s_next = agent.state_from_temp(temp_next)
            agent.learn(s, a_idx, reward, s_next)
            temp = temp_next
        agent.epsilon = max(0.01, agent.epsilon * 0.993)
    return agent

# -----------------------
# APP UI
# -----------------------
st.set_page_config(page_title="Hyperthermia Temperature Control", layout="wide")
st.title("Hyperthermia Temperature Control")
st.write("This system predicts temperature and automatically adjusts SAR to maintain **41–45°C** therapeutic range.")

# --- SIMPLE top UI (your screenshot style) ---
st.markdown("### Inputs (change these to update initial prediction)")
colA, colB = st.columns([2,1])
with colA:
    init_SAR = st.number_input("Initial SAR (W/kg)", min_value=0.0, max_value=20.0, value=7.0, step=0.1, key="init_sar")
    init_Efield = st.number_input("Electric Field (V/m)", min_value=0.0, max_value=500.0, value=150.0, step=1.0, key="init_ef")
    init_Freq = st.number_input("Frequency Index", min_value=1, max_value=10, value=3, step=1, key="init_freq")
    init_Depth = st.number_input("Depth (cm)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, key="init_depth")
with colB:
    init_AMC = st.selectbox("AMC Presence", ["Yes", "No"], key="init_amc")
    init_Tissue = st.selectbox("Tissue Region", ["Muscle", "Tumor", "Fat"], key="init_tissue")
    # Predict & Regulate button in top UI
    predict_and_regulate = st.button("Predict & Regulate")

# Load model (cached)
with st.spinner("Loading model..."):
    model = load_model()

# Live initial prediction (updates whenever inputs change)
initial_temp = predict_temp(model, init_SAR, init_Efield, init_Freq, init_Depth, init_AMC, init_Tissue)
st.markdown(f"**Initial Predicted Temperature:** **{initial_temp:.2f} °C**")
if initial_temp < TARGET_MIN:
    st.warning("Initial temperature below therapeutic range — consider increasing SAR/E-field.")
elif initial_temp > TARGET_MAX:
    st.error("Initial temperature above safe range — reduce SAR or exposure.")
else:
    st.success("Initial temperature is within therapeutic (41–45°C) range.")

st.markdown("---")

# --- Advanced sidebar controls ---
st.sidebar.header("Advanced Simulation & Control")
sim_duration = st.sidebar.slider("Simulation duration (seconds)", min_value=10, max_value=600, value=120, step=10)
time_step = st.sidebar.slider("Simulation update interval (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
auto_regulate = st.sidebar.checkbox("Use auto-regulation SAR loop (heuristic)", value=True)
use_rl = st.sidebar.checkbox("Use RL controller (policy after training)", value=False)
train_rl_now = st.sidebar.checkbox("Train RL policy now (lightweight)", value=False)
do_sensitivity = st.sidebar.checkbox("Run sensitivity analysis (OAT)", value=False)
sensitivity_delta = st.sidebar.number_input("Sensitivity perturbation (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

# placeholders for main area
col_main, col_side = st.columns([3,1])
chart_ph = col_main.empty()
metrics_ph = col_main.empty()
cem43_ph = col_main.empty()
controls_ph = col_side.empty()

# Precompute base_inputs for sensitivity / RL
base_inputs = {
    "SAR": init_SAR,
    "E_field": init_Efield,
    "Frequency_Index": init_Freq,
    "Depth": init_Depth,
    "AMC_Presence": init_AMC,
    "Tissue_Region": init_Tissue
}

# Run sensitivity if requested (non-blocking)
sensitivity_result = None
if do_sensitivity:
    with st.spinner("Running sensitivity analysis..."):
        base_df = build_input_df(**base_inputs)
        base_temp = float(model.predict(base_df)[0])
        impacts = {}
        features = list(base_inputs.keys())
        for feat in features:
            if feat in ["AMC_Presence", "Tissue_Region"]:
                if feat == "AMC_Presence":
                    alt = "No" if base_inputs[feat] == "Yes" else "Yes"
                    t_alt = float(model.predict(build_input_df(**{**base_inputs, feat: alt}))[0])
                    impacts[feat] = abs(t_alt - base_temp)
                else:
                    vals = [v for v in ["Muscle","Fat","Tumor"] if v != base_inputs[feat]]
                    diffs = []
                    for v in vals:
                        diffs.append(abs(float(model.predict(build_input_df(**{**base_inputs, feat: v}))[0]) - base_temp))
                    impacts[feat] = max(diffs)
            else:
                val = base_inputs[feat]
                delta = val * (sensitivity_delta / 100.0)
                max_diff = 0.0
                for sign in [+1, -1]:
                    newval = val + sign * delta
                    t_alt = float(model.predict(build_input_df(**{**base_inputs, feat: newval}))[0])
                    max_diff = max(max_diff, abs(t_alt - base_temp))
                impacts[feat] = max_diff
        sensitivity_result = (base_temp, impacts)

# RL training if requested
rl_agent = None
if train_rl_now:
    with st.spinner("Training lightweight RL policy..."):
        sar_vals = np.linspace(max(0, init_SAR-4), init_SAR+4, 9)
        temp_bins = np.linspace(30, 50, 11)
        rl_agent = rl_train_policy(model, base_inputs, sar_values=sar_vals, temp_bins=temp_bins, episodes=150, max_steps=30)
    st.success("RL policy (light) trained.")

# When user presses Predict & Regulate (top button) start full simulation (C behavior)
if predict_and_regulate:
    # Use the current top inputs as starting params
    SAR = init_SAR
    E_field = init_Efield
    Frequency_Index = init_Freq
    Depth = init_Depth
    AMC_Presence = init_AMC
    Tissue_Region = init_Tissue

    # If RL is selected but not trained, train a lightweight policy on-the-fly
    if use_rl and rl_agent is None:
        with st.spinner("Training quick RL policy for this run..."):
            sar_vals = np.linspace(max(0, SAR-4), SAR+4, 9)
            temp_bins = np.linspace(30, 50, 11)
            rl_agent = rl_train_policy(model, base_inputs, sar_values=sar_vals, temp_bins=temp_bins, episodes=80, max_steps=30)

    timestamps = []
    temps = []
    sars = []
    cem43_seconds = 0.0

    # initialize current temp to initial predicted temp for immediate visual continuity
    current_temp = predict_temp(model, SAR, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region)
    current_sar = SAR

    fig, ax = plt.subplots(figsize=(9,4))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.axhspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.12)

    max_time = sim_duration
    t = 0.0
    stop_sim = False

    # run simulation loop
    while t <= max_time:
        # Decide SAR
        if use_rl and rl_agent is not None:
            current_sar = rl_agent.policy_sar(current_temp)
        elif auto_regulate:
            pred_with_current = predict_temp(model, current_sar, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region)
            if pred_with_current < TARGET_MIN:
                current_sar += 0.2
            elif pred_with_current > TARGET_MAX:
                current_sar = max(0.0, current_sar - 0.2)

        # Model predicted target at this SAR
        model_pred = predict_temp(model, current_sar, E_field, Frequency_Index, Depth, AMC_Presence, Tissue_Region)

        # Thermal inertia update
        current_temp = current_temp + THERMAL_INERTIA * (model_pred - current_temp)

        # accumulate CEM43 for this timestep
        cem43_seconds += compute_cem43_increment(current_temp, time_step)

        # record
        timestamps.append(t)
        temps.append(current_temp)
        sars.append(current_sar)

        # update plot
        ax.clear()
        ax.plot(timestamps, temps, label="Temp (°C)")
        # plot sar scaled to temperature axis for quick view
        if len(sars) > 1:
            sar_scaled = np.interp(sars, (min(sars), max(sars)), (min(temps), max(temps)))
            ax.plot(timestamps, sar_scaled, '--', label="SAR (scaled)")
        ax.axhspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        chart_ph.pyplot(fig)

        # update metrics box
        metrics_ph.metric("Current Temp (°C)", f"{current_temp:.2f}")
        metrics_ph.metric("Current SAR (W/kg)", f"{current_sar:.2f}")

        # small pause to let UI update (not real-time)
        time.sleep(max(0.01, time_step * 0.1))
        t += time_step

    # Final outputs
    st.success("Regulation simulation complete.")
    cem43_minutes = cem43_seconds / 60.0
    cem43_ph.markdown(f"**Total CEM43 (s):** {cem43_seconds:.1f} s  —  **{cem43_minutes:.2f} minutes**")

    # SAR history plot
    fig2, ax2 = plt.subplots()
    ax2.plot(timestamps, sars)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SAR (W/kg)")
    st.pyplot(fig2)

    # Show final table
    df_out = pd.DataFrame({"time_s": timestamps, "temperature_C": temps, "SAR_Wkg": sars})
    st.dataframe(df_out.tail(10))

    # Final message
    if temps[-1] < TARGET_MIN:
        st.warning("Final temperature below therapeutic range.")
    elif temps[-1] > TARGET_MAX:
        st.error("Final temperature above safe limit.")
    else:
        st.success("Final temperature is within therapeutic window.")

# Show sensitivity results if available
if sensitivity_result is not None:
    base_temp, impacts = sensitivity_result
    st.subheader("Sensitivity Analysis (One-at-a-time)")
    st.write(f"Base predicted temp: {base_temp:.2f} °C")
    items = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    fig3, ax3 = plt.subplots()
    ax3.barh(names[::-1], vals[::-1], color=cm.viridis(np.linspace(0.2,0.9,len(vals))))
    ax3.set_xlabel("Absolute change in temperature (°C)")
    ax3.set_title("Parameter sensitivity (higher = more influential)")
    st.pyplot(fig3)

# If RL agent trained, show Q-table snapshot
if 'rl_agent' in locals() and rl_agent is not None:
    st.subheader("RL Policy Snapshot (Q-table)")
    st.write("SAR choices:", getattr(rl_agent, "sar_values", "N/A"))
    try:
        st.write(pd.DataFrame(rl_agent.Q, columns=[f"SAR={s:.2f}" for s in rl_agent.sar_values]))
    except Exception:
        st.write("RL Q-table not available for display.")

st.markdown("---")
st.caption("Note: This is a simulation using your ML model as a surrogate. For clinical usage integrate validated thermal models and real-time sensor feedback.")

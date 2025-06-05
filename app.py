import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler

# --------------------------------
# Load model and data
# --------------------------------
model = joblib.load("/Users/praneethbadanapally/Desktop/UMassD/Spring 2025/DSC 550 Major Project/AI_HeartDiseasePrediction/models/logistic_regression_model.pkl")
data = pd.read_csv("/Users/praneethbadanapally/Desktop/UMassD/Spring 2025/DSC 550 Major Project/AI_HeartDiseasePrediction/data/processed/processed_framingham_heart_study.csv")

expected_cols = model.feature_names_in_
scaler = StandardScaler()
scaler.fit(data[expected_cols])

# Create users.csv if not exists
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["email", "password"]).to_csv("users.csv", index=False)

# --------------------------------
# Dash App Setup
# --------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUX])
server = app.server

# --------------------------------
# Page Layouts
# --------------------------------
def login_page():
    return dbc.Container([
        
        html.H2("🏥 Welcome to AI Heart Disease Risk Predictor", className="text-center mt-3 mb-3"),
        html.H2("🔐 Login ", className="text-center mt-4 mb-4"),
        dbc.Input(id="login-email", placeholder="Enter email", type="email", className="mb-3"),
        dbc.Input(id="login-password", placeholder="Enter password", type="password", className="mb-3"),
        dbc.Button("Login", id="login-btn", color="primary", className="mb-2", n_clicks=0, style={"width":"100%"}),
        dbc.Button("Create Account", href="/signup", color="link", className="mb-2", style={"width":"100%"}),
        dbc.Button("Forgot Password?", href="/reset", color="link", style={"width":"100%"}),
        html.Div(id="login-msg", className="text-danger mt-2")
    ], style={"maxWidth": "400px", "marginTop": "50px"})

def signup_page():
    return dbc.Container([
        html.H2("📝 Signup", className="text-center mt-4 mb-4"),
        dbc.Input(id="signup-email", placeholder="Enter email", type="email", className="mb-3"),
        dbc.Input(id="signup-password", placeholder="Enter password", type="password", className="mb-3"),
        dbc.Button("Sign Up", id="signup-btn", color="success", className="mb-2", n_clicks=0, style={"width":"100%"}),
        dbc.Button("Back to Login", href="/", color="link", style={"width":"100%"}),
        html.Div(id="signup-msg", className="text-danger mt-2")
    ], style={"maxWidth": "400px", "marginTop": "50px"})

def reset_page():
    return dbc.Container([
        html.H2("🔄 Reset Password", className="text-center mt-4 mb-4"),
        dbc.Input(id="reset-email", placeholder="Enter your email", type="email", className="mb-3"),
        dbc.Input(id="reset-password", placeholder="Enter new password", type="password", className="mb-3"),
        dbc.Button("Reset Password", id="reset-btn", color="warning", className="mb-2", n_clicks=0, style={"width":"100%"}),
        dbc.Button("Back to Login", href="/", color="link", style={"width":"100%"}),
        html.Div(id="reset-msg", className="text-danger mt-2")
    ], style={"maxWidth": "400px", "marginTop": "50px"})

# Prediction Form Fields
form_fields = []
field_configs = {
    "gender": {"type": "dropdown", "options": [{"label": "Male", "value": 1}, {"label": "Female", "value": 0}]},
    "age": {"type": "number", "min": 20, "max": 100},
    "education": {"type": "dropdown", "options": [{"label": str(i), "value": i} for i in range(1, 5)]},
    "currentSmoker": {"type": "dropdown", "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]},
    "cigsPerDay": {"type": "number", "min": 0, "max": 50},
    "BPMeds": {"type": "dropdown", "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]},
    "prevalentStroke": {"type": "dropdown", "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]},
    "prevalentHyp": {"type": "dropdown", "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]},
    "diabetes": {"type": "dropdown", "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]},
    "totChol": {"type": "number", "min": 100, "max": 600},
    "systolic_bp": {"type": "number", "min": 80, "max": 300},
    "diastolic_bp": {"type": "number", "min": 40, "max": 200},
    "BMI": {"type": "number", "min": 10, "max": 60},
    "heartRate": {"type": "number", "min": 30, "max": 200},
    "glucose": {"type": "number", "min": 50, "max": 500},
    "pulsePressure": {"type": "number", "min": 10, "max": 150},
    "Smoking_Years": {"type": "number", "min": 0, "max": 70}
}

for col in expected_cols:
    config = field_configs[col]
    form_fields.append(dbc.Label(col, style={"marginTop": "10px"}))
    if config["type"] == "dropdown":
        form_fields.append(dcc.Dropdown(
            id=f"input-{col}",
            options=config["options"],
            placeholder=f"Select {col}",
            className="mb-3"
        ))
    else:
        form_fields.append(dcc.Input(
            id=f"input-{col}",
            type="number",
            min=config.get("min"),
            max=config.get("max"),
            step=1,
            placeholder=f"Enter {col}",
            className="mb-3 form-control"
        ))

def prediction_page():
    return dbc.Container([
        html.H2("🩺 AI Heart Disease Risk Predictor", className="text-center mt-4 mb-4"),
        dbc.Button("Logout", href="/", color="danger", className="mb-3", style={"width":"100%"}),
        dbc.Row([
            dbc.Col([
                html.Div(form_fields),
                dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3", style={"width":"100%"}),
                html.Div(id="prediction-output", className="mt-4")
            ], md=6),
            dbc.Col([
                html.H4("Feature Importance", className="text-center"),
                dcc.Graph(id="shap-graph")
            ], md=6)
        ])
    ], fluid=True)

# --------------------------------
# App Layout
# --------------------------------
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content")
])

# --------------------------------
# Callbacks
# --------------------------------

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/signup":
        return signup_page()
    elif pathname == "/reset":
        return reset_page()
    elif pathname == "/predict":
        return prediction_page()
    else:
        return login_page()

@app.callback(Output("login-msg", "children"), Input("login-btn", "n_clicks"), State("login-email", "value"), State("login-password", "value"), prevent_initial_call=True)
def login(n_clicks, email, password):
    users = pd.read_csv("users.csv")
    if ((users["email"] == email) & (users["password"] == password)).any():
        return dcc.Location(href="/predict", id="redirect-predict")
    else:
        return "Invalid login. Please try again."

@app.callback(Output("signup-msg", "children"), Input("signup-btn", "n_clicks"), State("signup-email", "value"), State("signup-password", "value"), prevent_initial_call=True)
def signup(n_clicks, email, password):
    users = pd.read_csv("users.csv")
    if (users["email"] == email).any():
        return "Email already exists."
    else:
        users = pd.concat([users, pd.DataFrame({"email": [email], "password": [password]})])
        users.to_csv("users.csv", index=False)
        return dcc.Location(href="/", id="redirect-login")

@app.callback(Output("reset-msg", "children"), Input("reset-btn", "n_clicks"), State("reset-email", "value"), State("reset-password", "value"), prevent_initial_call=True)
def reset_password(n_clicks, email, new_password):
    users = pd.read_csv("users.csv")
    if (users["email"] == email).any():
        users.loc[users["email"] == email, "password"] = new_password
        users.to_csv("users.csv", index=False)
        return dcc.Location(href="/", id="redirect-login2")
    else:
        return "Email not found."

@app.callback(
    Output("prediction-output", "children"),
    Output("shap-graph", "figure"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{col}", "value") for col in expected_cols],
    prevent_initial_call=True
)
def predict(n_clicks, *inputs):
    if n_clicks:
        input_df = pd.DataFrame([inputs], columns=expected_cols)
        scaled_input = scaler.transform(input_df)
        risk_score = model.predict_proba(scaled_input)[0][1]

        explainer = shap.Explainer(model, masker=scaler.transform(data[expected_cols]))
        shap_values = explainer(scaled_input)

        shap_df = pd.DataFrame({
            'Feature': expected_cols,
            'SHAP Value': shap_values.values[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        fig = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker=dict(color=shap_df["SHAP Value"], colorscale="RdBu")
        ))
        fig.update_layout(title="Feature Impact on Prediction", height=600)

        return (
            html.Div([
                html.H4(f"Predicted CHD Risk Score: {round(risk_score * 100, 2)}%", style={"color": "red"}),
                html.P("Higher score → Higher CHD risk.")
            ]),
            fig
        )
    return "", go.Figure()

# --------------------------------
# Run Server
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8055)


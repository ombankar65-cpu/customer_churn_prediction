import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the AdaBoost Model
MODEL_PATH = "ada_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found in the current directory.")

# Feature specification matching your pickled model's input features
FEATURE_NAMES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction"
]

# Single-file HTML/CSS/JS Dashboard Template with Animations & Plotly Visualizations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Analytics & Prediction Engine</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-hover: rgba(255, 255, 255, 0.08);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --card-radius: 20px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: hidden;
        }

        /* Glowing background orbs animation */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(90px);
            z-index: -1;
            opacity: 0.5;
            animation: float 10s ease-in-out infinite alternate;
        }
        .orb-1 {
            width: 350px;
            height: 350px;
            background: var(--accent-purple);
            top: -50px;
            left: -50px;
        }
        .orb-2 {
            width: 400px;
            height: 400px;
            background: var(--accent-pink);
            bottom: -100px;
            right: -100px;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(30px, 50px) scale(1.1); }
        }

        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            grid-column: 1 / -1;
            text-align: center;
            margin-bottom: 1rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #a855f7, #ec4899, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--card-radius);
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.25);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: var(--text-main);
        }

        .card-title i {
            color: var(--accent-purple);
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            color: white;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 10px 20px rgba(139, 92, 246, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(139, 92, 246, 0.4);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* Output & Visualization Section */
        .results-wrapper {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: 100%;
        }

        .prediction-badge {
            text-align: center;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            position: relative;
            overflow: hidden;
        }

        .prediction-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .prediction-value {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .prob-text {
            font-size: 0.95rem;
            color: var(--accent-cyan);
        }

        #chart-container {
            width: 100%;
            height: 280px;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <div class="header">
            <h1>AdaBoost Predictive Dashboard</h1>
            <p>Enter user telemetry parameters to perform real-time model inference</p>
        </div>

        <!-- Input Form Section -->
        <div class="glass-card">
            <div class="card-title">
                <i class="fa-solid fa-sliders"></i> Feature Inputs
            </div>
            <form id="prediction-form" class="form-grid">
                <div class="input-group">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" value="30" min="18" max="100" required>
                </div>
                
                <div class="input-group">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="0">Male (0)</option>
                        <option value="1">Female (1)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="Tenure">Tenure (Months)</label>
                    <input type="number" id="Tenure" name="Tenure" value="12" min="0" required>
                </div>

                <div class="input-group">
                    <label for="Usage Frequency">Usage Frequency</label>
                    <input type="number" id="Usage Frequency" name="Usage Frequency" value="15" min="0" required>
                </div>

                <div class="input-group">
                    <label for="Support Calls">Support Calls</label>
                    <input type="number" id="Support Calls" name="Support Calls" value="2" min="0" required>
                </div>

                <div class="input-group">
                    <label for="Payment Delay">Payment Delay (Days)</label>
                    <input type="number" id="Payment Delay" name="Payment Delay" value="1" min="0" required>
                </div>

                <div class="input-group">
                    <label for="Subscription Type">Subscription Type</label>
                    <select id="Subscription Type" name="Subscription Type" required>
                        <option value="0">Basic (0)</option>
                        <option value="1">Standard (1)</option>
                        <option value="2">Premium (2)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="Contract Length">Contract Length</label>
                    <select id="Contract Length" name="Contract Length" required>
                        <option value="0">Monthly (0)</option>
                        <option value="1">Quarterly (1)</option>
                        <option value="2">Annual (2)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="Total Spend">Total Spend ($)</label>
                    <input type="number" step="0.01" id="Total Spend" name="Total Spend" value="450.00" min="0" required>
                </div>

                <div class="input-group">
                    <label for="Last Interaction">Last Interaction (Days ago)</label>
                    <input type="number" id="Last Interaction" name="Last Interaction" value="5" min="0" required>
                </div>

                <button type="submit" class="submit-btn" id="submit-btn">
                    <span id="btn-text">Run Inference</span>
                    <div class="spinner" id="btn-spinner"></div>
                </button>
            </form>
        </div>

        <!-- Output Visualization Section -->
        <div class="glass-card">
            <div class="card-title">
                <i class="fa-solid fa-chart-pie"></i> Model Output & Analytics
            </div>
            <div class="results-wrapper">
                <div class="prediction-badge">
                    <div class="prediction-title">Classification Output</div>
                    <div class="prediction-value" id="pred-class" style="color: var(--accent-cyan);">Ready</div>
                    <div class="prob-text" id="pred-prob">Submit form data to calculate probability</div>
                </div>

                <div id="chart-container"></div>
            </div>
        </div>
    </div>

    <script>
        // Default Gauge Render
        function renderGauge(score) {
            const data = [{
                type: "indicator",
                mode: "gauge+number",
                value: score * 100,
                title: { text: "Confidence Probability (%)", font: { size: 14, color: "#94a3b8" } },
                number: { suffix: "%", font: { color: "#f8fafc", size: 24 } },
                gauge: {
                    axis: { range: [0, 100], tickwidth: 1, tickcolor: "#94a3b8" },
                    bar: { color: "#8b5cf6" },
                    bgcolor: "rgba(255, 255, 255, 0.05)",
                    borderwidth: 1,
                    bordercolor: "rgba(255, 255, 255, 0.12)",
                    steps: [
                        { range: [0, 50], color: "rgba(6, 182, 212, 0.2)" },
                        { range: [50, 100], color: "rgba(236, 72, 153, 0.2)" }
                    ]
                }
            }];

            const layout = {
                margin: { t: 30, r: 30, l: 30, b: 10 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: "#f8fafc", family: "Plus Jakarta Sans" }
            };

            Plotly.newPlot('chart-container', data, layout, {responsive: true, displayModeBar: false});
        }

        renderGauge(0);

        // Form Submit Ajax Handler
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();

            const btnSpinner = document.getElementById('btn-spinner');
            const btnText = document.getElementById('btn-text');
            btnSpinner.style.display = 'block';
            btnText.textContent = 'Processing...';

            const formData = new FormData(this);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    const classElem = document.getElementById('pred-class');
                    const probElem = document.getElementById('pred-prob');

                    classElem.textContent = `Class ${result.prediction}`;
                    classElem.style.color = result.prediction === 1 ? 'var(--accent-pink)' : 'var(--accent-cyan)';
                    
                    probElem.textContent = `Target Probability: ${(result.probability * 100).toFixed(2)}%`;
                    
                    renderGauge(result.probability);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Prediction Request Failed: ' + error);
            } finally {
                btnSpinner.style.display = 'none';
                btnText.textContent = 'Run Inference';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model file not found on server."})

    try:
        # Extract features in exact pickle training order
        features = [float(request.form.get(name, 0)) for name in FEATURE_NAMES]
        input_data = np.array([features])

        # Inference Execution
        prediction = int(model.predict(input_data)[0])
        
        # Calculate probability score if classifier supports predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability = float(probabilities[prediction])
        else:
            probability = 1.0

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

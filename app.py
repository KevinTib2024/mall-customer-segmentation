from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def cargar_datos():
    df = pd.read_csv("mall_customers.csv")
    return df

def clustering():
    df = cargar_datos()

    # Usar solo variables importantes
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)

    return df.to_dict(orient="records")

@app.route("/")
def index():
    datos = clustering()
    return render_template("index.html", clientes=datos)

if __name__ == "__main__":
    app.run(debug=True)
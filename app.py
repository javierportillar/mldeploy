import os
import csv
from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Carga el modelo al arrancar la app
with open("model_admit5.pkl", "rb") as f:
    model = pickle.load(f)

scaler = StandardScaler()
scaler.mean_ = np.array(
    [1.96139401e-16, 1.94828721e-16, 1.99840144e-16, -1.23358114e-18, -9.47390314e-16]
)
scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# Variable global para el ID
current_id = 1


# Ruta del archivo CSV
csv_file_path = os.path.join("data", "predictions.csv")

# Crea el archivo CSV si no existe y escribe el encabezado
if not os.path.exists(csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Type_A", "Type_B", "Type_C", "Non_urgent", "Urgent"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global current_id  # Declara que usarás la variable global
    feature_order = ["Type_A", "Type_B", "Type_C", "Non_urgent", "Urgent"]
    form = request.form.to_dict()

    # Validación y conversión a float
    try:
        raw_vals = [float(form[name]) for name in feature_order]
    except KeyError as e:
        missing = e.args[0]
        return render_template("index.html", error=f"Falta el campo «{missing}».")
    except ValueError:
        return render_template("index.html", error="Por favor ingresa números válidos.")

    # Valores aprendidos por StandardScaler
    media = np.array([52.3672, 100.46431111, 137.14106667, 166.69064444, 116.30835556])
    std = np.array([17.87748566, 45.41122318, 36.66441601, 65.52173911, 23.61432266])

    # Transformación manual (estandarización)
    x_test_scaled = (raw_vals - media) / std

    intercept = 2.89972578e02  # primer valor de coef_lasso_orig

    coefs = np.array(
        [
            1.79258355e01,  # coef para Type_B
            4.55345338e01,  # coef para Non_urgent
            3.67863855e01,  # coef para Banking_1
            -1.97548282e-01,  # coef para Order_type_A (p. ej.)
            -6.79282460e-02,  # coef para Order_type_C (p. ej.)
        ]
    )

    # Predicción
    pred2 = intercept + np.dot(x_test_scaled[0], coefs)
    pred1 = model.predict([x_test_scaled])[0]
    app.logger.info(pred1)

    # Guarda los datos en el archivo CSV
    # with open(csv_file_path, mode="a", newline="") as file:
    #    writer = csv.writer(file)
    # writer.writerow([current_id] + raw_vals.tolist() + [pred2])
    #    writer.writerow([current_id] + raw_vals + [pred2])

    # Incrementa el ID
    current_id += 1

    print(x_test_scaled)

    return render_template(
        "index.html",
        prediccion_text=f"Total de órdenes a entregar: {pred1:.2f}",
        # form_data=form,
        form_data={feature_order[i]: raw_vals[i] for i in range(len(feature_order))},
    )


if __name__ == "__main__":
    app.run(debug=True)

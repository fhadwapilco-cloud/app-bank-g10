import numpy as np
import pandas as pd


def log_firmado(x):
    # log1p con signo: soporta saldos negativos sin perder el signo (igual que en el notebook)
    return np.sign(x) * np.log1p(np.abs(x))


def registrar_funciones():
    # joblib guarda log_firmado como '__main__.log_firmado'; esto la hace visible al cargar el modelo
    import __main__
    __main__.log_firmado = log_firmado


def preparar_cliente(cliente, orden_columnas):
    # Recibe un diccionario con los datos 'en crudo' de un cliente (como vienen en bank-full)
    # y aplica las mismas transformaciones de la seccion 5 antes de pasarlo al pipeline.
    d = dict(cliente)
    for c in ["default", "housing", "loan"]:
        if d.get(c) in ("yes", "no"):
            d[c] = 1 if d[c] == "yes" else 0
    pdays = d.get("pdays", -1)
    d["contactado_antes"] = int(pdays != -1)
    d["pdays"] = np.nan if pdays == -1 else pdays
    if d.get("poutcome") == "unknown":
        d["poutcome"] = "sin_contacto_previo"
    return pd.DataFrame([d])[orden_columnas]

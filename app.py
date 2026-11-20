"""
App de priorización de llamadas - Proyecto Applied Machine Learning · Grupo 10
Bank Marketing: ¿a qué clientes llamar primero para ofrecer un depósito a plazo fijo?

Modelo: LightGBM con pesos de clase (el campeón del notebook).
Ejecutar con:  streamlit run app.py
Necesita la carpeta modelos_streamlit/ (exportada desde Colab, sección 6 del notebook)
al lado de este archivo:   modelos_streamlit/LightGBM/modelo.joblib  y  ficha.json
"""
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Priorización de llamadas · G10", page_icon="📞", layout="wide")

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # paleta del notebook (apta para daltonismo)


# ------------------------------------------------------------
# Función que el pipeline guardado necesita (igual que en el notebook).
# joblib la guardó como '__main__.log_firmado', por eso debe existir en este archivo
# y registrarse ANTES de cargar el .joblib.
# ------------------------------------------------------------
def log_firmado(x):
    # log1p con signo: soporta saldos negativos sin perder el signo
    return np.sign(x) * np.log1p(np.abs(x))


sys.modules["__main__"].log_firmado = log_firmado

# ------------------------------------------------------------
# Textos en español para las categorías del dataset
# ------------------------------------------------------------
JOB = {"admin.": "Administrativo", "blue-collar": "Obrero / operario", "entrepreneur": "Empresario",
       "housemaid": "Empleada doméstica", "management": "Gerencia", "retired": "Jubilado",
       "self-employed": "Independiente", "services": "Servicios", "student": "Estudiante",
       "technician": "Técnico", "unemployed": "Desempleado", "unknown": "Sin dato"}
MARITAL = {"married": "Casado/a", "single": "Soltero/a", "divorced": "Divorciado/a o viudo/a"}
EDU = {"primary": "Primaria", "secondary": "Secundaria", "tertiary": "Superior", "unknown": "Sin dato"}
POUT = {"failure": "No suscribió", "other": "Otro resultado", "success": "Sí suscribió",
        "sin_contacto_previo": "Sin contacto previo"}
VAR = {"age": "Edad", "job": "Ocupación", "marital": "Estado civil", "education": "Educación",
       "default": "Mora", "balance": "Saldo", "housing": "Hipoteca", "loan": "Préstamo personal",
       "pdays": "Días desde el último contacto", "previous": "Contactos previos",
       "poutcome": "Resultado de la campaña anterior", "contactado_antes": "Contactado antes"}
COLS_CAT = ["job", "marital", "education", "poutcome"]      # las que el pipeline convierte en One-Hot
COLS_BIN = ["default", "housing", "loan"]

# ------------------------------------------------------------
# Resultados del notebook que no viajan en ficha.json (secciones 5.6.3, 5.6.4 y 5.6.5).
# Si vuelves a ejecutar el notebook y cambian, actualiza estas tablas.
# ------------------------------------------------------------
REF_LIFT = pd.DataFrame({
    "% de clientes llamados": [5, 10, 20, 30, 40, 50],
    "Llamadas": [452, 904, 1809, 2713, 3617, 4522],
    "Suscripciones captadas": [258, 405, 558, 658, 736, 810],
    "% de suscripciones capturadas": [24.4, 38.3, 52.7, 62.2, 69.6, 76.6],
    "Precisión": [0.571, 0.448, 0.308, 0.243, 0.203, 0.179],
    "Lift": [4.88, 3.83, 2.64, 2.07, 1.74, 1.53],
})
REF_SENS = pd.DataFrame({
    "Valor suscripción / costo llamada": [3, 5, 8.5, 10, 15, 20, 30],
    "Umbral": [0.76, 0.62, 0.51, 0.47, 0.35, 0.27, 0.13],
    "% de clientes llamados": [6.8, 11.9, 22.8, 28.1, 58.1, 81.0, 99.7],
    "% de suscripciones capturadas": [29.7, 41.8, 55.6, 61.0, 81.2, 92.6, 99.5],
    "Beneficio con el modelo": [324, 1130, 2934, 3907, 7631, 12272, 22571],
    "Beneficio llamando a todos": [-5869, -3753, -50, 1537, 6827, 12117, 22697],
})


# ============================================================
# FUNCIONES DE APOYO (no dependen de la interfaz)
# ============================================================
def leer_modelos(carpeta):
    """Busca ficha.json + modelo.joblib en la carpeta indicada (relativa a donde está app.py, o ruta completa),
    incluso si quedaron dentro de subcarpetas anidadas al descomprimir el zip."""
    modelos = {}
    aqui = Path(__file__).resolve().parent
    base = Path(carpeta.strip().strip('"'))
    for b in ([base] if base.is_absolute() else [aqui / base, Path.cwd() / base]):
        if not b.exists():
            continue
        for ficha_json in sorted(b.rglob("ficha.json")):
            sub = ficha_json.parent
            if (sub / "modelo.joblib").exists():
                ficha = json.loads(ficha_json.read_text(encoding="utf-8"))
                ficha["ruta"] = str(sub / "modelo.joblib")
                ficha["carpeta"] = str(sub)
                modelos[ficha["nombre"]] = ficha
        if modelos:
            break
    return modelos


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo(ruta):
    """Carga el pipeline completo (preprocesamiento + LightGBM) una sola vez."""
    return joblib.load(ruta)


def preparar_clientes(df, orden_columnas):
    """Datos 'en crudo' (como en bank-full) -> columnas que espera el pipeline.
    Aplica las mismas transformaciones deterministas de la sección 3 del notebook."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    faltan = [c for c in orden_columnas if c not in d.columns and c != "contactado_antes"]
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))
    for c in COLS_BIN:                                           # yes/no -> 1/0
        d[c] = pd.to_numeric(d[c].replace({"yes": 1, "no": 0, "Yes": 1, "No": 0}), errors="coerce")
    for c in ["age", "balance", "previous"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    pdays = pd.to_numeric(d["pdays"], errors="coerce").fillna(-1)   # vacío o -1 = nunca contactado
    d["contactado_antes"] = (pdays >= 0).astype(int)
    d["pdays"] = pdays.where(pdays >= 0, np.nan)                  # el -1 era un centinela, no un número de días
    d["poutcome"] = d["poutcome"].replace({"unknown": "sin_contacto_previo"})
    return d[orden_columnas]


def puntuar(modelo, X):
    return modelo.predict_proba(X)[:, 1]


def variable_original(nombre_columna):
    n = nombre_columna.split("__", 1)[-1]
    for c in COLS_CAT:
        if n.startswith(c + "_"):
            return c
    return n


def contribuciones(modelo, X_fila):
    """Aporte de cada variable al puntaje de UN cliente (valores tipo SHAP de LightGBM, en log-odds)."""
    prep, lgb = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    Xt = prep.transform(X_fila)
    contrib = np.asarray(lgb.predict(Xt, pred_contrib=True))[0][:-1]   # el último valor es el sesgo
    nombres = [variable_original(n) for n in prep.get_feature_names_out()]
    return pd.Series(contrib, index=nombres).groupby(level=0).sum().sort_values(key=np.abs, ascending=False)


def tabla_lift(y_true, score, fracs=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)):
    orden = np.argsort(-score)
    ys, base, filas = y_true[orden], y_true.mean(), []
    for f in fracs:
        k = max(int(round(f * len(ys))), 1)
        cap = ys[:k].sum()
        filas.append({"% de clientes llamados": int(f * 100), "Llamadas": k, "Suscripciones captadas": int(cap),
                      "% de suscripciones capturadas": round(cap / max(ys.sum(), 1) * 100, 1),
                      "Precisión": round(cap / k, 3), "Lift": round((cap / k) / base, 2) if base > 0 else np.nan})
    return pd.DataFrame(filas)


def leer_etiqueta(df):
    """Si el CSV trae el resultado real (y_real o y), lo devuelve como 0/1; si no, None."""
    for c in ["y_real", "y"]:
        if c in df.columns:
            s = df[c].replace({"yes": 1, "no": 0})
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().all() and set(s.unique()) <= {0, 1}:
                return s.astype(int).to_numpy()
    return None


def rango(ficha, col, defecto):
    info = ficha["columnas_entrada"].get(col, {})
    return info.get("min", defecto[0]), info.get("max", defecto[1]), info.get("mediana", defecto[2])


def opciones(ficha, col, fallback):
    return ficha["columnas_entrada"].get(col, {}).get("opciones", fallback)


# ============================================================
# INTERFAZ
# ============================================================
st.session_state.setdefault("carpeta", "modelos_streamlit")
modelos_disp = leer_modelos(st.session_state.carpeta)
ficha = None
if modelos_disp:
    ficha = modelos_disp.get("LightGBM") or next(iter(modelos_disp.values()))
    st.session_state.setdefault("umbral", float(ficha["umbral"]))
    st.session_state.setdefault("valor", float(ficha.get("supuestos", {}).get("valor_suscripcion", 10.0)))
    st.session_state.setdefault("costo", float(ficha.get("supuestos", {}).get("costo_llamada", 1.0)))

st.title("📞 Priorización de llamadas · Depósitos a plazo fijo")
st.caption("Proyecto Applied Machine Learning · Grupo 10 · Bank Marketing. "
           "El modelo ordena a los clientes por probabilidad de suscribir **antes de llamarlos**, "
           "para decidir a quién contactar primero.")

tab_conf, tab_modelo, tab_cliente, tab_lote, tab_metricas = st.tabs(
    ["⚙️ Configuración", "🧠 Modelo", "👤 Cliente individual", "📂 Lote de clientes", "📊 Métricas y límites"])

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
with tab_conf:
    st.header("⚙️ Configuración")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Carpeta del modelo", key="carpeta",
                      help="Carpeta donde descomprimiste modelos_streamlit.zip")
        if ficha:
            st.slider("Umbral de decisión (puntaje mínimo para llamar)", 0.01, 0.99, step=0.01, key="umbral",
                      help="Un cliente se marca 'LLAMAR' si su puntaje es mayor o igual a este valor. "
                           "El recomendado sale del análisis costo-beneficio del notebook (sección 4.4).")
            if st.button("↩️ Volver al umbral recomendado"):
                st.session_state.umbral = float(ficha["umbral"])
                st.rerun()
    with col2:
        if ficha:
            st.number_input("Valor de una suscripción (unidades)", min_value=0.1, step=1.0, key="valor",
                            help="Solo se usa para calcular el beneficio en el lote de clientes cuando el CSV trae el resultado real.")
            st.number_input("Costo de una llamada (unidades)", min_value=0.1, step=0.5, key="costo")
            st.caption("Los valores por defecto (10 y 1) son los supuestos **ilustrativos** del notebook; "
                       "ajústalos con datos reales del banco.")

    if ficha:
        st.success(f"Modelo cargado: **{ficha['nombre']}** · umbral recomendado {float(ficha['umbral']):.2f}")
        if abs(st.session_state.umbral - float(ficha["umbral"])) > 1e-9:
            st.warning(f"Estás usando el umbral {st.session_state.umbral:.2f}, distinto del recomendado "
                       f"({float(ficha['umbral']):.2f}). Las métricas de la pestaña *Métricas* corresponden al recomendado.")
    else:
        st.error(f"No se encontró el modelo en la carpeta '{st.session_state.carpeta}'. "
                 "Debe existir modelos_streamlit/LightGBM/modelo.joblib y ficha.json al lado de app.py.")
        st.caption(f"La app está en: {Path(__file__).resolve().parent} · "
                   f"Streamlit se ejecutó desde: {Path.cwd()}. "
                   "También puedes pegar arriba la ruta completa de la carpeta.")

if ficha is None:
    st.stop()

try:
    modelo = cargar_modelo(ficha["ruta"])
except Exception as e:                                          # versiones distintas de las librerías, etc.
    import sklearn
    st.error("No se pudo cargar el modelo. Casi siempre es porque las versiones de las librerías no coinciden "
             "con las de Colab. Instala las del archivo `modelos_streamlit/requirements.txt` "
             "(`pip install -r modelos_streamlit/requirements.txt`).")
    st.code(f"{type(e).__name__}: {e}\nscikit-learn instalado: {sklearn.__version__}")
    st.stop()

UMBRAL = float(st.session_state.umbral)
ORDEN = ficha["orden_columnas"]
M = ficha.get("metricas_test", {})

# ============================================================
# 2. MODELO
# ============================================================
with tab_modelo:
    st.header("🧠 Modelo elegido")
    st.subheader(ficha["nombre"])
    st.write(ficha.get("descripcion", ""))
    st.markdown(
        "**Por qué este modelo.** Entre 8 modelos comparados (regresión logística, árbol, Random Forest, Gradient Boosting, "
        "XGBoost, LightGBM, CatBoost y red neuronal) tuvo el mejor PR-AUC en validación cruzada. Los pesos de clase "
        "funcionaron mejor que `SMOTENC` en los 7 modelos de árboles/lineales. Las diferencias entre los ensambles son pequeñas.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC (test)", f"{M.get('PR-AUC', float('nan')):.3f}", help="Un modelo al azar obtendría ≈ 0.117 (la tasa base).")
    c2.metric("ROC-AUC (test)", f"{M.get('ROC-AUC', float('nan')):.3f}")
    c3.metric("Recall (test)", f"{M.get('Recall', float('nan')):.1%}")
    c4.metric("Precisión (test)", f"{M.get('Precision', float('nan')):.1%}")
    st.markdown("**Variables que usa el modelo** (solo información disponible *antes* de llamar):")
    st.write(", ".join(f"`{c}`" for c in ORDEN))
    st.markdown("**Variables que se excluyeron a propósito:** `duration` (solo se conoce al terminar la llamada), "
                "`campaign` (contactos de *esta* campaña), `month`, `day` y `contact` (identifican la fase de la campaña, "
                "no al cliente). Así se evita la fuga de información.")

# ============================================================
# 3. CLIENTE INDIVIDUAL
# ============================================================
PRESETS = {
    "Joven sin hipoteca": dict(f_age=22, f_job="student", f_marital="single", f_edu="tertiary", f_balance=800,
                               f_default=False, f_housing=False, f_loan=False, f_previo=False),
    "Jubilado con campaña previa exitosa": dict(f_age=66, f_job="retired", f_marital="married", f_edu="secondary",
                                               f_balance=3500, f_default=False, f_housing=False, f_loan=False,
                                               f_previo=True, f_pdays=120, f_previous=2, f_pout="success"),
    "Adulto con hipoteca y préstamo": dict(f_age=42, f_job="blue-collar", f_marital="married", f_edu="secondary",
                                           f_balance=300, f_default=False, f_housing=True, f_loan=True, f_previo=False),
}


def aplicar_preset(nombre):
    for k, v in PRESETS[nombre].items():
        st.session_state[k] = v


with tab_cliente:
    st.header("👤 ¿Llamamos a este cliente?")
    st.write("Ingresa los datos del cliente. No se pide nada de la llamada (duración, mes, canal), "
             "porque el modelo decide **antes** de llamar.")
    st.caption("Ejemplos rápidos:")
    cols_p = st.columns(len(PRESETS))
    for col, nombre in zip(cols_p, PRESETS):
        col.button(nombre, on_click=aplicar_preset, args=(nombre,), key="btn_" + nombre)

    col_form, col_res = st.columns([1, 1.2])
    with col_form:
        a_min, a_max, a_med = rango(ficha, "age", (18, 95, 39))
        b_min, b_max, b_med = rango(ficha, "balance", (-8019, 102127, 448))
        jobs = opciones(ficha, "job", list(JOB))
        maritals = opciones(ficha, "marital", list(MARITAL))
        edus = opciones(ficha, "education", list(EDU))
        pouts = [p for p in opciones(ficha, "poutcome", list(POUT)) if p != "sin_contacto_previo"]

        # Valores iniciales (se definen en session_state para que los ejemplos rápidos puedan cambiarlos)
        for k, v in dict(f_age=int(round(a_med)), f_job=jobs[0], f_marital=maritals[0], f_edu=edus[0],
                         f_balance=int(round(b_med)), f_default=False, f_housing=False, f_loan=False,
                         f_previo=False, f_pdays=180, f_previous=1, f_pout=pouts[0]).items():
            st.session_state.setdefault(k, v)

        st.number_input("Edad", min_value=int(a_min), max_value=int(a_max), key="f_age")
        st.selectbox("Ocupación", jobs, format_func=lambda x: JOB.get(x, x), key="f_job")
        st.selectbox("Estado civil", maritals, format_func=lambda x: MARITAL.get(x, x), key="f_marital")
        st.selectbox("Educación", edus, format_func=lambda x: EDU.get(x, x), key="f_edu")
        st.number_input("Saldo anual promedio (euros)", min_value=-20000, max_value=150000, step=100, key="f_balance")
        cc1, cc2, cc3 = st.columns(3)
        cc1.checkbox("En mora", key="f_default")
        cc2.checkbox("Tiene hipoteca", key="f_housing")
        cc3.checkbox("Tiene préstamo personal", key="f_loan")
        previo = st.checkbox("Fue contactado en una campaña anterior", key="f_previo")
        if previo:
            _, p_max, _ = rango(ficha, "pdays", (1, 871, 180))
            _, q_max, _ = rango(ficha, "previous", (0, 275, 1))
            st.number_input("Días desde el último contacto", min_value=0, max_value=int(max(p_max, 900)), key="f_pdays")
            st.number_input("Número de contactos previos", min_value=1, max_value=int(max(q_max, 1)), key="f_previous")
            st.selectbox("Resultado de esa campaña", pouts, format_func=lambda x: POUT.get(x, x), key="f_pout")
            pdays, previous, poutcome = st.session_state.f_pdays, st.session_state.f_previous, st.session_state.f_pout
        else:
            pdays, previous, poutcome = -1, 0, "unknown"

    cliente = {"age": st.session_state.f_age, "job": st.session_state.f_job, "marital": st.session_state.f_marital,
               "education": st.session_state.f_edu, "default": int(st.session_state.f_default),
               "balance": st.session_state.f_balance, "housing": int(st.session_state.f_housing),
               "loan": int(st.session_state.f_loan), "pdays": pdays, "previous": previous, "poutcome": poutcome}
    X_cli = preparar_clientes(pd.DataFrame([cliente]), ORDEN)
    score = float(puntuar(modelo, X_cli)[0])

    with col_res:
        if score >= UMBRAL:
            st.success("### 📞 LLAMAR: prioridad alta")
        else:
            st.info("### ⏸️ No priorizar por ahora")
        r1, r2, r3 = st.columns(3)
        r1.metric("Puntaje del modelo", f"{score:.2f}")
        r2.metric("Umbral", f"{UMBRAL:.2f}")
        r3.metric("Diferencia", f"{score - UMBRAL:+.2f}")
        st.progress(min(max(score, 0.0), 1.0))
        prec = M.get("Precision")
        if prec and abs(UMBRAL - float(ficha["umbral"])) < 1e-9:
            st.caption(f"Con este umbral, en el conjunto de prueba suscribió el {prec:.0%} de los clientes marcados "
                       f"para llamar (frente a 11.7% si se llama sin filtrar): unas {1 / prec:.1f} llamadas por suscripción.")
        st.caption("⚠️ El puntaje sirve para **ordenar** clientes; no es una probabilidad calibrada "
                   "(los pesos de clase lo inflan). Un puntaje de 0.60 no significa 60% de probabilidad de suscribir.")

        st.markdown("**¿Qué empuja el puntaje de este cliente?**")
        try:
            contrib = contribuciones(modelo, X_cli)
            top = contrib.head(6)[::-1]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh([VAR.get(i, i) for i in top.index], top.values,
                    color=[AZUL if v >= 0 else NARANJA for v in top.values], height=0.65)
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_xlabel("← baja el puntaje          sube el puntaje →")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Efecto de cada variable respecto al cliente promedio del modelo (escala log-odds, valores tipo SHAP).")
        except Exception as e:
            st.caption(f"No se pudo calcular la explicación ({type(e).__name__}).")

# ============================================================
# 4. LOTE DE CLIENTES
# ============================================================
with tab_lote:
    st.header("📂 Priorizar una lista de clientes")
    st.write("Sube un CSV con una fila por cliente y estas columnas (como en `bank-full.csv`): "
             + ", ".join(f"`{c}`" for c in ORDEN if c != "contactado_antes") +
             ". `pdays = -1` (o vacío) significa que nunca fue contactado. El separador puede ser `,` o `;`.")
    ejemplo_csv = Path(ficha["carpeta"]).parent / "clientes_ejemplo.csv"
    if not ejemplo_csv.exists():
        ejemplo_csv = Path(ficha["carpeta"]) / "clientes_ejemplo.csv"
    archivo = st.file_uploader("Archivo CSV de clientes", type=["csv"])
    usar_ejemplo = ejemplo_csv.exists() and st.checkbox(f"Usar {ejemplo_csv.name} (30 clientes del test, con su resultado real)")

    df_lote, origen = None, None
    try:
        if archivo is not None:
            df_lote = pd.read_csv(io.BytesIO(archivo.getvalue()), sep=None, engine="python")
            origen = archivo.name
        elif usar_ejemplo:
            df_lote, origen = pd.read_csv(ejemplo_csv), ejemplo_csv.name
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")

    if df_lote is not None:
        try:
            X_lote = preparar_clientes(df_lote, ORDEN)
        except ValueError as e:
            st.error(str(e))
            X_lote = None
        if X_lote is not None:
            nulos = X_lote.drop(columns=["pdays"]).isna().sum()
            if nulos.sum() > 0:
                st.warning("Hay valores vacíos o no numéricos en: "
                           + ", ".join(f"{c} ({n})" for c, n in nulos[nulos > 0].items())
                           + ". El modelo los tolera, pero el puntaje puede ser menos confiable.")
            s = puntuar(modelo, X_lote)
            out = df_lote.copy()
            out["puntaje"] = s.round(4)
            out["decision"] = np.where(s >= UMBRAL, "LLAMAR", "No priorizar")
            out["ranking"] = pd.Series(s).rank(ascending=False, method="first").astype(int).to_numpy()
            out = out.sort_values("ranking").reset_index(drop=True)

            n_llamar = int((s >= UMBRAL).sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Clientes en el archivo", f"{len(out):,}")
            k2.metric("Marcados para llamar", f"{n_llamar:,}", f"{n_llamar / len(out):.1%} de la lista", delta_color="off")
            k3.metric("Puntaje mediano", f"{np.median(s):.2f}")

            st.subheader("Lista priorizada (mejores primero)")
            st.dataframe(out.head(500), hide_index=True)
            if len(out) > 500:
                st.caption(f"Se muestran los 500 primeros de {len(out):,}; el CSV descargable trae todos.")
            st.download_button("⬇️ Descargar lista priorizada (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"priorizacion_{Path(origen).stem}.csv", mime="text/csv")

            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.hist(s, bins=30, color=AZUL, alpha=0.9)
            ax.axvline(UMBRAL, color="red", ls="--", lw=1.2, label=f"Umbral = {UMBRAL:.2f}")
            ax.set_xlabel("Puntaje del modelo"); ax.set_ylabel("Clientes"); ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            y_real = leer_etiqueta(df_lote)
            if y_real is not None and y_real.sum() > 0:
                st.subheader("Verificación con el resultado real del archivo")
                llamar = s >= UMBRAL
                cap = int(y_real[llamar].sum())
                valor, costo = st.session_state.valor, st.session_state.costo
                b_mod = cap * valor - int(llamar.sum()) * costo
                b_todos = int(y_real.sum()) * valor - len(y_real) * costo
                v1, v2, v3 = st.columns(3)
                v1.metric("Suscripciones capturadas", f"{cap} de {int(y_real.sum())}", f"{cap / y_real.sum():.0%}", delta_color="off")
                v2.metric("Beneficio llamando solo a los marcados", f"{b_mod:,.0f}")
                v3.metric("Beneficio llamando a todos", f"{b_todos:,.0f}")
                st.dataframe(tabla_lift(y_real, s), hide_index=True)
                if len(y_real) < 500:
                    st.caption("Con tan pocos clientes las cifras son muy ruidosas: sirven para probar la app, no para evaluar el modelo.")

# ============================================================
# 5. MÉTRICAS Y LÍMITES
# ============================================================
with tab_metricas:
    st.header("📊 Métricas y límites del modelo")
    st.subheader("Desempeño en el conjunto de prueba (9,043 clientes, split aleatorio)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", f"{M.get('PR-AUC', float('nan')):.3f}", help="Tasa base 0.117")
    m2.metric("ROC-AUC", f"{M.get('ROC-AUC', float('nan')):.3f}")
    m3.metric("F1", f"{M.get('F1-Score', float('nan')):.3f}")
    m4.metric("Beneficio (test)", f"{M.get('Beneficio test', float('nan')):,.0f}",
              help="Con costo 1 por llamada y valor 10 por suscripción. Llamar a todos: 1,537.")
    n1, n2 = st.columns(2)
    n1.metric("% de clientes a llamar", f"{M.get('% clientes a llamar', float('nan')):.1f}%")
    n2.metric("% de suscripciones capturadas", f"{M.get('% suscripciones capturadas', float('nan')):.1f}%")

    col_cm, col_imp = st.columns(2)
    with col_cm:
        st.subheader("Matriz de confusión (test)")
        cm = np.array(ficha["matriz_confusion"])
        et = ficha.get("etiquetas_matriz", {"filas": ["No suscribe", "Suscribe"], "columnas": ["No llamar", "Llamar"]})
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks(range(len(et["columnas"]))); ax.set_xticklabels(et["columnas"])
        ax.set_yticks(range(len(et["filas"]))); ax.set_yticklabels(et["filas"])
        ax.set_xlabel("Decisión del modelo"); ax.set_ylabel("Resultado real")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col_imp:
        st.subheader("Variables más importantes")
        imp = ficha.get("importancia_variables")
        if imp:
            s_imp = pd.Series(imp).sort_values()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh([VAR.get(i, i) for i in s_imp.index], s_imp.values, color=AZUL, height=0.65)
            ax.set_xlabel("Caída del PR-AUC al mezclar la variable")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("Importancia por permutación en test (sección 5.3.1 del notebook).")
        else:
            st.info("La ficha no trae importancia de variables.")

    st.subheader("¿Cuánto ayuda priorizar? (lift en test)")
    st.dataframe(REF_LIFT, hide_index=True)
    st.caption("Llamando solo al 20% mejor puntuado se capturan ≈ 53% de las suscripciones. "
               "Cifras de la sección 5.6.3 del notebook.")

    st.subheader("¿Cómo cambia el umbral si cambia el valor de una suscripción?")
    st.dataframe(REF_SENS, hide_index=True)
    st.caption("Si una suscripción vale menos de ≈ 8.5 veces el costo de una llamada, llamar a todos da pérdidas y el modelo "
               "es imprescindible. Cifras de la sección 5.6.4. Usa el umbral correspondiente en *Configuración*.")

    st.subheader("⚠️ Límites que hay que tener presentes")
    st.warning(
        "- **Datos de 2008–2010 de un banco portugués.** No se asume que las tasas y patrones se trasladen a otro mercado o época.\n"
        "- **Validación temporal (sección 5.6.5):** entrenando con el primer 80% de las filas y probando con el último 20%, "
        "el ROC-AUC baja a ≈ 0.69 y el PR-AUC es 0.477 con una tasa base de 31.6%. Con el umbral fijo del pasado, "
        "el modelo **no superó a llamar a todos** en ese periodo. Conviene **recalibrar el umbral con datos recientes**.\n"
        "- **El puntaje ordena, no es una probabilidad calibrada.**\n"
        "- **Costo y valor son ilustrativos** (1 y 10). El umbral óptimo depende de los costos reales del call center.\n"
        "- La herramienta apoya la decisión de a quién llamar primero; no reemplaza el criterio del equipo comercial.")

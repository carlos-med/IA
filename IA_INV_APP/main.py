import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="IA INV",
    page_icon="🥬",
    layout="wide"
)

# =========================
# RUTA DEL MODELO ENTRENADO
# =========================
RUTA_MODELO = "runs/classify/train-4/weights/best.pt"


# =========================
# CARGAR MODELO
# =========================
@st.cache_resource
def cargar_modelo():
    return YOLO(RUTA_MODELO)


# =========================
# INTERFAZ
# =========================
st.title("🥬 IA INV")

st.subheader(
    "Sistema Inteligente de Diagnóstico de Enfermedades en Lechugas"
)

st.divider()


# =========================
# VERIFICAR MODELO
# =========================
if not os.path.exists(RUTA_MODELO):
    st.error(f"❌ No se encontró el modelo: {RUTA_MODELO}")
    st.stop()


modelo = cargar_modelo()


# =========================
# SUBIR IMAGEN
# =========================
imagen_subida = st.file_uploader(
    "📷 Sube una imagen de la lechuga",
    type=["jpg", "jpeg", "png"]
)


if imagen_subida is not None:

    imagen = Image.open(imagen_subida).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            imagen,
            caption="Imagen seleccionada",
            use_container_width=True
        )

    with col2:

        st.write("## 🧠 Análisis de IA")

        if st.button(
            "🔍 ANALIZAR LECHUGA",
            use_container_width=True
        ):

            with st.spinner(
                "La IA está analizando la imagen..."
            ):

                resultados = modelo(imagen)

                resultado = resultados[0]

                clase_id = resultado.probs.top1

                confianza = (
                    resultado.probs.top1conf.item()
                )

                diagnostico = (
                    resultado.names[clase_id]
                )

            st.success("✅ Análisis completado")

            st.write("## 📋 Resultado")

            # Resultado saludable
            if diagnostico.lower() == "healthy":

                st.success(
                    f"🥬 Estado: {diagnostico}"
                )

            # Resultado con enfermedad
            else:

                st.error(
                    f"⚠️ Diagnóstico: {diagnostico}"
                )

            # Confianza
            st.metric(
                "🎯 Confianza de la IA",
                f"{confianza * 100:.2f}%"
            )

            # Mensaje según confianza
            if confianza < 0.70:

                st.warning(
                    "⚠️ Resultado poco confiable. "
                    "La iluminación o calidad de la imagen "
                    "podría afectar el diagnóstico."
                )

            elif confianza < 0.85:

                st.info(
                    "ℹ️ Resultado con confianza moderada. "
                    "Se recomienda analizar otra imagen "
                    "para confirmar."
                )

            else:

                st.success(
                    "🟢 Resultado con alta confianza."
                )
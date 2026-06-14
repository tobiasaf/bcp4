# BCPsim - Simulador Estratégico Agrícola (Demo)

Este repositorio contiene la versión interactiva y autocontenida del **Simulador de Escenarios** diseñado para la **Bolsa de Cereales y Productos de Bahía Blanca (BCP)**. 

Está diseñado para correr en plataformas de hosting gratuitas como **Streamlit Community Cloud** sin necesidad de bases de datos externas ni credenciales de APIs.

## ¿Qué contiene este demo?
- **Motor de Simulación Dinámico:** Simula shocks (como sequías, bajantes del río o paros de transporte) día a día.
- **Modelo de Machine Learning Persistido:** El motor y las reglas de comportamiento se cargan desde una base de datos local SQLite (`data/bcp_models.db`) previamente calibrada por el equipo de economía.
- **Gráficos Interactivos:** Proyecciones temporales y bandas de confianza probabilísticas (Monte Carlo).
- **Módulo de Reportes:** Generación de un reporte ejecutivo descargable en base a los escenarios simulados.

## Instrucciones para desplegar gratis en Streamlit Cloud:

1. **Crear un nuevo repositorio en tu GitHub** (ej: `simulador-bcp-demo`).
2. **Subir todos los archivos** de esta carpeta a ese repositorio.
3. Ir a **[share.streamlit.io](https://share.streamlit.io)** e iniciar sesión con tu cuenta de GitHub.
4. Presionar el botón **"Create app"** (o "New app").
5. Configurar los parámetros:
   - **Repository:** `tu-usuario/simulador-bcp-demo`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Presionar **"Deploy!"**. En aproximadamente 2 minutos la aplicación estará en línea y lista para ser compartida mediante una URL pública con el jurado.

## Ejecución Local
Si querés probar la aplicación de forma local, simplemente corré en tu consola:
```bash
pip install -r requirements.txt
streamlit run app.py
```

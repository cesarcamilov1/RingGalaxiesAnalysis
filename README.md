# RingGalaxiesAnalysis

# ***Proyecto Detección de Barras y Anillos.***

**Equipo 39**

*Integrantes:*

Cesar Benjamín Nájera Camilo - A01796663

Luis Ángel García Mata - A01795985

Franco Quintanilla Fuentes - A00826953

## Objetivos del Proyecto

El problema se aborda en dos tareas principales:
1. **Tarea 1 (Clasificación binaria):** Distinguir entre galaxias **sin anillo** y galaxias con **anillo interno** (con o sin componente externo).
2. **Tarea 2 (Regresión):** Estimar propiedades físicas como la **distancia del anillo al centro de la galaxia** ($r_{ring}$).

## Estructura del Proyecto

- `catalogs/` - Contiene los catálogos base utilizados para el etiquetado y los datasets resultantes.
- `Data-Download/` - Scripts y cuadernos para la obtención de datos:
  - `ClasificationData.ipynb`: Filtrado y combinación de catálogos, mapeo de etiquetas (sin anillo vs. anillo interno) y creación del dataset final para la tarea de clasificación.
  - `RingDataDownload.ipynb`: Generación de recortes (cutouts) automatizados y descarga de imágenes astronómicas en formato FITS directamente de `legacysurvey.org`.
- `EDA/` - Análisis Exploratorio de Datos:
  - `eda_exploratory_analysis.ipynb`: Análisis profundo de distribuciones geométricas (RA, DEC), corrimiento al rojo (redshift $z$), desbalance de clases y definición de la estrategia de preprocesamiento.
- `data/` - *(Ignorada en git)* Directorio local donde se almacenan las imágenes FITS descargadas organizadas por clase.

## Configuración y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd "proyecto integrador"
   ```

2. **Crear e inicializar el entorno virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias necesarias:**
   Se requieren librerías como `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `astropy`, `requests` y `scikit-learn` para ejecutar los notebooks actuales.

4. **Ejecutar el Pipeline:**
   Se recomienda seguir el orden de los notebooks:
   1. `Data-Download/ClasificationData.ipynb`
   2. `EDA/eda_exploratory_analysis.ipynb`
   3. `Data-Download/RingDataDownload.ipynb`

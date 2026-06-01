"""
Baseline Geométrico Clásico — Clasificación de galaxias con/sin anillo.

Pipeline (sin Deep Learning ni ML, sólo geometría + fotometría):
    Fase 1 — Muestreo balanceado y carga de la banda r (FITS cubo grz, idx=1).
    Fase 2 — Resta de fondo, escalado arcsinh, recorte de percentiles.
    Fase 3 — Perfil radial vectorizado + scipy.signal.find_peaks.
    Fase 4 — Confusion matrix + accuracy + figura de diagnóstico.

Dependencias: astropy, numpy, scipy, matplotlib, pandas, scikit-learn (sólo métricas).
Uso:
    python geometric_baseline.py
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pdPer
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d, label as nd_label
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, confusion_matrix


# ----------------------------------------------------------------------------
# Configuración global
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data_download" / "classification_dataset.csv"
DATA_DIR = ROOT / "data_download" / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

R_BAND_INDEX = 1            # cubo (g, r, z) descargado de Legacy Survey
N_PER_CLASS = 100           # muestreo balanceado por clase
RANDOM_SEED = 42
CENTROID_SEARCH_FRAC = 0.4  # buscar Xc,Yc dentro del 40% central
BG_BORDER_FRAC = 0.10       # 10% más externo = ruido de cielo
PERCENTILE_LO = 1.0
PERCENTILE_HI = 99.5

# Reglas geométricas para "anillo"
PEAK_MIN_DISTANCE = 3         # separación mínima entre picos (px)
PEAK_SMOOTH_SIGMA = 2.0       # suavizado gaussiano del perfil antes de find_peaks (px)
PEAK_PROMINENCE_FRAC = 0.02   # prominencia mín. como fracción del flujo central
GAL_EDGE_THRESH = 0.02        # define el borde de la galaxia: I(r_gal) = bg + 2%·(I0-bg)
PEAK_R_MIN_FRAC = 0.15        # los picos válidos están entre 15% y 100% de r_gal
MASK_K_SIGMA = 4.0            # umbral (MAD) para identificar fuentes brillantes de campo
SHOULDER_SMOOTH_SIGMA = 2.5   # suavizado de la derivada logarítmica
SHOULDER_PROMINENCE_K = 1.5   # prominencia mín. de hombros en unidades de std(d ln I/dr)


# ----------------------------------------------------------------------------
# Fase 1 — Muestreo y carga
# ----------------------------------------------------------------------------
def sample_balanced_control(csv_path: Path, n_per_class: int, seed: int) -> pd.DataFrame:
    """Devuelve una muestra balanceada (n_per_class por clase) del catálogo."""
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    parts = []
    for lbl in (0, 1):
        sub = df[df["label"] == lbl]
        take = min(n_per_class, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_name_to_path(data_dir: Path) -> dict[str, Path]:
    """Mapea name (id SDSS) → ruta FITS, parseando el nombre de archivo una vez."""
    pat = re.compile(r"_(\d+)_ra")
    mapping: dict[str, Path] = {}
    for f in data_dir.iterdir():
        if f.suffix.lower() != ".fits":
            continue
        m = pat.search(f.name)
        if m:
            mapping[m.group(1)] = f
    return mapping


def load_r_band(fits_path: Path) -> np.ndarray:
    """Carga la banda r (float64) desde un cubo FITS (3, H, W)."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"HDU primario sin datos: {fits_path.name}")
        if data.ndim == 3:
            if data.shape[0] < R_BAND_INDEX + 1:
                raise ValueError(f"Cubo con menos bandas de lo esperado: {data.shape}")
            band = data[R_BAND_INDEX]
        elif data.ndim == 2:
            band = data  # ya es 2D, asumimos r
        else:
            raise ValueError(f"Dimensión no soportada {data.ndim} en {fits_path.name}")
        return np.asarray(band, dtype=np.float64)


def find_centroid(img: np.ndarray, search_frac: float = CENTROID_SEARCH_FRAC) -> tuple[int, int]:
    """Píxel de flujo máximo dentro de una caja central. Retorna (yc, xc)."""
    h, w = img.shape
    dy = int(h * search_frac / 2)
    dx = int(w * search_frac / 2)
    y0, y1 = h // 2 - dy, h // 2 + dy
    x0, x1 = w // 2 - dx, w // 2 + dx
    sub = img[y0:y1, x0:x1]
    # argmax sobre la región central; NaN-safe
    flat_idx = np.nanargmax(sub)
    ly, lx = np.unravel_index(flat_idx, sub.shape)
    return y0 + int(ly), x0 + int(lx)


# ----------------------------------------------------------------------------
# Fase 2 — Preprocesamiento astrofísico
# ----------------------------------------------------------------------------
def subtract_background(img: np.ndarray, border_frac: float = BG_BORDER_FRAC) -> np.ndarray:
    """Resta la mediana del cielo estimada con un anillo periférico."""
    h, w = img.shape
    by, bx = int(h * border_frac), int(w * border_frac)
    mask = np.ones_like(img, dtype=bool)
    mask[by:h - by, bx:w - bx] = False  # True sólo en el borde
    sky = np.nanmedian(img[mask])
    return img - sky


def asinh_stretch(img: np.ndarray) -> np.ndarray:
    """Estira el rango dinámico con arcsinh — núcleo comprimido, anillos visibles."""
    # Escalamos por la sigma robusta para no aplastar todo a ~0 cuando el flujo es alto.
    scale = np.nanstd(img) + 1e-12
    return np.arcsinh(img / scale)


def percentile_clip(img: np.ndarray,
                    lo: float = PERCENTILE_LO,
                    hi: float = PERCENTILE_HI) -> np.ndarray:
    """Recorta al rango [p_lo, p_hi] y normaliza a [0, 1]."""
    p_lo, p_hi = np.nanpercentile(img, [lo, hi])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(img)
    out = np.clip(img, p_lo, p_hi)
    return (out - p_lo) / (p_hi - p_lo)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Pipeline completo: fondo → arcsinh → recorte percentiles."""
    x = subtract_background(img)
    x = asinh_stretch(x)
    x = percentile_clip(x)
    return x


def mask_field_sources(img: np.ndarray, yc: int, xc: int,
                       k_sigma: float = MASK_K_SIGMA) -> np.ndarray:
    """NaN-ea píxeles brillantes NO conectados a la galaxia central.

    Estrellas y galaxias vecinas inflan el promedio anular en su radio →
    producen picos espurios en el perfil. Las descartamos vía componentes
    conectados sobre una máscara por umbral robusto (MAD).
    """
    med = float(np.nanmedian(img))
    mad = float(np.nanmedian(np.abs(img - med)))
    sigma = 1.4826 * mad + 1e-12
    bright = img > med + k_sigma * sigma
    labels, _ = nd_label(bright)
    gal = int(labels[yc, xc])
    if gal == 0:
        return img  # el centroide no cayó sobre un objeto brillante
    drop = (labels > 0) & (labels != gal)
    out = img.copy()
    out[drop] = np.nan
    return out


# ----------------------------------------------------------------------------
# Fase 3 — Perfil radial e identificación de anillos
# ----------------------------------------------------------------------------
def radial_profile(img: np.ndarray, yc: int, xc: int) -> tuple[np.ndarray, np.ndarray]:
    """Calcula <flujo>(r) agrupando píxeles por radio entero. 100% vectorizado."""
    h, w = img.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - xc, yy - yc).astype(np.int32).ravel()
    v = img.ravel()
    # Limita el radio al máximo que cabe completo en la imagen (evita anillos truncados)
    r_max = min(yc, h - 1 - yc, xc, w - 1 - xc)
    valid = (r <= r_max) & np.isfinite(v)  # ignora NaN del enmascarado
    r, v = r[valid], v[valid]

    sums = np.bincount(r, weights=v, minlength=r_max + 1)
    counts = np.bincount(r, minlength=r_max + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / counts, 0.0)
    radii = np.arange(r_max + 1)
    return radii, profile


def detect_rings(radii: np.ndarray, profile: np.ndarray) -> tuple[int, np.ndarray, dict]:
    """Detecta picos/anillos dentro de la extensión real de la galaxia.

    Estrategia:
      1) Suaviza el perfil con Gaussiana (PEAK_SMOOTH_SIGMA) para domar ruido.
      2) Estima el nivel de cielo en el 20% externo del perfil.
      3) Define r_gal donde I(r) cae al `bg + GAL_EDGE_THRESH·(I0-bg)`.
      4) Busca picos sólo en [PEAK_R_MIN_FRAC·r_gal, r_gal] con prominencia
         escalada al flujo central (no a la varianza del cielo).
    """
    if len(profile) < 8:
        return 0, np.array([], dtype=int), {}

    sm = gaussian_filter1d(profile, sigma=PEAK_SMOOTH_SIGMA)
    i0 = float(sm[0])
    tail_start = max(int(0.8 * len(sm)), 1)
    bg = float(np.nanmedian(sm[tail_start:]))
    contrast = i0 - bg
    if contrast <= 0:
        return 0, np.array([], dtype=int), {"reason": "no central contrast"}

    # Borde de la galaxia: último radio por encima del umbral
    threshold = bg + GAL_EDGE_THRESH * contrast
    above = np.where(sm > threshold)[0]
    if len(above) < 5:
        return 0, np.array([], dtype=int), {"reason": "galaxy too compact"}
    r_gal = int(above[-1])

    r_min = max(int(PEAK_R_MIN_FRAC * r_gal), 3)
    if r_min >= r_gal - 2:
        return 0, np.array([], dtype=int), {"reason": "window collapsed"}

    # --- Señal A: picos literales del perfil (anillos externos brillantes) ---
    prominence = max(PEAK_PROMINENCE_FRAC * contrast, 1e-6)
    peaks_a, _ = find_peaks(sm, distance=PEAK_MIN_DISTANCE, prominence=prominence)
    peaks_a = peaks_a[(peaks_a >= r_min) & (peaks_a <= r_gal)]

    # --- Señal B: hombros vía derivada logarítmica (anillos internos/barras) ---
    # Para un disco exponencial puro, d ln(I)/dr es constante (= -1/h).
    # Un anillo crea un máximo local: la pendiente decreciente "se afloja"
    # momentáneamente. find_peaks sobre dlog detecta esa inflexión.
    eps = 1e-6 * contrast
    log_I = np.log(np.maximum(sm - bg, eps))
    dlog = np.gradient(log_I)
    dlog = gaussian_filter1d(dlog, sigma=SHOULDER_SMOOTH_SIGMA)
    window = dlog[r_min:r_gal + 1]
    if window.size >= 5:
        sigma_dlog = float(np.nanstd(window))
        prom_b = max(SHOULDER_PROMINENCE_K * sigma_dlog, 1e-6)
        rel_peaks, _ = find_peaks(window, distance=PEAK_MIN_DISTANCE, prominence=prom_b)
        peaks_b = rel_peaks + r_min
    else:
        peaks_b = np.array([], dtype=int)

    # Unión (cualquiera de las dos señales = anillo)
    if len(peaks_a) + len(peaks_b) > 0:
        peaks = np.unique(np.concatenate([peaks_a, peaks_b]))
    else:
        peaks = np.array([], dtype=int)

    pred = 1 if len(peaks) > 0 else 0
    info = {"prominence": prominence, "r_gal": r_gal, "r_min": r_min,
            "bg": bg, "i0": i0, "smoothed": sm, "dlog": dlog,
            "peaks_a": peaks_a, "peaks_b": peaks_b}
    return pred, peaks, info


# ----------------------------------------------------------------------------
# Pipeline por galaxia
# ----------------------------------------------------------------------------
@dataclass
class GalaxyResult:
    name: str
    label: int
    pred_label: int
    n_peaks: int
    error: str | None = None
    img_proc: np.ndarray | None = None
    radii: np.ndarray | None = None
    profile: np.ndarray | None = None
    smoothed: np.ndarray | None = None
    dlog: np.ndarray | None = None
    peaks: np.ndarray | None = None
    peaks_a: np.ndarray | None = None
    peaks_b: np.ndarray | None = None
    r_gal: int | None = None
    r_min: int | None = None


def classify_one(name: str, label: int, fits_path: Path) -> GalaxyResult:
    """Ejecuta el pipeline completo en una galaxia. Tolera fallos."""
    try:
        raw = load_r_band(fits_path)
        if not np.any(np.isfinite(raw)):
            raise ValueError("imagen completamente NaN")
        yc, xc = find_centroid(raw)
        proc = preprocess(raw)
        masked = mask_field_sources(proc, yc, xc)
        radii, prof = radial_profile(masked, yc, xc)
        pred, peaks, info = detect_rings(radii, prof)
        return GalaxyResult(
            name=name, label=label, pred_label=pred, n_peaks=int(len(peaks)),
            img_proc=masked, radii=radii, profile=prof,
            smoothed=info.get("smoothed"), dlog=info.get("dlog"),
            peaks=peaks, peaks_a=info.get("peaks_a"), peaks_b=info.get("peaks_b"),
            r_gal=info.get("r_gal"), r_min=info.get("r_min"),
        )
    except Exception as e:
        return GalaxyResult(
            name=name, label=label, pred_label=0, n_peaks=0,
            error=f"{type(e).__name__}: {e}",
        )


# ----------------------------------------------------------------------------
# Fase 4 — Evaluación y diagnóstico visual
# ----------------------------------------------------------------------------
def plot_diagnostic(result: GalaxyResult, out_path: Path) -> None:
    """Guarda figura: imagen procesada (izq) + perfil radial con picos (der)."""
    if result.img_proc is None:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Mostramos la imagen enmascarada (NaN se ve como hueco), con NaN → 0 al pintar
    display = np.where(np.isfinite(result.img_proc), result.img_proc, 0.0)
    ax1.imshow(display, origin="lower", cmap="magma")
    ax1.set_title(f"{result.name} — label={result.label} pred={result.pred_label}")
    ax1.set_xlabel("x [px]"); ax1.set_ylabel("y [px]")

    # Panel 2: perfil radial + picos literales (señal A)
    ax2.plot(result.radii, result.profile, lw=1.0, alpha=0.5, label="⟨flujo⟩(r) bruto")
    if result.smoothed is not None:
        ax2.plot(result.radii, result.smoothed, lw=1.6, label="suavizado")
    if result.r_min is not None and result.r_gal is not None:
        ax2.axvspan(result.r_min, result.r_gal, alpha=0.12, color="green",
                    label=f"ventana [{result.r_min},{result.r_gal}]")
    if result.peaks_a is not None and len(result.peaks_a):
        src = result.smoothed if result.smoothed is not None else result.profile
        ax2.plot(result.radii[result.peaks_a], src[result.peaks_a],
                 "rv", ms=9, label=f"picos A ({len(result.peaks_a)})")
    ax2.set_xlabel("radio [px]"); ax2.set_ylabel("flujo medio (norm.)")
    ax2.set_title("Perfil radial — picos literales")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

    # Panel 3: derivada logarítmica + hombros (señal B)
    if result.dlog is not None:
        ax3.plot(result.radii, result.dlog, lw=1.4, color="C2", label="d ln(I)/dr")
        if result.r_min is not None and result.r_gal is not None:
            ax3.axvspan(result.r_min, result.r_gal, alpha=0.12, color="green")
            ax3.set_xlim(0, result.r_gal * 1.5)
        if result.peaks_b is not None and len(result.peaks_b):
            ax3.plot(result.radii[result.peaks_b], result.dlog[result.peaks_b],
                     "b^", ms=9, label=f"hombros B ({len(result.peaks_b)})")
    ax3.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax3.set_xlabel("radio [px]"); ax3.set_ylabel("d ln(I) / dr")
    ax3.set_title("Derivada logarítmica — inflexiones")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def report(results: list[GalaxyResult]) -> pd.DataFrame:
    """Imprime matriz de confusión y accuracy; retorna DataFrame de resultados."""
    df = pd.DataFrame([{
        "name": r.name, "label": r.label, "pred_label": r.pred_label,
        "n_peaks": r.n_peaks, "error": r.error,
    } for r in results])

    ok = df[df["error"].isna()]
    if ok.empty:
        print("\n[!] No se pudo clasificar ninguna galaxia.")
        return df

    y_true, y_pred = ok["label"].to_numpy(), ok["pred_label"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print(f"Galaxias procesadas: {len(ok)} / {len(df)}  (errores: {df['error'].notna().sum()})")
    print("-" * 60)
    print(f"Matriz de confusión [filas=real, cols=pred]\n{cm}")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"Accuracy global: {acc:.4f}")
    if tp + fp > 0: print(f"Precisión (clase 1): {tp / (tp + fp):.4f}")
    if tp + fn > 0: print(f"Recall    (clase 1): {tp / (tp + fn):.4f}")
    print("=" * 60)
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    print(f"[1/4] Muestra balanceada ({N_PER_CLASS} por clase, seed={RANDOM_SEED})")
    sample = sample_balanced_control(CSV_PATH, N_PER_CLASS, RANDOM_SEED)
    print(f"      → {len(sample)} galaxias seleccionadas")

    print(f"[2/4] Indexando FITS en {DATA_DIR}")
    name_to_path = build_name_to_path(DATA_DIR)
    print(f"      → {len(name_to_path)} archivos FITS encontrados")

    print(f"[3/4] Procesando {len(sample)} galaxias…")
    results: list[GalaxyResult] = []
    missing = 0
    for _, row in sample.iterrows():
        name = str(row["name"])
        label = int(row["label"])
        path = name_to_path.get(name)
        if path is None:
            missing += 1
            results.append(GalaxyResult(name, label, 0, 0, error="FITS no encontrado"))
            continue
        results.append(classify_one(name, label, path))
    print(f"      → faltantes en disco: {missing}")

    print("[4/4] Métricas y figura de diagnóstico")
    df_out = report(results)
    df_out.to_csv(OUT_DIR / "baseline_predictions.csv", index=False)
    print(f"      predicciones → {OUT_DIR / 'baseline_predictions.csv'}")

    # Una figura de diagnóstico por clase (la primera exitosa de cada una)
    for lbl in (0, 1):
        ex = next((r for r in results if r.label == lbl and r.error is None), None)
        if ex:
            out = OUT_DIR / f"diagnostic_label{lbl}_{ex.name}.png"
            plot_diagnostic(ex, out)
            print(f"      figura → {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

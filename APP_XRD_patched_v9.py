#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gonio viewer — filtres + couleurs (légende dynamique + anneal filter)
# Dépendances: numpy, matplotlib, pandas, tkinter

from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from itertools import cycle
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, filedialog
import sys
import warnings
import time
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button as MplButton


_UNIT_RE = re.compile(r"^\s*(?P<name>.*?)\s*\[(?P<unit>.+?)\]\s*$")

def split_col_and_unit(colname: str):
    """
    'PbI2 [mol/L]' -> ('PbI2', 'mol/L')
    'annealing[min]' -> ('annealing', 'min')  (si tu mets un espace ça marche aussi)
    sinon -> ('colname', None)
    """
    s = str(colname).strip()
    m = _UNIT_RE.match(s)
    if not m:
        return s, None
    return m.group("name").strip(), m.group("unit").strip()




# -------------------- Réglages généraux --------------------
GLOB_PATTERN = "*.xy"
LINEWIDTH = 1.2
ALPHA = 0.95
TITLE = "Gonio scans"
XLABEL = r"2$\theta$ (°)"
YLABEL = "Intensity (a.u.)"
MAX_LEGEND = 40  # > 15 courbes -> pas de légende matplotlib

# -------------------- Excel metadata (optionnel) --------------------
METADATA_XLSX = "metadata.xlsx"
METADATA_SHEET = "FILES"   # onglet
STRICT_METADATA = True          # exige une ligne Excel pour chaque fichier .xy
REQUIRED_META_COLS = ["sample_id"]  # champs obligatoires non vides (tu peux en ajouter)


# -------------------- Phase quantification (template → batch) --------------------
# ITO offset table (2 colonnes: 2θ, intensité). Mettre ITO.txt dans le même dossier que metadata.xlsx
ITO_TABLE_FILE = "ITO.txt"

# Fenêtres (°2θ) où l’ITO a des pics bien marqués (robuste pour estimer l’offset)
# (ajuste si ton ITO / scan a des bornes différentes)
ITO_WINDOWS = [(20.5, 21.5), (29.0, 31.0), (34.0, 36.0), (50.0, 51.5), (59.5, 61.0)]

# Liste de phases proposées dans le menu déroulant (tu peux compléter)
PHASE_CHOICES = [
    "α-FAPI", "δ-FAPI", "PbI₂", "FAI",
    "PbI₂·DMSO", "PbI₂·DMF", "PbI₂·2DMSO","4H", "6H",
    "ITO", "Autre/Unknown"
]

PHASE_COLOR_MAP = {
    "α-FAPI": "#1f77b4",
    "δ-FAPI": "#ff7f0e",
    "PbI₂":   "#2ca02c",
    "FAI":    "#9467bd",
    "PbI₂·DMSO": "#8c564b",
    "PbI₂·DMF":  "#e377c2",
    "PbI₂·2DMSO":"#7f7f7f",
    "4H": "#bcbd22",
    "6H": "#d62728",
    "ITO":    "#17becf",
    "Autre/Unknown": "#444444",
}

def phase_color_for(phase: str) -> str:
    return PHASE_COLOR_MAP.get(str(phase), "#444444")


# --- Matching candidates (comme Plot_reasearch_pics_XY_FILES_CLEAN / fenêtre 4) ---
MATCH_TOL_DEG = 1.0   # rayon de recherche ± autour du pic (° 2θ)
W_DIST = 0.5          # poids de la distance (↑ = privilégie proximité)
W_I    = 0.5          # poids de l'intensité théorique (↑ = privilégie I_th)
P_DIST = 1.0          # exponent sur distance (1=linéaire, >1 pénalise plus vite)
MAX_CANDIDATES = 6    # nb d’alternatives proposées dans la liste

# (fichier, label_phase) — tous les fichiers sont optionnels (si absent: ignoré)
PHASE_REF_FILES = [
    ("alpha_phase_FAPI_cubic.txt", "α-FAPI"),
    ("delta_phase_FAPI.txt",       "δ-FAPI"),
    ("PbI2.txt",                   "PbI₂"),
    ("FAI.txt",                    "FAI"),
    ("PbI2_DMSO.txt",              "PbI₂·DMSO"),
    ("PbI2_DMF.txt",               "PbI₂·DMF"),
    ("PbI2·2DMSO.txt",             "PbI₂·2DMSO"),
    ("PbI2_2DMSO.txt",             "PbI₂·2DMSO"),  # fallback (certains OS n'aiment pas '·')
    ("4H.txt",                     "4H"),
    ("6H.txt",                     "6H"),
    ("ITO.txt",                    "ITO"),
]

def match_score(delta_deg, I_th, I_th_max, tol_deg, w_dist, w_I, p_dist):
    """Score à MINIMISER combinant proximité et intensité théorique."""
    d_norm = min(abs(float(delta_deg)) / max(float(tol_deg), 1e-12), 1.0)
    I_norm = float(I_th) / max(float(I_th_max), 1e-12)
    return w_dist * (d_norm ** p_dist) + w_I * (1.0 - I_norm)

def read_phase_table_vesta(path: Path):
    """Parse un fichier de raies théoriques type VESTA / FullProf (>=9 colonnes).
    On attend typiquement: h k l ... ... ... ... 2theta I
    Retour: (tth, I, hkl_str) arrays.
    """
    tths, ints, hkls = [], [], []
    for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.replace(",", " ").replace(";", " ").split()
        if len(toks) < 9:
            continue
        try:
            tt = float(toks[7]); inten = float(toks[8])
        except Exception:
            continue
        hkl = ""
        try:
            hkl = f"({toks[0]} {toks[1]} {toks[2]})"
        except Exception:
            hkl = ""
        tths.append(tt); ints.append(inten); hkls.append(hkl)
    if len(tths) == 0:
        return np.zeros(0, float), np.zeros(0, float), []
    return np.asarray(tths, float), np.asarray(ints, float), hkls

def read_ito_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lit une table ITO *théorique*.

    Supporte 2 formats (détectés automatiquement) :
      1) 2 colonnes : `2theta  intensity` (type .xy)
      2) format "VESTA / raies" : au moins 9 colonnes numériques où
         `tokens[7] = 2theta` et `tokens[8] = intensity` (comme tes fichiers de phases).

    Retour:
      (x_ito, y_ito) triés par 2θ.
    """
    path = Path(path)

    rows: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if (not s) or s.startswith(("#", "%", "//")):
                continue
            s = s.replace(",", " ").replace(";", " ")
            toks = s.split()
            nums: list[float] = []
            for t in toks:
                try:
                    nums.append(float(t))
                except Exception:
                    break  # stop at first non-numeric token
            if len(nums) >= 9:
                rows.append((float(nums[7]), float(nums[8])))
            elif len(nums) >= 2:
                rows.append((float(nums[0]), float(nums[1])))

    if not rows:
        raise ValueError(f"ITO: aucune donnée lisible dans {path}")

    arr = np.asarray(rows, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    o = np.argsort(x)
    return np.asarray(x)[o], np.asarray(y)[o]


def _peak_in_window(x: np.ndarray, y: np.ndarray, w: tuple[float,float]) -> tuple[float,float] | None:
    a, b = w
    m = (x >= a) & (x <= b)
    if not np.any(m):
        return None
    xx = x[m]; yy = y[m]
    i = int(np.argmax(yy))
    return float(xx[i]), float(yy[i])

def estimate_ito_offset_deg(x_meas: np.ndarray, y_meas: np.ndarray,
                            x_ito: np.ndarray, y_ito: np.ndarray,
                            windows=ITO_WINDOWS,
                            max_shift_deg: float = 0.6) -> float:
    """
    Estime l’offset en °2θ : offset = (pic_mesuré - pic_ITO_théorique).

    Version robuste:
    - essaie d'abord une corrélation (pattern matching) dans plusieurs fenêtres ITO
      (marche même si le pic est faible / un peu masqué),
    - sinon fallback sur détection de max dans la fenêtre.
    Retour:
      float (peut être 0.0 si parfaitement aligné). Si rien n’est exploitable: np.nan.
    """
    x_meas = np.asarray(x_meas, float)
    y_meas = np.asarray(y_meas, float)
    x_ito  = np.asarray(x_ito,  float)
    y_ito  = np.asarray(y_ito,  float)

    def _zscore(a):
        a = np.asarray(a, float)
        a = a - np.nanmean(a)
        s = np.nanstd(a)
        return a / s if (s and np.isfinite(s)) else a

    def _corr_shift(xm, ym, xt, yt, a, b):
        # extrait fenêtre
        mskm = (xm >= a) & (xm <= b)
        mskt = (xt >= a) & (xt <= b)
        if np.sum(mskm) < 10 or np.sum(mskt) < 10:
            return None

        xm_w = xm[mskm]; ym_w = ym[mskm]
        xt_w = xt[mskt]; yt_w = yt[mskt]

        # grille commune: pas ~ pas mesuré
        dx = float(np.median(np.diff(xm_w)))
        if not np.isfinite(dx) or dx <= 0:
            return None
        dx = max(dx, 0.002)

        # limite de shift en indices
        K = int(round(max_shift_deg / dx))
        if K < 1:
            return None

        grid = np.arange(a, b + dx/2, dx)
        if grid.size < 20:
            return None

        ym_i = np.interp(grid, xm_w, ym_w)
        yt_i = np.interp(grid, xt_w, yt_w)

        # pré-traitement léger: enlève offset + normalize
        ym_i = _zscore(ym_i)
        yt_i = _zscore(yt_i)

        best_k = None
        best_c = -np.inf

        # corrélation par recouvrement (pas de wrap)
        for k in range(-K, K + 1):
            if k == 0:
                a1 = ym_i; b1 = yt_i
            elif k > 0:
                # ITO shifté à droite: comparer ym[k:] avec yt[:-k]
                a1 = ym_i[k:]
                b1 = yt_i[:-k]
            else:
                kk = -k
                a1 = ym_i[:-kk]
                b1 = yt_i[kk:]
            if a1.size < 10:
                continue
            c = float(np.dot(a1, b1) / a1.size)
            if c > best_c:
                best_c = c
                best_k = k

        if best_k is None:
            return None

        # seuil de corrélation (évite de “matcher” du bruit)
        if not np.isfinite(best_c) or best_c < 0.15:
            return None

        return float(best_k * dx)

    # 1) corrélation (robuste)
    deltas = []
    for (a, b) in windows:
        d = _corr_shift(x_meas, y_meas, x_ito, y_ito, a, b)
        if d is not None and np.isfinite(d):
            deltas.append(d)

    # 2) fallback peaks si corrélation insuffisante
    if len(deltas) < 2:
        for w in windows:
            p_meas = _peak_in_window(x_meas, y_meas, w)
            p_theo = _peak_in_window(x_ito, y_ito, w)
            if (p_meas is None) or (p_theo is None):
                continue
            xm, _ = p_meas
            xt, _ = p_theo
            d = float(xm - xt)
            if np.isfinite(d):
                deltas.append(d)

    if not deltas:
        return float("nan")

    d = np.asarray(deltas, float)

    # médiane robuste + rejet MAD si assez de fenêtres
    med = float(np.median(d))
    if len(d) >= 3:
        mad = float(np.median(np.abs(d - med)))
        if mad > 1e-6:
            keep = np.abs(d - med) <= 3.5 * 1.4826 * mad
            if np.any(keep):
                med = float(np.median(d[keep]))

    return med

# --- Voigt (copié/adapté de ton Plot_research) ---
def _colvec(a):
    return np.asarray(a).reshape(-1)

def pseudo_voigt_normalized(x, x0, fwhm, eta):
    x = _colvec(x)
    fwhm = max(float(fwhm), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    sigma = fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
    G = np.exp(-(x - x0)**2 / (2.0*sigma**2)) / (sigma*np.sqrt(2.0*np.pi))  # aire=1
    L = (0.5*fwhm/np.pi) / ((x - x0)**2 + (0.5*fwhm)**2)                    # aire=1
    return eta*L + (1.0 - eta)*G

def _estimate_linear_baseline(x, y, edge_frac=0.20, min_edge_pts=5,
                              use_savgol=True, sg_window_pts=7, sg_poly=2,
                              method="percentile_edges", q=20.0):
    x = _colvec(x); y = _colvec(y)
    n = len(x)
    if n < 7:
        return np.zeros_like(x)
    y_in = y
    if use_savgol and n >= sg_window_pts and sg_window_pts % 2 == 1:
        try:
            y_in = savgol_filter(y, sg_window_pts, sg_poly, mode="interp")
        except Exception:
            y_in = y
    k = max(min_edge_pts, int(edge_frac * n))
    if 2 * k >= n:
        k = max(3, n // 4)
    xL, yL = x[:k], y_in[:k]
    xR, yR = x[-k:], y_in[-k:]
    if method == "percentile_edges":
        yL0 = float(np.percentile(yL, q))
        yR0 = float(np.percentile(yR, q))
        x1, x2 = float(np.mean(xL)), float(np.mean(xR))
        if x2 == x1:
            a, b = 0.0, yL0
        else:
            a = (yR0 - yL0) / (x2 - x1)
            b = yL0 - a * x1
        return a * x + b
    Xb = np.concatenate([xL, xR])
    Yb = np.concatenate([yL, yR])
    A = np.vstack([Xb, np.ones_like(Xb)]).T
    a, b = np.linalg.lstsq(A, Yb, rcond=None)[0]
    return a * x + b

def _deg_to_samples(x_deg, delta_deg):
    """Convertit une distance en degrés vers une distance en indices (approx. médiane Δx)."""
    x = np.asarray(x_deg, float)
    if len(x) < 2:
        return 1
    dx = np.diff(x)
    med_dx = np.median(dx[dx > 0]) if np.any(dx > 0) else (x[-1] - x[0]) / max(len(x) - 1, 1)
    return max(int(round(delta_deg / max(med_dx, 1e-9))), 1)
def estimate_bg_envelope(x, y, win_pts=151, q=0.5, sg_win=101, sg_poly=2):
    y = np.asarray(y, float); n = len(y)
    win = max(5, int(win_pts) | 1)
    qwin = max(3, win//2)
    # quantile roulant naïf (vectorisé simple)
    ypad = np.r_[y[qwin-1:0:-1], y, y[-2:-qwin-2:-1]]
    out = np.empty(n)
    for i in range(n):
        seg = ypad[i:i+2*qwin+1]
        out[i] = np.quantile(seg, q)
    # lissage doux
    sg_win = min(max(5, int(sg_win) | 1), (n//2)*2+1)
    try:
        out = savgol_filter(out, sg_win, sg_poly)
    except Exception:
        pass
    return out
def make_envelope_baseline(x, y, deg_window=1.2, q=0.25, sg_poly=2):
    """Enveloppe bas = quantile roulant (q) converti en points depuis deg_window, puis SG."""
    # même conversion ° → points que dans le DEBUG
    win_pts = max(51, _deg_to_samples(x, deg_window) * 2 + 1)  # impaire
    sg_win  = win_pts
    return estimate_bg_envelope(x, y, win_pts=win_pts, q=q, sg_win=sg_win, sg_poly=sg_poly)


def _initial_fwhm_from_window(x):
    return max((np.max(x) - np.min(x)) * 0.18, 0.06)

def _fit_voigt_area_with_given_baseline(xloc, yloc, yb_vec):
    xloc = _colvec(xloc); yloc = _colvec(yloc); yb = _colvec(yb_vec)
    n = len(xloc)
    if n < 7 or len(yb) != n:
        return dict(A=0.0, x0=np.nan, fwhm=np.nan, eta=np.nan,
                    yfit=np.zeros_like(yloc), y_base=np.array(yb, copy=True) if len(yb)==n else np.zeros_like(yloc),
                    ok=False)
    ycorr = np.clip(yloc - yb, 0.0, None)
    x0_init = float(xloc[np.argmax(ycorr)])
    fwhm_init = _initial_fwhm_from_window(xloc)
    eta_init = 0.5
    A_init = max(np.trapz(ycorr, xloc), 1e-6)

    def model(x, x0, fwhm, eta, A):
        return A * pseudo_voigt_normalized(x, x0, fwhm, eta)

    bounds = ([xloc.min(), 0.02, 0.0, 0.0],
              [xloc.max(), 2.00, 1.0, np.inf])
    p0 = [x0_init, fwhm_init, eta_init, A_init]
    try:
        popt, _ = curve_fit(model, xloc, ycorr, p0=p0, bounds=bounds, maxfev=4000)
        x0, fwhm, eta, A = popt
        yfit = model(xloc, x0, fwhm, eta, A)
        return dict(A=float(A), x0=float(x0), fwhm=float(fwhm), eta=float(eta),
                    yfit=yfit, y_base=np.array(yb, copy=True), ok=True)
    except Exception:
        return dict(A=0.0, x0=x0_init, fwhm=fwhm_init, eta=eta_init,
                    yfit=np.zeros_like(yloc), y_base=np.array(yb, copy=True), ok=False)

META_OVERWRITE = True           # Excel prioritaire sur les colonnes communes (recommandé avec STRICT)

# Colonnes "réservées" = pas des paramètres
RESERVED_COLS = {
    "filename", "path", "path_abs", "path_rel", "path_key",
    "sample_id", "cycle_pair",
    "_group", "_group_sort", "_sample_sort"
}

# Unités (pour l'affichage)
UNITS = {
    "PbI2": "mol/L",
    "FAI": "mg/ml",
    "OFN": "mg/ml",
    "anneal": "min",
}

def _norm_path_key(p: str) -> str:
    """Normalise un chemin en clé comparable (slashes, strip)."""
    if p is None:
        return ""
    p = str(p).strip().replace("\\", "/")
    return p

def load_metadata_table(root: Path) -> pd.DataFrame | None:
    """
    Charge metadata.xlsx si présent.
    Attendu: onglet FILES avec colonne path (recommandé) ou filename.
    """
    xlsx = root / METADATA_XLSX
    if not xlsx.exists():
        return None

    try:
        meta = pd.read_excel(xlsx, sheet_name=METADATA_SHEET)
    except Exception:
        # fallback: premier onglet
        meta = pd.read_excel(xlsx)

    # Normalise noms de colonnes
    meta.columns = [str(c).strip() for c in meta.columns]

    # --- rendre le code insensible à la casse pour les colonnes clés ---
    colmap = {c.lower(): c for c in meta.columns}

    # normaliser PATH -> path
    if "path" in colmap and colmap["path"] != "path":
        meta.rename(columns={colmap["path"]: "path"}, inplace=True)

    # normaliser SAMPLE_ID -> sample_id (au cas où)
    if "sample_id" in colmap and colmap["sample_id"] != "sample_id":
        meta.rename(columns={colmap["sample_id"]: "sample_id"}, inplace=True)


    if "path" in meta.columns:
        meta["path_key"] = meta["path"].apply(_norm_path_key)
    elif "filename" in meta.columns:
        meta["filename"] = meta["filename"].astype(str).str.strip()
    else:
        raise ValueError("metadata.xlsx doit contenir une colonne 'path' ou 'filename'.")
    
    # --- extraire unités des noms de colonnes, et nettoyer les noms ---
    units = {}
    new_cols = []
    for c in meta.columns:
        base, unit = split_col_and_unit(c)
        new_cols.append(base)
        if unit and base.lower() not in ("path", "filename", "sample_id"):
            units[base] = unit

    meta.columns = new_cols

    # on renvoie aussi les unités
    meta.attrs["units"] = units

    # colonnes "export" (metadata) : on exclut les colonnes techniques ajoutées par l'app
    export_bases = [c for c in meta.columns if c not in ("path_key",)]
    export_map = {}
    for c in export_bases:
        if c in ("path", "filename", "sample_id"):
            export_map[c] = c
        else:
            u = units.get(c, None)
            export_map[c] = f"{c} [{u}]" if u else c
    meta.attrs["export_bases"] = export_bases
    meta.attrs["export_map"] = export_map

    return meta


# -------------------- util tri naturel --------------------
_num_re = re.compile(r"(\d+)")


# --- AJOUTE ceci en haut, après les imports tkinter ---
class ScrollFrame(ttk.Frame):
    """Cadre scrollable vertical (barre + molette)."""
    def __init__(self, parent, width=420, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = ttk.Frame(self.canvas)

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        # maj auto de la zone scrollable
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # molette
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)       # Windows
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)   # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta/120), "units")

    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

def canon_group(v):
    """Remplace NaN par '(none)' et traite récursivement les tuples."""
    if isinstance(v, tuple):
        return tuple(canon_group(x) for x in v)
    try:
        if pd.isna(v):
            return "(none)"
    except Exception:
        pass
    return v

def natural_key(s: str):
    parts = _num_re.split(str(s))
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            try: out.append(int(p))
            except: out.append(p)
        else:
            out.append(p.lower())
    return tuple(out)

def is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating))

def sort_key_value(v):
    """Tuple comparable: gère nombres / str (tri naturel) / tuples / NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return (9,)
    if isinstance(v, tuple): return (0,) + tuple(sort_key_value(e) for e in v)
    if is_num(v): return (1, float(v))
    if isinstance(v, str):
        try: return (1, float(v))
        except: return (2,) + natural_key(v)
    return (3, str(v))

# -------------------- lecture .xy --------------------
def read_xy(path: Path, downsample_every: int = 1, normalize: bool = False):
    def _fallback_read(p: Path):
        rows = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", ";", "//")):
                    continue
                parts = re.split(r"[,\s;]+", line)
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        if not rows:
            raise ValueError("Aucune donnée lisible")
        arr = np.array(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    try:
        data = np.loadtxt(path, comments=("#", ";", "//"), dtype=float, ndmin=2)
        x, y = data[:, 0], data[:, 1]
    except Exception:
        x, y = _fallback_read(path)

    if downsample_every > 1:
        x = x[::downsample_every]
        y = y[::downsample_every]

    if normalize:
        ymax = float(np.max(y)) if y.size else 1.0
        if ymax > 0:
            y = y / ymax

    return x, y

# -------------------- parsing des noms --------------------
# 1) Nom “complet” (underscore après sample_id optionnel; segment libre optionnel avant 'cycle')
PAT_FULL = re.compile(
    r"""
    ^(?P<sample>[^_]+)_?                 # sample_id (1.10 ou 1.10_)
    PbI2_(?P<pbi2>[\d\.]+)_
    FAI_(?P<fai>[\d\.]+)_
    OFN_(?P<ofn>[\d\.]+)_
    HA_(?P<ha>[A-Za-z0-9\.\-]+)_
    (?:[A-Za-z0-9\.\-]+_)?               # ex. petrish (facultatif)
    cycle_(?P<cin>\d+)in(?P<cout>\d+)out_
    Ann(?:ealing)?_(?P<ann>[\d\.]+)
    """, re.IGNORECASE | re.VERBOSE,
)

# 2) “template” minimal : 1.12_PbI2_1.5.xy  OU  1.12PbI2_1.5.xy
PAT_MINI = re.compile(r"^(?P<sample>[^_]+)_?PbI2_(?P<pbi2>[\d\.]+)$", re.IGNORECASE)

# 3) ITO.xy (substrat seul)
PAT_ITO = re.compile(r"^ITO$", re.IGNORECASE)

def _num_or_str(s):
    try:
        if re.fullmatch(r"\d+", s): return int(s)
        return float(s)
    except Exception:
        return s

def parse_params(p: Path):
    """
    Parse depuis le nom, mais garde aussi:
    - path_abs : chemin absolu pour lire le .xy
    - path_rel : relatif au dossier racine (self.here)
    - path_key : version normalisée (pour merge Excel)
    """
    fname = p.name
    stem = p.stem

    # base infos
    out = {
        "filename": fname,
        "path_abs": str(p.resolve()),
        # path_rel rempli plus tard (car on a besoin du root pour relative_to)
        "path_rel": None,
        "path_key": None,
        "sample_id": stem,
        "PbI2": np.nan, "FAI": np.nan, "OFN": np.nan,
        "HA": np.nan, "cycle_in": np.nan, "cycle_out": np.nan, "anneal": np.nan,
    }

    if PAT_ITO.match(stem):
        out["sample_id"] = "ITO"
        return out

    m = PAT_FULL.search(stem)
    if m:
        d = m.groupdict()
        out.update({
            "sample_id": d["sample"],
            "PbI2": _num_or_str(d["pbi2"]),
            "FAI": _num_or_str(d["fai"]),
            "OFN": _num_or_str(d["ofn"]),
            "HA": _num_or_str(d["ha"]),
            "cycle_in": int(d["cin"]),
            "cycle_out": int(d["cout"]),
            "anneal": _num_or_str(d["ann"]),
        })
        return out

    m2 = PAT_MINI.search(stem)
    if m2:
        d = m2.groupdict()
        out.update({
            "sample_id": d["sample"],
            "PbI2": _num_or_str(d["pbi2"]),
        })
        return out

    # fallback = sample_id = stem déjà
    return out



class MyToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, reset_cb, *args, **kwargs):
        self._reset_cb = reset_cb
        super().__init__(canvas, window, pack_toolbar=False)

    # Remplace le "Home" natif par TON reset (mêmes effets que le double-clic)
    def home(self, *args, **kwargs):
        try:
            self._reset_cb()
        except Exception:
            # fallback au cas où
            super().home(*args, **kwargs)


# -------------------- App Tk --------------------
class GonioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gonio viewer — filtres + couleurs")
        self.geometry("1280x860")

        self.here = Path(".").resolve()
        self.files = sorted(self.here.rglob("*.xy"), key=lambda p: p.as_posix().lower())

        if not self.files:
            messagebox.showwarning("Avertissement", f"Aucun fichier {GLOB_PATTERN} trouvé dans {self.here}")

        self.df = self._build_df(self.files)
        self.xy_cache = {}
        self.group_color_overrides = {}
        self.color_by_var = tk.StringVar(value="sample_id")

        self.normalize_var = tk.BooleanVar(value=False)
        self.ito_corr_var = tk.BooleanVar(value=False)  # correction d\'offset ITO (alignement)
        self.legend_outside_var = tk.BooleanVar(value=True)
        # --- Phase quantification state ---
        self.phase_apply_offset = False   # si True, _update_plot affiche x corrigé ITO
        self.phase_template_mode = False  # si True, clic sur le plot = ajout de pic template
        self.phase_peaks = []             # list[dict]: {x, phase_var}
        self.phase_markers = []           # artistes (vlines) sur le plot
        self.phase_offsets = {}           # path_key -> offset_deg
        self.phase_results = None         # DataFrame résultats (merge metadata + phases)
        self.phase_review_data = []       # list de dict pour review (optionnel)

        self._last_peak_added = None  # (timestamp, x) pour annuler le 1er clic d'un double-clic
        self._phase_theo_catalog = None
        self._phase_theo_Imax = 1.0
        self._phase_theo_xlim = None

        # Paramètres (mêmes idées que Plot_research)
        self.phase_search_halfwidth = tk.DoubleVar(value=0.5)   # ±° autour du pic template pour trouver le max
        self.phase_fit_halfwidth    = tk.DoubleVar(value=0.5)   # fenêtre locale Voigt
        self.phase_min_amp_frac     = tk.DoubleVar(value=0.05)  # seuil amplitude (fraction de max(y))
        self.phase_min_area_frac    = tk.DoubleVar(value=0.01) # seuil aire (fraction de max(y)*2*fit_hw)

        self._last_full_xlim = None
        self._last_full_ylim = None
        self._prev_normalize = self.normalize_var.get()
        self._skip_restore_once = False  # si True: ne pas restaurer x/y à l'appel suivant
        # --- Survol (hover) ---
        self._line_to_label = {}
        self._hover_anno = None
        self._hover_visible = False
                # --- Fermer = quitter net ---
        self.protocol("WM_DELETE_WINDOW", self._on_close)


        # mapping affichage -> valeur réelle (gère "(none)")
        self.display_to_value = {}

        self._build_ui()
        self._populate_filters()
        self._update_plot()

    def _build_df(self, paths):
        """
        Construit le DataFrame principal à partir des fichiers .xy,
        puis (optionnellement) merge metadata.xlsx.

        Dépend de:
        - parse_params(Path)  -> dict
        - load_metadata_table(root: Path) -> DataFrame|None
        - _norm_path_key(str) -> str
        - natural_key(str)    -> key
        - RESERVED_COLS, UNITS
        - flags: STRICT_METADATA, REQUIRED_META_COLS, META_OVERWRITE
        """
        # ----------------------------
        # 1) Parse fichiers
        # ----------------------------
        rows = [parse_params(p) for p in paths]
        df = pd.DataFrame(rows)

        # ----------------------------
        # 2) Ajoute path_rel + path_key
        # ----------------------------
        rels = []
        for p in paths:
            try:
                rel = p.resolve().relative_to(self.here).as_posix()
            except Exception:
                # fallback si relative_to échoue
                rel = p.name
            rels.append(rel)

        df["path_rel"] = rels
        df["path_key"] = df["path_rel"].apply(_norm_path_key)

        # ----------------------------
        # 3) Charge metadata (optionnel)
        # ----------------------------
        meta = load_metadata_table(self.here)
        # unités venant de l'Excel (en-têtes avec [..])
        self.units = {}
        if meta is not None:
            self.units.update(meta.attrs.get("units", {}))
            self._meta_export_bases = list(meta.attrs.get("export_bases", []))
            self._meta_export_map = dict(meta.attrs.get("export_map", {}))
        else:
            self._meta_export_bases = []
            self._meta_export_map = {}
        # conserve une copie nettoyée de la table metadata pour l'export RESULTS
        self._meta_df = meta.copy() if meta is not None else None


        if meta is not None:
            # --- Choix de la clé de merge
            merge_on = None
            if "path_key" in meta.columns:
                merge_on = "path_key"

                # Contrôle doublons dans Excel (path dupliqué = ambigu)
                dup = meta["path_key"][meta["path_key"].duplicated()].unique()
                if len(dup) > 0:
                    raise ValueError(
                        f"[metadata.xlsx] chemins dupliqués dans Excel (colonne 'path') : "
                        f"{list(dup)[:10]}{' ...' if len(dup) > 10 else ''}"
                    )

                # Mode strict: chaque fichier doit être listé dans Excel
                if STRICT_METADATA:
                    missing = df.loc[~df["path_key"].isin(meta["path_key"]), "path_rel"].tolist()
                    if missing:
                        msg = "\n".join(missing[:30])
                        raise ValueError(
                            "[metadata.xlsx] Il manque des lignes pour certains fichiers .xy.\n"
                            "Ajoute-les dans l’onglet FILES (colonne 'path'). Exemples:\n"
                            f"{msg}\n"
                            f"(+ {max(0, len(missing)-30)} autres)"
                        )

            elif "filename" in meta.columns:
                merge_on = "filename"

                if STRICT_METADATA:
                    missing = df.loc[~df["filename"].isin(meta["filename"]), "filename"].tolist()
                    if missing:
                        msg = "\n".join(missing[:30])
                        raise ValueError(
                            "[metadata.xlsx] Il manque des lignes pour certains fichiers .xy.\n"
                            "Ajoute-les dans l’onglet FILES (colonne 'filename'). Exemples:\n"
                            f"{msg}\n"
                            f"(+ {max(0, len(missing)-30)} autres)"
                        )
            else:
                raise ValueError("[metadata.xlsx] doit contenir une colonne 'path' ou 'filename'.")

            # --- Merge
            df = df.merge(meta, on=merge_on, how="left", suffixes=("", "__meta"))

            # --- Applique la politique de priorité (Excel vs parsing)
            # Colonnes Excel à appliquer = toutes sauf la clé et 'path' texte
            meta_cols = list(meta.columns)

            for c in meta_cols:
                if c in {merge_on, "path", "path_key"}:
                    continue
                cm = f"{c}__meta"
                if cm not in df.columns:
                    continue

                meta_ser = df[cm]

                # base colonne (si elle n'existait pas avant)
                if c in df.columns:
                    base_ser = df[c]
                else:
                    base_ser = pd.Series([np.nan] * len(df), index=df.index)

                if META_OVERWRITE:
                    # Excel prioritaire
                    df[c] = meta_ser.combine_first(base_ser)
                else:
                    # parsing prioritaire
                    df[c] = base_ser.combine_first(meta_ser)

                df.drop(columns=[cm], inplace=True)


            # ----------------------------
            # 4) Validation champs obligatoires
            # ----------------------------
            if STRICT_METADATA:
                for col in REQUIRED_META_COLS:
                    if col not in df.columns:
                        raise ValueError(f"[metadata.xlsx] Colonne obligatoire manquante: '{col}'")

                    bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
                    if not bad.empty:
                        examples = bad["path_rel"].head(30).tolist()
                        msg = "\n".join(examples)
                        raise ValueError(
                            f"[metadata.xlsx] Champ obligatoire vide: '{col}' pour certains fichiers.\n"
                            f"{msg}\n"
                            f"(+ {max(0, len(bad)-30)} autres)"
                        )

        # ----------------------------
        # 5) Colonnes dérivées / label dynamiques
        # ----------------------------
        # cycle_pair (si tu utilises encore cycle_in/out)
        if "cycle_in" in df.columns and "cycle_out" in df.columns:
            df["cycle_pair"] = list(zip(df["cycle_in"], df["cycle_out"]))
        else:
            df["cycle_pair"] = None

        # Liste des colonnes à afficher dans le label:
        # - exclure colonnes réservées
        # - exclure colonnes entièrement vides
        reserved = set(RESERVED_COLS)
        candidate = [c for c in df.columns if c not in reserved]
        candidate = [c for c in candidate if not df[c].isna().all()]

        # Colonnes filtrables dynamiques = paramètres non vides
        # (on exclut sample_id et les colonnes techniques)
        self.filter_cols = [c for c in candidate if c != "sample_id"]

        # Optionnel: ne pas créer un filtre si trop de valeurs uniques (sinon listbox énorme)
        MAX_UNIQUE = 120
        self.filter_cols = [c for c in self.filter_cols if df[c].nunique(dropna=False) <= MAX_UNIQUE]


        # ordre préféré puis extras triés
        preferred = ["PbI2", "FAI", "OFN", "HA", "anneal", "cycle_in", "cycle_out"]
        extras = [c for c in candidate if c not in preferred and c != "sample_id"]
        extras = sorted(extras, key=lambda s: natural_key(str(s)))

        self.label_cols = [c for c in preferred if c in candidate] + extras
        # sample_id = identifiant -> toujours string (sinon filtrage casse)
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str).str.strip()
            df["sample_id"] = df["sample_id"].replace({"nan": np.nan, "None": np.nan})

        return df


    # --- MODIFIE ta méthode _build_ui() comme suit ---
    def _build_ui(self):
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # Colonne de gauche scrollable
        ctrl_sf = ScrollFrame(pane, width=420)
        ctrl = ctrl_sf.inner  # on remplit ce frame interne
        plot = ttk.Frame(pane)
        pane.add(ctrl_sf, weight=0)
        pane.add(plot, weight=1)

        # Top controls
        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row1, text="Color by:").pack(side=tk.LEFT)

        # --- options couleur dynamiques ---
        color_opts = ["filename"]
        if "sample_id" in self.df.columns:
            color_opts.append("sample_id")

        # ajoute toutes les colonnes filtrables dynamiques (celles de l'Excel/DF)
        for c in getattr(self, "filter_cols", []):
            if c not in color_opts:
                color_opts.append(c)

        # (optionnel) si tu veux aussi autoriser la coloration par d'autres colonnes présentes
        # mais non filtrées, décommente ça :
        # for c in getattr(self, "label_cols", []):
        #     if c not in color_opts and c != "sample_id":
        #         color_opts.append(c)

        ttk.OptionMenu(
            row1,
            self.color_by_var,
            self.color_by_var.get(),
            *color_opts,
            command=lambda *_: self._update_plot()
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            row1, text="Éditer couleurs…",
            command=lambda: getattr(self, "_edit_colors", lambda: None)()
        ).pack(side=tk.LEFT, padx=6)


        row2 = ttk.Frame(ctrl); row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Checkbutton(row2, text="Normalize", variable=self.normalize_var,
                        command=self._update_plot).pack(side=tk.LEFT)
        ttk.Checkbutton(row2, text="Corriger ITO", variable=self.ito_corr_var,
                        command=self._on_toggle_ito_corr).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(row2, text="Légende à droite", variable=self.legend_outside_var,
                        command=self._update_plot).pack(side=tk.LEFT, padx=10)

        # Filtres (dynamiques)
        self.filter_boxes = {}
        self._make_filter_box(ctrl, "sample_id", "Samples (ID)").pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        for key in getattr(self, "filter_cols", []):
            title = key
            unit = getattr(self, "units", {}).get(key)
            if unit:
                title = f"{key} ({unit})"
            self._make_filter_box(ctrl, key, title).pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        # Actions bas
        row3 = ttk.Frame(ctrl); row3.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row3, text="Tout sélectionner", command=self._select_all_filters).pack(side=tk.LEFT)
        ttk.Button(row3, text="Tout désélectionner", command=self._clear_all_filters).pack(side=tk.LEFT, padx=6)
        ttk.Button(row3, text="Exporter PNG…", command=self._export_png).pack(side=tk.LEFT, padx=18)

        # --------- Phase quantification (template → batch) ---------
        phf = ttk.LabelFrame(ctrl, text="Phase (template → batch)")
        phf.pack(fill=tk.X, padx=8, pady=(0, 8))

        rowp = ttk.Frame(phf); rowp.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Button(rowp, text="Créer / éditer template", command=self._phase_start_template).pack(side=tk.LEFT)
        ttk.Button(rowp, text="Annuler", command=self._phase_cancel_template).pack(side=tk.LEFT, padx=6)

        rowp2 = ttk.Frame(phf); rowp2.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(rowp2, text="Suppr. dernier pic", command=self._phase_pop_peak).pack(side=tk.LEFT)
        ttk.Button(rowp2, text="Clear pics", command=self._phase_clear_peaks).pack(side=tk.LEFT, padx=6)
        ttk.Button(rowp2, text="Calculer + review + export", command=self._phase_run_batch).pack(side=tk.LEFT, padx=18)

        pars = ttk.Frame(phf); pars.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(pars, text="Search ±°:").grid(row=0, column=0, sticky="w")
        ttk.Entry(pars, textvariable=self.phase_search_halfwidth, width=7).grid(row=0, column=1, sticky="w", padx=(4,12))
        ttk.Label(pars, text="Fit hw °:").grid(row=0, column=2, sticky="w")
        ttk.Entry(pars, textvariable=self.phase_fit_halfwidth, width=7).grid(row=0, column=3, sticky="w", padx=(4,12))
        ttk.Label(pars, text="Min amp frac:").grid(row=1, column=0, sticky="w", pady=(2,0))
        ttk.Entry(pars, textvariable=self.phase_min_amp_frac, width=7).grid(row=1, column=1, sticky="w", padx=(4,12), pady=(2,0))
        ttk.Label(pars, text="Min area frac:").grid(row=1, column=2, sticky="w", pady=(2,0))
        ttk.Entry(pars, textvariable=self.phase_min_area_frac, width=7).grid(row=1, column=3, sticky="w", padx=(4,12), pady=(2,0))

        self._phase_hint = tk.StringVar(value="(1) Clique sur 'Créer/éditer' → offset ITO appliqué. (2) Clique sur le plot pour ajouter des pics. (3) Choisis phases. (4) Calculer.")
        ttk.Label(phf, textvariable=self._phase_hint, wraplength=380).pack(fill=tk.X, padx=8, pady=(0, 6))

        self._phase_peaks_frame = ttk.Frame(phf)
        self._phase_peaks_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Zone plot + toolbar (inchangé)
        self.fig, self.ax = plt.subplots(figsize=(9.6, 7.0))
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = MyToolbar(self.canvas, plot, self._reset_view)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar.home = lambda *a, **k: self._reset_view()



        # Écoute des événements Matplotlib (double-clic)
        self.cid_click = self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
        # Survol (hover): bouge = affiche info-bulle, sortie figure = cache
        self.cid_move  = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.cid_leave = self.canvas.mpl_connect("figure_leave_event", lambda e: self._hide_hover())




    def _make_filter_box(self, parent, key, title):
        frame = ttk.LabelFrame(parent, text=title)

        # Liste + scrollbar
        lb = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

        # Auto-apply
        lb.bind("<<ListboxSelect>>", lambda _e: self._update_plot())

        # Barre d’actions locale (à droite, compacte)
        btnrow = ttk.Frame(frame)
        btnrow.pack(anchor="e", padx=6, pady=(0, 6))
        ttk.Button(btnrow, text="Tous",  width=6, command=lambda lb=lb: self._lb_select_all(lb)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnrow, text="Aucun", width=6, command=lambda lb=lb: self._lb_clear_all(lb)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnrow, text="Inverser", width=7, command=lambda lb=lb: self._lb_invert(lb)).pack(side=tk.LEFT, padx=2)

        # Menu contextuel (clic droit)
        menu = tk.Menu(lb, tearoff=0)
        menu.add_command(label="Tout sélectionner",  command=lambda lb=lb: self._lb_select_all(lb))
        menu.add_command(label="Tout désélectionner", command=lambda lb=lb: self._lb_clear_all(lb))
        menu.add_command(label="Inverser la sélection", command=lambda lb=lb: self._lb_invert(lb))

        def _popup(event, lb=lb):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        lb.bind("<Button-3>", _popup)   # Windows/Linux
        lb.bind("<Button-2>", _popup)   # macOS (clic droit)

        self.filter_boxes[key] = lb
        return frame

    def _populate_filters(self):
        self.display_to_value.clear()

        # Sample IDs (tri naturel)
        key = "sample_id"
        lb = self.filter_boxes[key]
        lb.delete(0, tk.END)
        samples = sorted(self.df["sample_id"].dropna().unique().tolist(), key=natural_key)
        self.display_to_value[key] = {s: s for s in samples}
        for s in samples:
            lb.insert(tk.END, s)
        lb.select_set(0, tk.END)

        # Autres filtres avec option "(none)" si NaN existants
        for key, lb in self.filter_boxes.items():
            if key == "sample_id":  # déjà fait
                continue
            lb.delete(0, tk.END)
            series = self.df[key]
            has_nan = series.isna().any()
            vals = series.dropna().unique().tolist()

            # tri: numérique si possible, sinon naturel
            try:
                vals = sorted(vals, key=lambda v: float(v))
            except Exception:
                vals = sorted(vals, key=lambda v: natural_key(str(v)))

            self.display_to_value[key] = {}
            if has_nan:
                self.display_to_value[key]["(none)"] = np.nan
                lb.insert(tk.END, "(none)")
            for v in vals:
                s = str(v)
                self.display_to_value[key][s] = v
                lb.insert(tk.END, s)
            lb.select_set(0, tk.END)

    def _select_all_filters(self):
        for lb in self.filter_boxes.values():
            lb.select_set(0, tk.END)
        self._update_plot()

    def _clear_all_filters(self):
        for lb in self.filter_boxes.values():
            lb.selection_clear(0, tk.END)
        self._update_plot()

    def _get_active_masks(self, df):
        """Masque combiné pour tous les filtres; gestion '(none)'/NaN."""
        mask = pd.Series(True, index=df.index)
        for key, lb in self.filter_boxes.items():
            labels = [lb.get(i) for i in lb.curselection()]
            if len(labels) == 0:
                return pd.Series(False, index=df.index)

            sel_vals, allow_nan = [], False
            for lab in labels:
                val = self.display_to_value[key][lab]
                if isinstance(val, float) and np.isnan(val):
                    allow_nan = True
                else:
                    sel_vals.append(val)

            col = df[key]
            m = col.isin(sel_vals)
            if allow_nan:
                m = m | col.isna()
            mask = mask & m
        return mask

    def _color_for_group(self, group_key, group_value, default_cycle):
        return self.group_color_overrides.get((group_key, group_value)) or next(default_cycle)

    def _make_group_value(self, row, group_key):
        if group_key == "cycle_pair":
            return (row["cycle_in"], row["cycle_out"])
        return row[group_key]

    # -------- légende dynamique --------
    def _fmt_val(self, v, unit=None):
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        if is_num(v):
            s = f"{float(v):g}"
            return f"{s} {unit}" if unit else s
        return str(v)

    def _row_label(self, row):
        sid = str(row.get("sample_id", ""))
        if sid.upper() == "ITO":
            return "ITO"

        parts = [sid]

        # Gestion spéciale cycles (pour garder le style 10in5out)
        cin = row.get("cycle_in", np.nan)
        cout = row.get("cycle_out", np.nan)

        for key in getattr(self, "label_cols", []):
            if key in {"cycle_in", "cycle_out"}:
                continue  # traité après

            v = row.get(key, np.nan)

            # vide ?
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, str) and not v.strip():
                continue

            # HA: si num -> g/m³ sinon string
            if key == "HA":
                if is_num(v):
                    parts.append(f"HA={float(v):g} g/m³")
                else:
                    parts.append(f"HA={str(v)}")
                continue

            # format standard
            unit = getattr(self, "units", {}).get(key)
            if is_num(v):
                s = f"{float(v):g}"
                parts.append(f"{key}={s}{(' ' + unit) if unit else ''}")
            else:
                parts.append(f"{key}={str(v)}{(' ' + unit) if unit else ''}")

        # cycles à la fin si présents
        if not (isinstance(cin, float) and np.isnan(cin)) and not (isinstance(cout, float) and np.isnan(cout)):
            try:
                parts.append(f"{int(cin)}in{int(cout)}out")
            except Exception:
                parts.append(f"{cin}in{cout}out")

        return " | ".join(parts)

    

    def _on_close(self):
        try:
            self.quit()
            self.destroy()
        finally:
            sys.exit(0)

    def _reset_view(self):
        """Réinitialise uniquement l'échelle (dé-zoom) à la dernière vue 'full' calculée."""
        if self._last_full_xlim is not None:
            self.ax.set_xlim(self._last_full_xlim)
        if self._last_full_ylim is not None:
            self.ax.set_ylim(self._last_full_ylim)
        self.canvas.draw_idle()

    
    def _on_mpl_click(self, event):
        """Gestion clics Matplotlib.

        - Double-clic gauche : reset zoom (et évite de laisser un pic “fantôme” créé par le 1er clic).
        - En mode template : clic simple gauche = ajout de pic, MAIS uniquement si la toolbar n'est pas en mode zoom/pan.
        """
        # --- 1) double-clic : reset + annule un éventuel pic ajouté au 1er clic ---
        if getattr(event, "dblclick", False) and getattr(event, "button", None) == 1:
            try:
                # Si le 1er clic du double-clic a ajouté un pic, on l'annule.
                if getattr(self, "_last_peak_added", None):
                    ts, x0 = self._last_peak_added
                    if (time.time() - float(ts)) <= 0.45:
                        # retire le pic le plus proche de x0 (tol élargie)
                        j_best, d_best = None, 1e9
                        for j, pk in enumerate(getattr(self, "phase_peaks", [])):
                            d = abs(float(pk.get("x", 1e9)) - float(x0))
                            if d < d_best:
                                d_best, j_best = d, j
                        if j_best is not None and d_best <= 0.25:
                            try:
                                self.phase_peaks.pop(j_best)
                                self._phase_rebuild_peaks_ui()
                            except Exception:
                                pass
                self._last_peak_added = None
            except Exception:
                pass
            self._reset_view()
            return

        # --- 2) clic simple : ajoute un pic template seulement si toolbar inactive ---
        if not (self.phase_template_mode and (getattr(event, "button", None) == 1) and (getattr(event, "inaxes", None) == self.ax)):
            return

        # si la toolbar est en mode zoom/pan, Matplotlib envoie quand même des clics: on ignore
        try:
            if getattr(self, "toolbar", None) is not None:
                mode = getattr(self.toolbar, "mode", "")
                if isinstance(mode, str) and mode.strip():
                    return
        except Exception:
            pass

        # sécurité: ignore si xdata None
        x_click = getattr(event, "xdata", None)
        if x_click is None:
            return
        x_click = float(x_click)

        self._phase_add_peak_from_click(x_click)
        return


    def _ensure_hover_annotation(self):
        """Crée (ou recrée) l'annotation hover liée à l'axe courant."""
        try:
            if self._hover_anno is not None and self._hover_anno.axes is not self.ax:
                try:
                    self._hover_anno.remove()
                except Exception:
                    pass
                self._hover_anno = None
            if self._hover_anno is None:
                self._hover_anno = self.ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(12, 12),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.9),
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    zorder=15,
                    visible=False,
                )
        except Exception:
            self._hover_anno = None

    def _hide_hover(self):
        if self._hover_anno is not None and self._hover_visible:
            try:
                self._hover_anno.set_visible(False)
                self._hover_visible = False
                self.canvas.draw_idle()
            except Exception:
                pass

    def _on_mouse_move(self, event):
        """Affiche la légende du sample près du curseur si on survole une courbe."""
        if event.inaxes is not self.ax or not self.ax.lines:
            self._hide_hover()
            return

        self._ensure_hover_annotation()
        if self._hover_anno is None:
            return

        hit_any = False
        for ln in reversed(self.ax.get_lines()):
            label = self._line_to_label.get(ln)
            if not label:
                continue
            try:
                contains, info = ln.contains(event)
            except Exception:
                contains, info = (False, {})
            if not contains:
                continue

            xdata = ln.get_xdata(orig=False)
            ydata = ln.get_ydata(orig=False)
            x, y = None, None
            try:
                ind = info.get("ind", [])
                if len(ind):
                    i0 = int(ind[0])
                    x, y = float(xdata[i0]), float(ydata[i0])
                else:
                    if event.xdata is not None:
                        import numpy as _np
                        i0 = int(_np.nanargmin(_np.abs(_np.array(xdata, dtype=float) - float(event.xdata))))
                        x, y = float(xdata[i0]), float(ydata[i0])
            except Exception:
                pass

            if x is None or y is None:
                if event.xdata is None or event.ydata is None:
                    continue
                x, y = float(event.xdata), float(event.ydata)

            try:
                self._hover_anno.xy = (x, y)
                self._hover_anno.set_text(str(label))
                self._hover_anno.set_visible(True)
                self._hover_visible = True
                self.canvas.draw_idle()
                hit_any = True
                break
            except Exception:
                pass

        if not hit_any:
            self._hide_hover()


    # -----------------------------------
    def _update_plot(self):
        # --- état courant & vue avant clear ---
        cur_norm = self.normalize_var.get()
        norm_changed = (cur_norm != self._prev_normalize)

        try:
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
        except Exception:
            cur_xlim = None
            cur_ylim = None

        # Était-on zoomé ? (comparé à la dernière vue "full")
        user_zoomed_x = False
        user_zoomed_y = False
        if (not self._skip_restore_once) and (self._last_full_xlim is not None) and (cur_xlim is not None):
            user_zoomed_x = (abs(cur_xlim[0] - self._last_full_xlim[0]) > 1e-9 or
                            abs(cur_xlim[1] - self._last_full_xlim[1]) > 1e-9)
        if (not self._skip_restore_once) and (self._last_full_ylim is not None) and (cur_ylim is not None):
            user_zoomed_y = (abs(cur_ylim[0] - self._last_full_ylim[0]) > 1e-9 or
                            abs(cur_ylim[1] - self._last_full_ylim[1]) > 1e-9)


        self.ax.clear()
        self.ax.set_title(TITLE)
        self.ax.set_xlabel(XLABEL)
        self.ax.set_ylabel(YLABEL)
        self.ax.grid(True, which="both", ls="--", alpha=0.3)

        df = self.df.copy()
        mask = self._get_active_masks(df)
        df = df[mask]

        if df.empty:
            self.ax.text(0.5, 0.5, "Aucune courbe après filtrage",
                        ha="center", va="center", transform=self.ax.transAxes)
            # Eviter de restaurer d'anciennes limites au prochain redraw
            self._last_full_xlim = None
            self._last_full_ylim = None
            self._skip_restore_once = True
            self.canvas.draw_idle()
            return


        self._line_to_label = {}
        group_key = self.color_by_var.get()
        df["_group"] = df.apply(lambda r: canon_group(self._make_group_value(r, group_key)), axis=1)


        # Tri robuste (par clé couleur puis par sample_id naturellement)
        df["_group_sort"] = df["_group"].map(sort_key_value)
        df["_sample_sort"] = df["sample_id"].map(natural_key)
        df = df.sort_values(by=["_group_sort", "_sample_sort"], kind="mergesort")

        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        default_cycle = cycle(default_colors if default_colors else ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])

        unique_groups = pd.unique(df["_group"])
        group_to_color = {g: self._color_for_group(group_key, g, default_cycle) for g in unique_groups}

        # mémorise la liste exacte des scans actuellement affichés (pour le batch phase)
        self._last_plotted_rows = []

        for _, row in df.iterrows():
            path = Path(row["path_abs"])
            cache_key = (row["filename"], self.normalize_var.get())
            if cache_key in self.xy_cache:
                x, y = self.xy_cache[cache_key]
            else:
                x, y = read_xy(path, normalize=self.normalize_var.get())
                self.xy_cache[cache_key] = (x, y)

            # option: afficher avec correction ITO
            x_plot = x
            if getattr(self, "ito_corr_var", None) is not None and self.ito_corr_var.get():
                off = self.phase_offsets.get(row.get("path_key", ""), 0.0)
                try:
                    off = float(off)
                except Exception:
                    off = 0.0
                if not np.isfinite(off):
                    off = 0.0
                x_plot = x - off

            # mémorise la ligne (keys utiles) pour les calculs batch
            try:
                self._last_plotted_rows.append({
                    "path_key": row.get("path_key", ""),
                    "path_rel": row.get("path_rel", ""),
                    "path_abs": row.get("path_abs", ""),
                    "filename": row.get("filename", ""),
                    "sample_id": row.get("sample_id", ""),
                    "cycle_in": row.get("cycle_in", None),
                    "cycle_out": row.get("cycle_out", None),
                    **{k: row.get(k) for k in row.index if k not in ["_group","_group_sort","_sample_sort"]}
                })
            except Exception:
                pass

            color = group_to_color[row["_group"]]
            label = self._row_label(row)
            # conserve le label complet (comme la légende) pour la fenêtre Review
            try:
                if self._last_plotted_rows:
                    self._last_plotted_rows[-1]["label"] = label
            except Exception:
                pass

            ln, = self.ax.plot(x_plot, y, lw=LINEWIDTH, alpha=ALPHA, color=color, label=label)
            try:
                ln.set_pickradius(7)
            except Exception:
                pass
            self._line_to_label[ln] = label

        # Légende : seulement si pas trop de courbes
        if len(df) <= MAX_LEGEND:
            if self.legend_outside_var.get():
                self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
                self.fig.tight_layout(rect=[0, 0, 0.80, 1])
            else:
                self.ax.legend(frameon=False, fontsize=8)
                self.fig.tight_layout()
        else:
            self.fig.tight_layout()

        # --- recalcul des limites complètes & mémorisation ---
        self.ax.relim()
        self.ax.autoscale_view()
        self._last_full_xlim = self.ax.get_xlim()
        self._last_full_ylim = self.ax.get_ylim()

        if not self._skip_restore_once:
            if cur_xlim is not None and user_zoomed_x:
                self.ax.set_xlim(cur_xlim)
            if (not norm_changed) and cur_ylim is not None and user_zoomed_y:
                self.ax.set_ylim(cur_ylim)

        # à partir de maintenant, la prochaine MAJ pourra restaurer normalement
        self._skip_restore_once = False
        self._prev_normalize = cur_norm

        # (re)crée l'annotation de survol et la cache
        self._ensure_hover_annotation()
        self._hide_hover()

        # marqueurs des pics template
        if getattr(self, 'phase_peaks', None):
            for pk in self.phase_peaks:
                try:
                    self.ax.axvline(float(pk.get('x', np.nan)), color='0.2', ls='--', lw=1.0, alpha=0.35)
                except Exception:
                    pass

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # ITO correction toggle (like Normalize)
    # ------------------------------------------------------------------
    def _load_ito_reference_if_needed(self):
        """Silently load ITO reference table (ITO.txt) if available."""
        if getattr(self, "_ito_loaded", False) and hasattr(self, "_ito_x") and hasattr(self, "_ito_y"):
            return
        try:
            ito_path = Path(self.here) / ITO_TABLE_FILE
        except Exception:
            return
        if not ito_path.exists():
            return
        try:
            self._ito_x, self._ito_y = read_ito_table(ito_path)
            self._ito_loaded = True
        except Exception:
            # keep silent: no correction
            self._ito_loaded = False
            self._ito_x, self._ito_y = None, None

    def _ensure_ito_offsets_for_current_rows(self):
        """Compute ITO offsets for currently displayed rows (self._last_plotted_rows).
        Stores in self.phase_offsets[path_key] = offset_deg. Missing/failed -> 0.0.
        """
        self._load_ito_reference_if_needed()
        if (not getattr(self, "_ito_loaded", False)) or self._ito_x is None or self._ito_y is None:
            return
        rows = getattr(self, "_last_plotted_rows", None)
        if not rows:
            return
        for row in rows:
            key = row.get("path_key", "")
            if not key:
                continue
            off0 = self.phase_offsets.get(key, None)
            if off0 is not None and np.isfinite(off0):
                continue
            try:
                path = Path(row["path_abs"])
                x_raw, y_raw = read_xy(path, normalize=False)
                off = estimate_ito_offset_deg(x_raw, y_raw, self._ito_x, self._ito_y)
                off = float(off) if np.isfinite(off) else 0.0
            except Exception:
                off = 0.0
            self.phase_offsets[key] = off

    def _on_toggle_ito_corr(self):
        """Callback when the user toggles the 'Corriger ITO' checkbox."""
        if self.ito_corr_var.get():
            self._ensure_ito_offsets_for_current_rows()
        self._update_plot()

    def _lb_select_all(self, lb):
        lb.select_set(0, tk.END)
        self._update_plot()

    def _lb_clear_all(self, lb):
        lb.selection_clear(0, tk.END)
        self._update_plot()

    def _lb_invert(self, lb):
        sel = set(lb.curselection())
        lb.selection_clear(0, tk.END)
        for i in range(lb.size()):
            if i not in sel:
                lb.selection_set(i)
        self._update_plot()
    

    
    # --------- Phase quantification (template → batch) ---------
    
    def _phase_rebuild_peaks_ui(self):
        """Reconstruit la liste des pics template + menus phases (dans le panneau gauche)."""
        if not hasattr(self, "_phase_peaks_frame") or self._phase_peaks_frame is None:
            return
        for w in self._phase_peaks_frame.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        if not self.phase_peaks:
            ttk.Label(self._phase_peaks_frame, text="Aucun pic template.").pack(anchor="w")
            return

        for i, pk in enumerate(self.phase_peaks, start=1):
            row = ttk.Frame(self._phase_peaks_frame)
            row.pack(fill=tk.X, pady=1)

            x0 = float(pk.get("x", np.nan))
            ttk.Label(row, text=f"{i:02d}) {x0:.3f}°").pack(side=tk.LEFT)

            # --- liste de choix (candidats + phases), comme Plot_research (W4) ---
            choice_var = pk.get("choice_var")
            if choice_var is None:
                choice_var = tk.StringVar(value=pk.get("phase_var").get() if pk.get("phase_var") else PHASE_CHOICES[0])
                pk["choice_var"] = choice_var

            values = pk.get("choice_values")
            choice_map = pk.get("choice_map", {})

            # Si pas déjà construit (ou si vide), on crée une liste minimaliste (phases seules)
            if not values:
                values = list(PHASE_CHOICES)
                choice_map = {ph: (ph, "") for ph in PHASE_CHOICES}
                pk["choice_values"] = values
                pk["choice_map"] = choice_map

            cb = ttk.Combobox(row, textvariable=choice_var, values=values, state="readonly", width=40)
            cb.pack(side=tk.LEFT, padx=6)

            def _on_pick(_ev=None, pk=pk):
                s = pk["choice_var"].get()
                ph, hkl = pk.get("choice_map", {}).get(s, (s, ""))
                try:
                    pk["phase_var"].set(ph)
                except Exception:
                    pk["phase_var"] = tk.StringVar(value=ph)
                pk["hkl"] = hkl
                self._update_plot()

            cb.bind("<<ComboboxSelected>>", _on_pick)

            # X = supprimer
            def _rm(idx=i - 1):
                try:
                    self.phase_peaks.pop(idx)
                except Exception:
                    return
                self._phase_rebuild_peaks_ui()
                self._update_plot()

            ttk.Button(row, text="X", width=2, command=_rm).pack(side=tk.RIGHT)


    def _phase_start_template(self):
        """
        Active le mode template :
        - calcule/charge la table ITO
        - applique l’offset ITO à tous les scans affichés
        - permet d’ajouter des pics template par clic sur le plot
        """
        if not hasattr(self, "_last_plotted_rows") or not self._last_plotted_rows:
            messagebox.showwarning("Phase", "Aucun scan affiché. Applique tes filtres puis clique sur 'Créer/éditer template'.")
            return

        # charge ITO
        ito_path = (self.here / ITO_TABLE_FILE)
        if not ito_path.exists():
            messagebox.showwarning("Phase", f"Impossible de trouver {ITO_TABLE_FILE} dans:\n{self.here}\n\nAjoute ITO.txt (2 colonnes: 2θ, I) dans le dossier de metadata.xlsx.")
            return

        try:
            self._ito_x, self._ito_y = read_ito_table(ito_path)
        except Exception as e:
            messagebox.showerror("Phase", f"Erreur lecture ITO:\n{e}")
            return

        # calc offsets pour les scans affichés (sur données brutes non-normalisées)
        self.phase_offsets = {}
        for row in self._last_plotted_rows:
            try:
                path = Path(row["path_abs"])
                x, y = read_xy(path, normalize=False)
                off = estimate_ito_offset_deg(x, y, self._ito_x, self._ito_y)
                try:
                    off = float(off)
                except Exception:
                    off = 0.0
                if not np.isfinite(off):
                    off = 0.0
                self.phase_offsets[row["path_key"]] = off
            except Exception:
                self.phase_offsets[row["path_key"]] = 0.0

        # on n'applique l'offset que si la case "Corriger ITO" est activée
        self.phase_template_mode = True
        self._update_plot()
        # reset catalogue théorique (si tu as modifié tes fichiers de raies)
        self._phase_theo_catalog = None
        self._phase_theo_Imax = 1.0
        self._phase_theo_xlim = None
        if hasattr(self, "_phase_hint"):
            self._phase_hint.set("Template actif: clique sur le plot pour ajouter un pic (2θ corrigé ITO). Double-clic = reset zoom. Choisis une phase pour chaque pic puis 'Calculer…'.")
        self._update_plot()

    def _phase_cancel_template(self):
        self.phase_template_mode = False
        if hasattr(self, "_phase_hint"):
            self._phase_hint.set("(1) Clique sur 'Créer/éditer' → offset ITO appliqué. (2) Clique sur le plot pour ajouter des pics. (3) Choisis phases. (4) Calculer.")
        self._update_plot()

    def _phase_pop_peak(self):
        if self.phase_peaks:
            self.phase_peaks.pop(-1)
            self._phase_rebuild_peaks_ui()
            self._update_plot()

    def _phase_clear_peaks(self):
        self.phase_peaks = []
        self._phase_rebuild_peaks_ui()
        self._update_plot()


    def _phase_ensure_theo_catalog(self, xlim=None):
        """Construit (si nécessaire) le catalogue de raies théoriques pour proposer des candidats.
        - Tolérant: si aucun fichier n'existe, le catalogue reste vide.
        """
        try:
            if xlim is None:
                if self.ax is not None:
                    xlim = tuple(map(float, self.ax.get_xlim()))
                else:
                    xlim = (0.0, 100.0)
        except Exception:
            xlim = (0.0, 100.0)

        x_min, x_max = float(min(xlim)), float(max(xlim))

        # cache: si déjà construit pour ces xlim (≈), on garde
        try:
            prev = getattr(self, "_phase_theo_xlim", None)
            if prev is not None:
                if abs(prev[0] - x_min) < 1e-6 and abs(prev[1] - x_max) < 1e-6 and getattr(self, "_phase_theo_catalog", None) is not None:
                    return
        except Exception:
            pass

        catalog = []
        Imax = 1.0
        for fname, ph in PHASE_REF_FILES:
            p = (self.here / fname)
            if not p.exists():
                continue
            try:
                tth, inten, hkls = read_phase_table_vesta(p)
            except Exception:
                continue
            if len(tth) == 0:
                continue
            Imax = max(Imax, float(np.max(inten)))
            for tt, I, hkl in zip(tth, inten, hkls):
                if (tt >= x_min - 1.0) and (tt <= x_max + 1.0):
                    catalog.append({"phase": ph, "tth": float(tt), "I": float(I), "hkl": str(hkl)})
        self._phase_theo_catalog = catalog
        self._phase_theo_Imax = float(Imax)
        self._phase_theo_xlim = (x_min, x_max)

    def _phase_candidates_for(self, x0: float):
        """Retourne une liste de candidats [(score, delta, rec), ...] triés."""
        if not np.isfinite(x0):
            return []
        self._phase_ensure_theo_catalog()
        catalog = getattr(self, "_phase_theo_catalog", []) or []
        Imax = float(getattr(self, "_phase_theo_Imax", 1.0) or 1.0)

        cands = []
        for rec in catalog:
            try:
                d = float(rec["tth"]) - float(x0)
            except Exception:
                continue
            if abs(d) <= MATCH_TOL_DEG:
                s = match_score(d, float(rec.get("I", 0.0)), Imax, MATCH_TOL_DEG, W_DIST, W_I, P_DIST)
                cands.append((float(s), float(d), rec))
        cands.sort(key=lambda t: t[0])
        return cands[:MAX_CANDIDATES]


    
    def _phase_add_peak_from_click(self, x_click: float):
        if not np.isfinite(x_click):
            return
        # évite les doublons quasi-identiques (tol 0.05°)
        for pk in self.phase_peaks:
            try:
                if abs(float(pk.get("x", 1e9)) - float(x_click)) < 0.05:
                    return
            except Exception:
                continue

        # construit la liste de choix "comme Plot_research": candidats (phase:hkl) + phases seules
        choice_values = []
        choice_map = {}

        try:
            cands = self._phase_candidates_for(float(x_click))
        except Exception:
            cands = []

        if cands:
            for (s, d, rec) in cands:
                ph = str(rec.get("phase", "Autre/Unknown"))
                hkl = str(rec.get("hkl", "")).strip()
                lab = f"{ph}:{hkl}  · score={s:.3f}  (Δ={d:+.2f}°)"
                choice_values.append(lab)
                choice_map[lab] = (ph, hkl)

            # séparateur "visuel" (Combobox: juste une string)
            sep = "──────────"
            choice_values.append(sep)
            choice_map[sep] = (PHASE_CHOICES[0], "")

        # phases seules (fallback + sélection manuelle)
        for ph in PHASE_CHOICES:
            choice_values.append(ph)
            choice_map[ph] = (ph, "")

        # valeur par défaut: meilleur candidat, sinon 1ère phase
        if cands:
            default_label = choice_values[0]
            default_phase, default_hkl = choice_map[default_label]
        else:
            default_label = PHASE_CHOICES[0]
            default_phase, default_hkl = default_label, ""

        v_phase = tk.StringVar(value=default_phase)
        v_choice = tk.StringVar(value=default_label)

        self.phase_peaks.append({
            "x": float(x_click),
            "phase_var": v_phase,
            "choice_var": v_choice,
            "choice_values": choice_values,
            "choice_map": choice_map,
            "hkl": default_hkl,
        })
        self.phase_peaks.sort(key=lambda d: d["x"])

        # mémorise le fait qu'on a ajouté un pic (utile pour annuler un double-clic)
        try:
            self._last_peak_added = (time.time(), float(x_click))
        except Exception:
            self._last_peak_added = None

        self._phase_rebuild_peaks_ui()
        self._update_plot()


    def _phase_compute_one_scan(self, row, peaks_template, phases_template):
        """
        Calcule les aires Voigt et % de phases pour UN scan.
        Retourne (perc_phase, quality_dict, review_dict)
        """
        path = Path(row["path_abs"])
        x_raw, y_raw = read_xy(path, normalize=False)

        # offset ITO (recalcule si nécessaire)
        off = self.phase_offsets.get(row["path_key"], None)
        # si absent ou non-fini -> (re)estime, sinon on garde
        if off is None:
            try:
                off = estimate_ito_offset_deg(x_raw, y_raw, self._ito_x, self._ito_y)
            except Exception:
                off = 0.0
        try:
            off = float(off)
        except Exception:
            off = 0.0
        if not np.isfinite(off):
            off = 0.0
        # mémorise (même si 0.0) pour cohérence export / affichage
        self.phase_offsets[row["path_key"]] = off

        # applique la correction ITO uniquement si la case est activée
        if getattr(self, "ito_corr_var", None) is not None and self.ito_corr_var.get():
            x = x_raw - off
        else:
            x = x_raw
        used_windows = []  # évite de réutiliser la même zone de fit pour 2 pics template
        y = y_raw

        # baseline enveloppe globale (comme dans Plot_reasearch) : quantile roulant + SavGol
        # -> permet une baseline qui "suit" le signal (et donc des intégrales plus proches de ton ancien code)
        try:
            y_bg_curve = make_envelope_baseline(x, y, deg_window=1.2, q=0.25, sg_poly=2)
        except Exception:
            # fallback : baseline plus simple si jamais l'enveloppe échoue
            y_bg_curve = _estimate_linear_baseline(x, y, method="percentile_edges", q=20.0)
        y_bg = y_bg_curve

        # paramètres seuils
        ymax = float(np.max(y)) if len(y) else 1.0
        fit_hw = float(self.phase_fit_halfwidth.get())
        amp_thr = float(self.phase_min_amp_frac.get()) * ymax
        area_thr = float(self.phase_min_area_frac.get()) * (ymax * (2.0*fit_hw))

        per_peak = []
        for x0_t, ph in zip(peaks_template, phases_template):
            # 1) localiser le max dans la fenêtre search
            shw = float(self.phase_search_halfwidth.get())
            m = (x >= (x0_t - shw)) & (x <= (x0_t + shw))
            if not np.any(m):
                per_peak.append({"phase": ph, "A": 0.0, "ok": False, "missing": True})
                continue
            xx = x[m]; yy = y[m]

            # Choisit un maximum local près de x0_t, sans réutiliser une zone déjà prise par un autre pic template.
            # Objectif: éviter que deux phases s'accrochent sur le même gros pic, et éviter qu'un pic 'manquant'
            # aille se coller sur le voisin massif trop loin de x0_t.
            fit_hw = float(self.phase_fit_halfwidth.get())
            max_shift = min(shw, 0.35)  # °2θ : dérive max autorisée vs position template
            # intensité corrigée baseline (plus robuste que yy brut)
            if len(y_bg_curve) == len(y):
                yyc = yy - np.interp(xx, x, y_bg_curve)
            else:
                yyc = yy
            # --- NOUVEAU: impose un vrai maximum local (évite de "prendre la pente" d'un gros pic voisin) ---
            yyc_det = np.asarray(yyc, float)

            # lissage léger pour stabiliser find_peaks (optionnel mais aide beaucoup)
            try:
                w = min(11, (len(yyc_det)//2)*2 + 1)  # impaire
                if w >= 5:
                    yyc_det = savgol_filter(yyc_det, w, 2, mode="interp")
            except Exception:
                pass

            # Prominence minimale: liée à ton seuil d'amplitude global (sinon tu détectes des "pics" de bruit)
            min_prom = max(0.6 * amp_thr, 1e-12)

            # Distance minimale entre pics (en indices) ~ 0.08° (ajuste si besoin)
            min_dist_idx = _deg_to_samples(xx, 0.08)

            p_idx, props = find_peaks(yyc_det, prominence=min_prom, distance=min_dist_idx)

            # On trie les pics détectés par hauteur (yyc original, pas la version lissée)
            order = p_idx[np.argsort(yyc[p_idx])[::-1]] if len(p_idx) else []

            x_peak = None
            for j in order:
                cand = float(xx[j])

                # 1) ne pas partir trop loin du template
                if abs(cand - float(x0_t)) > max_shift:
                    continue

                # 2) ne pas tomber dans une zone déjà utilisée (fenêtre de fit)
                if any((cand >= lo) and (cand <= hi) for (lo, hi) in used_windows):
                    continue

                x_peak = cand
                break

            if x_peak is None:
                per_peak.append({"phase": ph, "A": 0.0, "ok": False, "missing": True})
                continue


            # réserve la zone de fit pour empêcher une autre phase de réutiliser le même pic (même si elle prend une 'épaule')
            used_windows.append((float(x_peak) - fit_hw, float(x_peak) + fit_hw))

            # 2) fenêtre fit
            mf = (x >= (x_peak - fit_hw)) & (x <= (x_peak + fit_hw))
            if not np.any(mf) or np.sum(mf) < 7:
                per_peak.append({"phase": ph, "A": 0.0, "ok": False, "missing": True})
                continue
            xloc = x[mf]; yloc = y[mf]

            # 3) baseline locale (interpolée depuis l'enveloppe globale → suit mieux la courbure)
            if len(y_bg_curve) == len(y):
                yb = np.interp(xloc, x, y_bg_curve)
            else:
                yb = _estimate_linear_baseline(xloc, yloc, method="percentile_edges", q=20.0)

            # 4) seuil missing: amplitude et aire min
            amp = float(np.max(np.clip(yloc - yb, 0, None)))
            rough_area = float(np.trapz(np.clip(yloc - yb, 0, None), xloc))
            if amp < amp_thr or rough_area < area_thr:
                per_peak.append({"phase": ph, "A": 0.0, "ok": False, "missing": True,
                                 "xloc": xloc, "yloc": yloc, "ybase": yb, "yfit": np.zeros_like(yloc)})
                continue

            res = _fit_voigt_area_with_given_baseline(xloc, yloc, yb)
            A = float(res.get("A", 0.0)) if res.get("ok", False) else 0.0
            ok = bool(res.get("ok", False)) and (A >= area_thr)

            per_peak.append({
                "phase": ph,
                "A": A if ok else 0.0,
                "ok": ok,
                "missing": not ok,
                "xloc": xloc,
                "yloc": yloc,
                "ybase": res.get("y_base", yb),
                "yfit": res.get("yfit", np.zeros_like(yloc)),
            })

        # sommes par phase
        phases_unique = sorted({p for p in phases_template if p is not None})
        S = {p: 0.0 for p in phases_unique}
        for d in per_peak:
            if d["phase"] in S:
                S[d["phase"]] += max(float(d["A"]), 0.0)
        total = float(sum(S.values()))
        perc = {p: (100.0 * S[p] / total) if total > 0 else 0.0 for p in phases_unique}

        qual = {
            "ito_offset_deg": float(off),
            "n_template_peaks": int(len(peaks_template)),
            "n_ok": int(sum(1 for d in per_peak if d.get("ok"))),
            "n_missing": int(sum(1 for d in per_peak if d.get("missing"))),
            "area_total": float(total),
        }

        review = {
            "name": row.get("label", row.get("sample_id", row.get("filename", "scan"))),
            "label": row.get("label", row.get("sample_id", row.get("filename", "scan"))),
            "path_rel": row.get("path_rel", ""),
            "x": x,
            "y": y,
            "y_bg": y_bg,
            "per_peak": per_peak,
            "perc": perc,
        }
        return perc, qual, review

    def _phase_run_batch(self):
        """Batch : calcule % de phases sur tous les scans affichés + ouvre un review + export RESULTS.xlsx."""
        if not hasattr(self, "_last_plotted_rows") or not self._last_plotted_rows:
            messagebox.showwarning("Phase", "Aucun scan affiché.")
            return
        if not self.phase_peaks:
            messagebox.showwarning("Phase", "Ajoute au moins 1 pic template (clic sur le plot en mode template).")
            return

        peaks_template = [float(d["x"]) for d in self.phase_peaks]
        phases_template = [d["phase_var"].get() for d in self.phase_peaks]

        # vérifie phases
        if any((p is None) or (str(p).strip() == "") for p in phases_template):
            messagebox.showwarning("Phase", "Choisis une phase pour chaque pic template.")
            return

        # re-charge ITO si nécessaire
        ito_path = (self.here / ITO_TABLE_FILE)
        if not getattr(self, "_ito_x", None) is not None or not ito_path.exists():
            # force start_template qui charge ITO, mais sans obliger à être en template mode
            if not ito_path.exists():
                messagebox.showwarning("Phase", f"Impossible de trouver {ITO_TABLE_FILE} dans:\n{self.here}")
                return
            self._ito_x, self._ito_y = read_ito_table(ito_path)

        # calcule pour chaque scan affiché
        results_rows = []
        review_data = []
        for row in self._last_plotted_rows:
            try:
                perc, qual, review = self._phase_compute_one_scan(row, peaks_template, phases_template)
            except Exception as e:
                perc = {ph: 0.0 for ph in sorted(set(phases_template))}
                qual = {"ito_offset_deg": np.nan, "n_template_peaks": len(peaks_template), "n_ok": 0, "n_missing": len(peaks_template), "area_total": 0.0, "error": str(e)}
                review = {"name": row.get("sample_id", row.get("filename", "scan")), "path_rel": row.get("path_rel",""), "x": np.array([]), "y": np.array([]), "y_bg": np.array([]), "per_peak": [], "perc": perc}

            out = {"path_key": row["path_key"]}
            # colonnes % par phase (nom stable)
            for ph in sorted(set(phases_template)):
                out[f"{ph} [%]"] = float(perc.get(ph, 0.0))
            out.update({f"Q_{k}": v for k, v in qual.items()})
            results_rows.append(out)
            review_data.append(review)

        self.phase_review_data = review_data

        df_res = pd.DataFrame(results_rows)

        # ---------------------------------------------------------
        # Export : seulement colonnes de metadata.xlsx + résultats
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Base export : metadata.xlsx (si dispo) sinon fallback
        # ---------------------------------------------------------
        df_base_all = pd.DataFrame(self._last_plotted_rows)

        # colonnes metadata (bases) provenant de metadata.xlsx, sinon fallback minimal
        meta_bases = getattr(self, "_meta_export_bases", None) or []
        if not meta_bases:
            meta_bases = ["path", "sample_id", "batch", "PbI2", "FAI", "Annealing"]

        df_base = None
        meta_df = getattr(self, "_meta_df", None)

        if meta_df is not None:
            # s'assure d'avoir path_key
            if "path_key" not in meta_df.columns:
                if "path" in meta_df.columns:
                    meta_df = meta_df.copy()
                    meta_df["path_key"] = meta_df["path"].apply(_norm_path_key)
                else:
                    meta_df = None

        if meta_df is not None:
            keep_meta = ["path_key"] + [c for c in meta_bases if c in meta_df.columns and c != "path_key"]
            df_base = meta_df.loc[meta_df["path_key"].isin(df_res["path_key"]), keep_meta].copy()
        else:
            # fallback: on se limite à quelques colonnes "propres" de l'app
            keep_cols = ["path_key"] + [c for c in meta_bases if c in df_base_all.columns and c != "path_key"]
            df_base = df_base_all.loc[:, keep_cols].copy()

        df_out = df_base.merge(df_res, on="path_key", how="left")

        # retire colonnes techniques éventuelles (sécurité)
        for c in list(df_out.columns):
            if c in ("path_key",):
                continue
            if str(c).lower() in ("path_rel", "path_abs"):
                df_out.drop(columns=[c], inplace=True)

        # renommer colonnes metadata avec unités (ex: PbI2 [mol/L])
        rename_map = getattr(self, "_meta_export_map", None) or {}
        if rename_map:
            df_out.rename(columns=rename_map, inplace=True)

        # on n'exporte pas path_key (technique)
        if "path_key" in df_out.columns:
            df_out.drop(columns=["path_key"], inplace=True)

        # export RESULTS.xlsx dans le dossier de metadata.xlsx
        out_path = self.here / "RESULTS.xlsx"
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                df_out.to_excel(w, index=False, sheet_name="RESULTS")
            messagebox.showinfo("Export", f"RESULTS.xlsx créé:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Export", f"Erreur export:\n{e}")
            return

        self.phase_results = df_out

        # ouvre review (fenêtre matplotlib) — optionnel
        try:
            self._phase_open_review_window()
        except Exception as e:
            messagebox.showwarning("Review", f"Review non lancé:\n{e}")

    def _phase_open_review_window(self):
        """Review interactif (type fenêtre 5) sur tous les scans calculés."""
        if not self.phase_review_data:
            return

        idx = {"i": 0}

        fig = plt.figure(figsize=(10.5, 7.5))
        fig.canvas.manager.set_window_title("Review — Proportions (aires Voigt)")
        gs = GridSpec(2, 1, height_ratios=[3.0, 0.9], hspace=0.18)
        ax = fig.add_subplot(gs[0])
        axb = fig.add_subplot(gs[1])

        # boutons
        bax_prev = fig.add_axes([0.78, 0.93, 0.08, 0.05])
        bax_next = fig.add_axes([0.88, 0.93, 0.08, 0.05])
        bprev = MplButton(bax_prev, "Prev")
        bnext = MplButton(bax_next, "Next")

        # important: garder les widgets au-dessus (sinon clic parfois ignoré)
        bax_prev.set_zorder(10)
        bax_next.set_zorder(10)

        def draw_one(i):
            ax.clear(); axb.clear()
            d = self.phase_review_data[i]
            x = d["x"]; y = d["y"]; y_bg = d["y_bg"]
            name = d.get("label", d.get("name", f"{i+1}"))

            if len(x) and len(y):
                ax.plot(x, y, lw=1.0, color="#777777", alpha=0.6, label="Signal")
                if len(y_bg) == len(y):
                    ax.plot(x, y_bg, lw=1.2, color="0.35", alpha=0.9, label="Baseline (envelope)")
            ax.set_title(f"{name}   ({i+1}/{len(self.phase_review_data)})")
            ax.set_xlabel("2θ (corr. ITO)")
            ax.set_ylabel("Intensity (a.u.)")

            # remplissage par pic
            for pk in d.get("per_peak", []):
                ph = pk.get("phase", "Autre/Unknown")
                col = phase_color_for(ph)
                if "xloc" in pk and pk.get("ok", False):
                    xloc = np.ravel(pk["xloc"])
                    ybase = np.ravel(pk.get("ybase", np.zeros_like(xloc)))
                    yfit = np.ravel(np.clip(pk.get("yfit", np.zeros_like(xloc)), 0, None))
                    ytop = ybase + yfit
                    ax.fill_between(xloc, ybase, ytop, color=col, alpha=0.25)

            # légende phases (unique)
            phases_seen = {}
            for ph, pct in sorted(d.get("perc", {}).items(), key=lambda kv: -kv[1]):
                if ph in phases_seen:
                    continue
                phases_seen[ph] = True
                ax.plot([], [], color=phase_color_for(ph), lw=3, alpha=0.6, label=f"{ph} ({pct:.1f}%)")
            ax.legend(loc="upper right", fontsize=8)

            # barre %
            perc = d.get("perc", {})
            phases = [p for p, _ in sorted(perc.items(), key=lambda kv: -kv[1])]
            vals = [float(perc[p]) for p in phases]
            if phases:
                axb.bar(range(len(phases)), vals, color=[phase_color_for(p) for p in phases])
                axb.set_xticks(range(len(phases)))
                axb.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
                # colore les labels comme les intégrales
                for tick, ph in zip(axb.get_xticklabels(), phases):
                    tick.set_color(phase_color_for(ph))
                axb.set_ylabel("%")
                axb.set_ylim(0, max(100.0, (max(vals) * 1.15 if vals else 100.0)))
            axb.set_title("Proportions par phase")
            fig.canvas.draw_idle()

        def on_prev(event):
            idx["i"] = (idx["i"] - 1) % len(self.phase_review_data)
            draw_one(idx["i"])

        def on_next(event):
            idx["i"] = (idx["i"] + 1) % len(self.phase_review_data)
            draw_one(idx["i"])

        cid_prev = bprev.on_clicked(on_prev)
        cid_next = bnext.on_clicked(on_next)

        def on_key(event):
            if event.key in ("left", "pageup"):
                on_prev(event)
            elif event.key in ("right", "pagedown"):
                on_next(event)
            elif event.key == "home":
                idx["i"] = 0
                draw_one(0)

        cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

        # garder des références (sinon les widgets peuvent être GC → boutons "morts")
        fig._phase_review_nav = {
            "bprev": bprev, "bnext": bnext,
            "cid_prev": cid_prev, "cid_next": cid_next, "cid_key": cid_key,
        }


        draw_one(0)
        fig.canvas.draw_idle()
        plt.show(block=False)

# --------- éditeur de couleurs ---------
    def _edit_colors(self):
        self._line_to_label = {}
        group_key = self.color_by_var.get()
        df = self.df.copy()
        mask = self._get_active_masks(df)
        df = df[mask]
        df["_group"] = df.apply(lambda r: canon_group(self._make_group_value(r, group_key)), axis=1)
        uniques_sorted = sorted(pd.unique(df["_group"]), key=sort_key_value)

        win = tk.Toplevel(self); win.title(f"Couleurs — group by {group_key}"); win.geometry("380x420")
        lb = tk.Listbox(win, exportselection=False)
        for g in uniques_sorted:
            lb.insert(tk.END, str(g))
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def pick():
            if not lb.curselection(): return
            g_val = uniques_sorted[lb.curselection()[0]]
            _, hexcol = colorchooser.askcolor(title=f"Couleur pour {g_val}")
            if hexcol:
                self.group_color_overrides[(group_key, g_val)] = hexcol
                self._update_plot()

        def reset():
            if not lb.curselection(): return
            g_val = uniques_sorted[lb.curselection()[0]]
            self.group_color_overrides.pop((group_key, g_val), None)
            self._update_plot()

        row = ttk.Frame(win); row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="Choisir couleur…", command=pick).pack(side=tk.LEFT)
        ttk.Button(row, text="Réinitialiser", command=reset).pack(side=tk.LEFT, padx=6)

    # --------- export ---------
    def _export_png(self):
        out = filedialog.asksaveasfilename(
            title="Exporter en PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png")], initialfile="gonio_multi.png",
        )
        if not out: return
        try:
            self.fig.savefig(out, dpi=200)
            messagebox.showinfo("Export", f"Image sauvegardée:\n{out}")
        except Exception as e:
            messagebox.showerror("Erreur export", str(e))

if __name__ == "__main__":
    app = GonioApp()
    app.mainloop()

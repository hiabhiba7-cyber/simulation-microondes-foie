
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Projet M115",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Propagation des micro-ondes dans une tumeur hépatique")
st.subheader("Mini-projet M115 - GSR1")

# Constantes physiques
EPS0 = 8.854e-12       # Permittivite du vide (F/m)
MU0 = 4 * np.pi * 1e-7 # Permeabilite du vide (H/m)
C0 = 1 / np.sqrt(EPS0 * MU0)

# ==========================================
# 2. DONNEES DIELECTRIQUES
# ==========================================

frequences = [0.915, 2.45]

# Foie sain
epsilon_sain = [59.94, 57.55]
sigma_sain = [1.16, 1.95]

# Foie tumoral
epsilon_tumeur = [64.09, 62.44]
sigma_tumeur = [1.34, 2.18]


# ==========================================
# 3. CALCUL DES PROPRIETES ELECTROMAGNETIQUES
# ==========================================

def calcul_proprietes(f_GHz, epsilon_r, sigma):

    f = f_GHz * 1e9
    omega = 2 * np.pi * f

    # Permittivite complexe
    epsilon_complexe = (
        epsilon_r - 1j * sigma / (omega * EPS0)
    )

    # Impedance intrinseque
    eta = np.sqrt(
        1j * omega * MU0 /
        (sigma + 1j * omega * EPS0 * epsilon_r)
    )

    # Constante de propagation
    gamma = np.sqrt(
        1j * omega * MU0 *
        (sigma + 1j * omega * EPS0 * epsilon_r)
    )

    # Coefficient d'attenuation (Np/m)
    alpha = np.real(gamma)

    # Profondeur de penetration en amplitude (m)
    profondeur = 1 / alpha

    return epsilon_complexe, eta, alpha, profondeur


# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================

st.sidebar.header("Parametres de simulation")

f = st.sidebar.slider(
    "Frequence (GHz)",
    0.915, 2.45, 0.915,
    step=0.005
)

diametre = st.sidebar.slider(
    "Diametre de la tumeur (mm)",
    10, 30, 20
)

duree = st.sidebar.slider(
    "Duree de chauffage (s)",
    0, 600, 300
)

st.sidebar.write("Puissance de la source : a definir")

# ==========================================
# 5. INTERPOLATION DES DONNEES
# ==========================================

eps_h = np.interp(f, frequences, epsilon_sain)
sig_h = np.interp(f, frequences, sigma_sain)

eps_t = np.interp(f, frequences, epsilon_tumeur)
sig_t = np.interp(f, frequences, sigma_tumeur)

# Calculs pour les deux tissus
res_h = calcul_proprietes(f, eps_h, sig_h)
res_t = calcul_proprietes(f, eps_t, sig_t)

# ==========================================
# 6. AFFICHAGE DES PARAMETRES
# ==========================================

st.header("Parametres selectionnes")

col1, col2, col3 = st.columns(3)

col1.metric("Frequence", f"{f:.3f} GHz")
col2.metric("Diametre", f"{diametre} mm")
col3.metric("Duree", f"{duree} s")

# ==========================================
# 7. RESULTATS ELECTROMAGNETIQUES
# ==========================================

st.header("Etude electromagnetique")

col1, col2 = st.columns(2)

for colonne, nom, eps, sig, resultat in [
    (col1, "Foie sain", eps_h, sig_h, res_h),
    (col2, "Foie tumoral", eps_t, sig_t, res_t)
]:

    epsilon_c, eta, alpha, profondeur = resultat

    with colonne:
        st.subheader(nom)

        st.write("**Proprietes du tissu**")

        st.write(f"Permittivite relative : {eps:.3f}")
        st.write(f"Conductivite : {sig:.3f} S/m")

        st.write("**Permittivite complexe**")
        st.write(
            f"{epsilon_c.real:.3f} "
            f"- j {abs(epsilon_c.imag):.3f}"
        )

        st.write("**Impedance intrinseque**")
        st.write(f"{abs(eta):.3f} Ohm")

        st.write("**Coefficient d'attenuation**")
        st.metric("Alpha", f"{alpha:.3f} Np/m")

        st.write("**Profondeur de penetration**")
        st.metric(
            "Delta",
            f"{profondeur * 1000:.2f} mm"
        )

st.info(
    "Les proprietes entre 0.915 et 2.45 GHz "
    "sont interpolees lineairement entre deux "
    "frequences experimentales."
)

st.warning(
    "Cette premiere version calcule les proprietes "
    "electromagnetiques. Elle ne simule pas encore "
    "la distribution spatiale du champ ni la temperature."
)

# ==========================================
# 8. CHAMP ELECTRIQUE ET DAS
# ==========================================

st.header("Champ electrique et absorption electromagnetique")

# Hypotheses pedagogiques
P_source = 10.0       # Puissance supposee en watts
rho = 1050.0          # Masse volumique supposee en kg/m3
rayon_source = 0.005 # Rayon minimal du modele en metres

# Recuperation des proprietes electromagnetiques
eta_h = res_h[1]
alpha_h = res_h[2]

eta_t = res_t[1]
alpha_t = res_t[2]

# Rayon de la tumeur
R_t = diametre / 2000.0  # conversion mm vers metres

# Domaine de calcul : tumeur + marge de 5 mm
R_max = R_t + 0.005

# Creation de la grille spatiale 2D
N = 301

x = np.linspace(-R_max, R_max, N)
y = np.linspace(-R_max, R_max, N)

X, Y = np.meshgrid(x, y)

# Distance au centre de la source
r = np.sqrt(X**2 + Y**2)

# Eviter la singularite au centre
r_calcul = np.maximum(r, rayon_source)

# Identifier les deux milieux
dans_tumeur = r <= R_t

sigma_locale = np.where(dans_tumeur, sig_t, sig_h)
eta_locale = np.where(dans_tumeur, eta_t, eta_h)

# Attenuation cumulee sur le trajet radial
chemin_attenuation = np.where(
    dans_tumeur,
    alpha_t * r_calcul,
    alpha_t * R_t + alpha_h * (r_calcul - R_t)
)

# Densite de puissance approximative (W/m2)
S = (
    P_source / (4 * np.pi * r_calcul**2)
    * np.exp(-2 * chemin_attenuation)
)

# Champ electrique efficace (V/m)
conductance_onde = np.real(1 / eta_locale)

E = np.sqrt(S / conductance_onde)

# Debit d'absorption specifique (W/kg)
DAS = sigma_locale * E**2 / rho

# Masquer la source centrale
zone_source = r < rayon_source

E = np.where(zone_source, np.nan, E)
DAS = np.where(zone_source, np.nan, DAS)

# ==========================================
# 9. AFFICHAGE DES CARTES


# ==========================================
# 9. CARTES DE PROPAGATION ET D'ABSORPTION
# ==========================================

st.header("Cartes de propagation et d'absorption")

# Limites de la tumeur et de la zone saine
st.write(
    "Le cercle représente la tumeur. "
    "La region exterieure represente le foie sain."
)

# ------------------------------------------
# CARTE 1 : CHAMP ELECTRIQUE
# ------------------------------------------

st.subheader("1. Distribution spatiale du champ electrique")

fig1, ax1 = plt.subplots(figsize=(8, 6))

im1 = ax1.contourf(
    X * 1000,
    Y * 1000,
    E,
    levels=30,
    cmap="turbo"
)

c1 = plt.Circle(
    (0, 0),
    R_t * 1000,
    color="white",
    fill=False,
    linewidth=2
)

ax1.add_patch(c1)

fig1.colorbar(
    im1,
    ax=ax1,
    label="Champ electrique (V/m)"
)

ax1.set_title("Propagation du champ electrique")
ax1.set_xlabel("Distance X (mm)")
ax1.set_ylabel("Distance Y (mm)")
ax1.set_aspect("equal")

st.pyplot(fig1)
plt.close(fig1)

# ------------------------------------------
# CARTE 2 : DAS
# ------------------------------------------

st.subheader("2. Distribution spatiale du DAS")

fig2, ax2 = plt.subplots(figsize=(8, 6))

im2 = ax2.contourf(
    X * 1000,
    Y * 1000,
    DAS,
    levels=30,
    cmap="hot"
)

c2 = plt.Circle(
    (0, 0),
    R_t * 1000,
    color="cyan",
    fill=False,
    linewidth=2
)

ax2.add_patch(c2)

fig2.colorbar(
    im2,
    ax=ax2,
    label="DAS (W/kg)"
)

ax2.set_title("Absorption de l'energie electromagnetique")
ax2.set_xlabel("Distance X (mm)")
ax2.set_ylabel("Distance Y (mm)")
ax2.set_aspect("equal")

st.pyplot(fig2)
plt.close(fig2)

# ------------------------------------------
# CARTE 3 : PROFILS RADIAUX
# ------------------------------------------

st.subheader("3. Evolution du champ et du DAS avec la distance")

# Extraction d'une ligne passant par le centre
indice_centre = N // 2

distance = x * 1000

champ_profil = E[indice_centre, :]
das_profil = DAS[indice_centre, :]

# On conserve uniquement la partie positive
masque = (distance >= rayon_source * 1000)

fig3, ax3 = plt.subplots(figsize=(9, 5))

ax3.plot(
    distance[masque],
    champ_profil[masque],
    label="Champ electrique",
    linewidth=2
)

ax3.axvline(
    R_t * 1000,
    linestyle="--",
    label="Limite de la tumeur"
)

ax3.set_title("Diminution du champ electrique")
ax3.set_xlabel("Distance depuis le centre (mm)")
ax3.set_ylabel("Champ electrique (V/m)")
ax3.grid(True)
ax3.legend()

st.pyplot(fig3)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(9, 5))

ax4.plot(
    distance[masque],
    das_profil[masque],
    label="DAS",
    linewidth=2
)

ax4.axvline(
    R_t * 1000,
    linestyle="--",
    label="Limite de la tumeur"
)

ax4.set_title("Evolution du DAS avec la distance")
ax4.set_xlabel("Distance depuis le centre (mm)")
ax4.set_ylabel("DAS (W/kg)")
ax4.grid(True)
ax4.legend()

st.pyplot(fig4)
plt.close(fig4)

# ------------------------------------------
# INFORMATIONS
# ------------------------------------------

st.subheader("Parametres de la representation")

st.write(f"Frequence : {f:.3f} GHz")
st.write(f"Diametre tumoral : {diametre} mm")
st.write(f"Puissance supposee : {P_source} W")
st.write("Marge de foie sain representee : 5 mm")

st.info(
    "Ces cartes representent une coupe 2D d'un modele "
    "de propagation radiale simplifie. Elles ne constituent "
    "pas une simulation electromagnetique medicale validee."
)


# ==========================================
# 12. SIMULATION THERMIQUE
# ==========================================

st.header("Simulation thermique des tissus")

st.write(
    "Etude de l'evolution de la temperature "
    "dans la tumeur et le foie sain."
)

# ------------------------------------------
# A. PROPRIETES THERMIQUES
# ------------------------------------------

T_initiale = 37.0       # Temperature initiale en degres C
T_seuil = 60.0          # Seuil thermique indicatif

rho_th = 1050.0         # Masse volumique kg/m3
Cp = 3600.0             # Capacite thermique J/(kg.K)
k_th = 0.5              # Conductivite thermique W/(m.K)

# Diffusivite thermique
diffusivite = k_th / (rho_th * Cp)

# ------------------------------------------
# B. PREPARATION DE LA GRILLE
# ------------------------------------------

# Reduction de la grille pour accelerer le calcul
pas = max(1, N // 100)

DAS_th = np.nan_to_num(
    DAS[::pas, ::pas],
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_th = X[::pas, ::pas]
Y_th = Y[::pas, ::pas]

distance_th = np.sqrt(X_th**2 + Y_th**2)

# Masque de la tumeur
masque_tumeur = distance_th <= R_t

# Masque du foie sain dans le domaine etudie
masque_foie = (
    (distance_th > R_t)
    & (distance_th <= R_max)
)

# Distance entre deux points de la grille
dx = abs(x[pas] - x[0]) if pas < N else abs(x[1] - x[0])

# ------------------------------------------
# C. INITIALISATION DE LA TEMPERATURE
# ------------------------------------------

temperature = np.full(
    DAS_th.shape,
    T_initiale,
    dtype=float
)

# Pas de temps stable pour la methode numerique
dt = min(
    0.25,
    0.20 * dx**2 / (4 * diffusivite)
)

# Nombre d'iterations
nombre_iterations = int(duree / dt)

# Historique des temperatures
temps_historique = [0.0]

temperature_tumeur = [
    np.mean(temperature[masque_tumeur])
]

temperature_foie = [
    np.mean(temperature[masque_foie])
]

# ------------------------------------------
# D. CALCUL DE LA CHALEUR
# ------------------------------------------

for i in range(nombre_iterations):

    # Extension des bords : flux thermique nul
    T_pad = np.pad(
        temperature,
        1,
        mode="edge"
    )

    # Laplacien thermique en 2D
    laplacien = (
        T_pad[1:-1, 2:]
        + T_pad[1:-1, :-2]
        + T_pad[2:, 1:-1]
        + T_pad[:-2, 1:-1]
        - 4 * temperature
    ) / dx**2

    # Equation de la chaleur
    temperature += dt * (
        diffusivite * laplacien
        + DAS_th / Cp
    )

    # Enregistrement des resultats
    intervalle = max(1, nombre_iterations // 200)

    if (i + 1) % intervalle == 0:
        temps_actuel = (i + 1) * dt

        temps_historique.append(temps_actuel)

        temperature_tumeur.append(
            np.mean(temperature[masque_tumeur])
        )

        temperature_foie.append(
            np.mean(temperature[masque_foie])
        )

# ------------------------------------------
# E. CARTE THERMIQUE
# ------------------------------------------

st.subheader("1. Carte thermique finale")

fig5, ax5 = plt.subplots(figsize=(8, 6))

im5 = ax5.contourf(
    X_th * 1000,
    Y_th * 1000,
    temperature,
    levels=30,
    cmap="hot"
)

cercle = plt.Circle(
    (0, 0),
    R_t * 1000,
    fill=False,
    color="cyan",
    linewidth=2
)

ax5.add_patch(cercle)

fig5.colorbar(
    im5,
    ax=ax5,
    label="Temperature (°C)"
)

ax5.set_title(
    f"Distribution thermique apres {duree} secondes"
)

ax5.set_xlabel("Position X (mm)")
ax5.set_ylabel("Position Y (mm)")
ax5.set_aspect("equal")

st.pyplot(fig5)
plt.close(fig5)

# ------------------------------------------
# F. EVOLUTION DE LA TEMPERATURE
# ------------------------------------------

st.subheader("2. Evolution de la temperature")

fig6, ax6 = plt.subplots(figsize=(9, 5))

ax6.plot(
    temps_historique,
    temperature_tumeur,
    label="Temperature moyenne de la tumeur",
    linewidth=2
)

ax6.plot(
    temps_historique,
    temperature_foie,
    label="Temperature moyenne du foie sain",
    linewidth=2
)

ax6.axhline(
    T_seuil,
    linestyle="--",
    label="Seuil indicatif de 60 °C"
)

ax6.set_xlabel("Temps (secondes)")
ax6.set_ylabel("Temperature (°C)")
ax6.set_title("Evolution thermique des deux tissus")
ax6.grid(True)
ax6.legend()

st.pyplot(fig6)
plt.close(fig6)

# ------------------------------------------
# G. POURCENTAGE DE LA TUMEUR A 60 °C
# ------------------------------------------

st.subheader("3. Evaluation du seuil thermique")

nombre_total = np.count_nonzero(masque_tumeur)

nombre_chauffe = np.count_nonzero(
    temperature[masque_tumeur] >= T_seuil
)

pourcentage = (
    100 * nombre_chauffe / nombre_total
    if nombre_total > 0 else 0
)

st.metric(
    "Volume tumoral ayant atteint 60 °C",
    f"{pourcentage:.2f} %"
)

st.metric(
    "Temperature maximale",
    f"{np.max(temperature[masque_tumeur]):.2f} °C"
)

st.metric(
    "Temperature moyenne de la tumeur",
    f"{np.mean(temperature[masque_tumeur]):.2f} °C"
)

st.metric(
    "Temperature moyenne du foie sain",
    f"{np.mean(temperature[masque_foie]):.2f} °C"
)

# Barre de progression du chauffage tumoral

st.subheader("Progression du chauffage tumoral")

st.progress(
    min(1.0, max(0.0, pourcentage / 100))
)

st.write(
    f"{pourcentage:.2f} % du volume tumoral "
    "a atteint 60 °C."
)

if pourcentage >= 99.9:
    st.success("Seuil thermique atteint dans presque toute la tumeur.")
else:
    st.info("Le chauffage de la tumeur reste incomplet.")


# ------------------------------------------
# H. INTERPRETATION
# ------------------------------------------

st.subheader("4. Interpretation des resultats")

if pourcentage >= 99:
    st.success(
        "Presque toute la tumeur a atteint le seuil "
        "thermique indicatif de 60 °C."
    )
elif pourcentage > 0:
    st.warning(
        "Une partie de la tumeur a atteint 60 °C, "
        "mais le chauffage reste incomplet."
    )
else:
    st.info(
        "Aucune partie de la tumeur n'a encore atteint 60 °C."
    )

st.caption(
    "Modele pedagogique simplifie : conduction thermique "
    "et absorption electromagnetique. La perfusion sanguine, "
    "les changements de proprietes avec la temperature et "
    "les effets biologiques ne sont pas modelises. "
    "Les resultats ne constituent pas un outil medical."
)

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Ustawienia strony aplikacji
st.set_page_config(page_title="Predyktor Opóźnień JFK-MIA", layout="centered")

st.title("✈️ Predykcja Opóźnień Lotów: JFK -> MIA")
st.write("Wprowadź parametry lotu oraz prognozę pogody, aby sprawdzić przewidywane opóźnienie przylotu.")

# 1. ŁADOWANIE MODELU Z PLIKU JSON
@st.cache_resource
def wczytaj_model():
    model = XGBRegressor()
    model.load_model('model_xgboost.json')
    return model

try:
    model = wczytaj_model()
except Exception as e:
    st.error(f"Nie udało się załadować modelu 'model_xgboost.json'. Upewnij się, że plik jest w tym samym folderze. Błąd: {e}")
    st.stop()

# 2. INTERFEJS UŻYTKOWNIKA (Formularz wejściowy)
st.header("📋 Dane wejściowe lotu")

# Podział na sekcje za pomocą kontenerów
with st.container():
    st.subheader("🗓️ Czas i kalendarz")
    month = st.slider("Miesiąc", min_value=1, max_value=12, value=9)
    day_of_week = st.slider("Dzień tygodnia (1 = Poniedziałek, 7 = Niedziela)", min_value=1, max_value=7, value=3)
    crs_dep_time = st.number_input("Planowana godzina wylotu (format HHMM, np. 1600 dla 16:00)", min_value=0, max_value=2359, value=1600)
    crs_arr_time = st.number_input("Planowana godzina przylotu (format HHMM, np. 1915 dla 19:15)", min_value=0, max_value=2359, value=1915)

with st.container():
    st.subheader("🗽 Pogoda w Nowym Jorku (JFK)")
    ny_prcp = st.number_input("Opady deszczu/śniegu w NYC (mm)", min_value=0.0, value=0.0, step=0.1)
    ny_awnd = st.number_input("Średnia prędkość wiatru w NYC (m/s)", min_value=0.0, value=8.5, step=0.5)
    ny_tmp_c = st.number_input("Temperatura w NYC (°C)", value=20.0, step=0.5)

with st.container():
    st.subheader("🌴 Pogoda w Miami (MIA)")
    mia_wnd = st.slider("Kierunek wiatru w MIA (w stopniach: 0-360°)", min_value=0.0, max_value=360.0, value=160.0)
    mia_cig = st.number_input("Podstawa chmur w MIA (w stopach)", min_value=0.0, value=800.0, step=100.0)
    mia_vis = st.number_input("Widoczność w MIA (w metrach)", min_value=0.0, value=2000.0, step=500.0)
    mia_tmp_c = st.number_input("Temperatura w MIA (°C)", value=25.0, step=0.5)

# 3. PRZETWARZANIE DANYCH WEJŚCIOWYCH
if st.button("🚀 Przewidź opóźnienie lotu", use_container_width=True):
    
    # Obliczanie sinusa i cosinusa kierunku wiatru dla Miami
    mia_wnd_rad = np.deg2rad(mia_wnd)
    
    # Tworzenie słownika z danymi
    dane_wejsciowe = {
        'month': [month],
        'day_of_week': [day_of_week],
        'crs_dep_time': [crs_dep_time],
        'crs_arr_time': [crs_arr_time],
        'NY_PRCP': [ny_prcp],
        'NY_AWND': [ny_awnd],
        'NY_TMP_C': [ny_tmp_c],
        'MIA_CIG': [mia_cig],
        'MIA_VIS': [mia_vis],
        'MIA_TMP_C': [mia_tmp_c],
        'MIA_WND_sin': [np.sin(mia_wnd_rad)],
        'MIA_WND_cos': [np.cos(mia_wnd_rad)]
    }
    
    # Konwersja na DataFrame
    df_input = pd.DataFrame(dane_wejsciowe)
    
    # KRYTYCZNY KROK: Wymuszenie identycznej kolejności kolumn jak w X_train
    kolejnosc_kolumn = [
        'month', 'day_of_week', 'crs_dep_time', 'crs_arr_time',
        'NY_PRCP', 'NY_AWND', 'NY_TMP_C', 'MIA_CIG', 'MIA_VIS',
        'MIA_TMP_C', 'MIA_WND_sin', 'MIA_WND_cos'
    ]
    df_input = df_input.reindex(columns=kolejnosc_kolumn)
    
    # 4. PREDYKCJA I WYŚWIETLENIE WYNIKU
    predykcja = model.predict(df_input)[0]
    
    st.write("---")
    st.subheader("📊 Wynik analizy modelu:")
    
    # Atrakcyjne wizualnie wyświetlenie wyniku za pomocą st.metric
    if predykcja < 0:
        st.metric(label="Przewidywany status przylotu", value=f"Przed czasem o {abs(predykcja):.1f} min")
        st.success("Świetne warunki! Lot powinien wylądować bez problemów.")
    elif predykcja <= 15:
        st.metric(label="Przewidywane opóźnienie przylotu", value=f"{predykcja:.1f} minut")
        st.success("Standardowe, nieznaczne przesunięcie czasowe. Lot mieści się w normie.")
    elif predykcja <= 45:
        st.metric(label="Przewidywane opóźnienie przylotu", value=f"{predykcja:.1f} minut")
        st.warning("Umiarkowane opóźnienie. Możliwe zatory na lotnisku docelowym.")
    else:
        st.metric(label="Przewidywane opóźnienie przylotu", value=f"{predykcja:.1f} minut")
        st.error("🚨 Wysokie ryzyko poważnego opóźnienia! Warunki meteorologiczne drastycznie ograniczają przepustowość.")
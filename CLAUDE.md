# CLAUDE.md

## Projekt
Interaktives Streamlit-Dashboard zur Analyse von Airbnb-Inseratsdaten in Texas.
Zielgruppe: nicht-technische Nutzer (Kunden-Demo). Jedes Dashboard muss demo-reif sein – der Kunde sieht es.

## Tech-Stack
- Streamlit + Pandas + Plotly Express
- Karten: Pydeck oder Folium
- NEVER matplotlib

## Regeln
- ALWAYS read the `developing-with-streamlit` skill before writing any Streamlit code.
- ALWAYS use `st.cache_data` for data loading.
- Alle Preise in USD (z. B. `$1,234`).
- Texte, Titel, Achsen und Tooltips verständlich für Nicht-Techniker – keine Spaltennamen wie `price_usd` oder Fachjargon in der UI.

## Pflicht-Aufbau jedes Dashboards
1. Filter in der Sidebar
2. KPI-Karten oben
3. Diagramme – alle MÜSSEN sich aktualisieren, wenn Filter geändert werden
4. Ein Data-Explorer-Tab mit den gefilterten Daten
5. CSV-Download-Button (exportiert die gefilterten Daten)

## Robustheit
- Die App darf bei leeren Filtern / leerem Ergebnis nicht abstürzen – stattdessen einen freundlichen Hinweis anzeigen (z. B. `st.info`) und `st.stop()`.

## Verifikation
- ALWAYS use the virtual environment in .venv for all Python commands.
  Start the app with: .venv\Scripts\python.exe -m streamlit run app.py
  Install packages only with: .venv\Scripts\python.exe -m pip install ...
  NEVER install packages globally.

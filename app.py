"""Dashboard zum Immobilienmarkt in Texas – Analyse von Airbnb-Inseraten."""

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from pricing_model import MAX_BEDROOMS, OTHER_CITY, PriceModel, train_price_model

DATA_PATH = Path(__file__).parent / "Airbnb_Texas_Rentals.csv"

# Farben: eine Hauptfarbe für Einzelreihen, eine geordnete Blau-Skala für Stufen
ACCENT = "#2a78d6"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

PRICE_BINS = [0, 75, 150, 300, float("inf")]
PRICE_CLASSES = ["bis $75", "$76 – $150", "$151 – $300", "über $300"]
PRICE_CLASS_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

PRICE_STEPS = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, 10000]

MONTHS_DE = [
    "Jan.", "Feb.", "März", "Apr.", "Mai", "Juni",
    "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez.",
]

st.set_page_config(
    page_title="Immobilienmarkt Texas",
    page_icon="🏠",
    layout="wide",
)


_DE_SEPARATORS = str.maketrans(",.", ".,")


def num(value: float, digits: int = 0) -> str:
    """Zahl in deutscher Schreibweise: 18.228 bzw. 6,8."""
    return f"{value:,.{digits}f}".translate(_DE_SEPARATORS)


def pct(share: float, digits: int = 1) -> str:
    """Anteil (0–1) als deutsche Prozentangabe, z. B. 0.068 -> „6,8 %“."""
    return f"{num(share * 100, digits)} %"


def usd(value: float) -> str:
    sign = "-" if round(value) < 0 else ""
    return f"{sign}${num(abs(value))}"


def month_label(ts: pd.Timestamp) -> str:
    return f"{MONTHS_DE[ts.month - 1]} {ts.year}"


def bedroom_label(n: int) -> str:
    return "Studio" if n == 0 else str(n)


BEDROOM_INPUT_OPTIONS = list(range(MAX_BEDROOMS + 1))


def bedroom_input_label(n: int) -> str:
    return f"{n}+" if n == MAX_BEDROOMS else bedroom_label(n)


@st.cache_data(show_spinner="Daten werden geladen …")
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    df = pd.DataFrame(
        {
            "title": raw["title"].fillna("Ohne Titel"),
            "city": raw["city"].str.strip(),
            "price": pd.to_numeric(
                raw["average_rate_per_night"].str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            ),
            "bedrooms": pd.to_numeric(
                raw["bedrooms_count"].replace("Studio", "0"), errors="coerce"
            ),
            "listed": pd.to_datetime(
                raw["date_of_listing"], format="%B %Y", errors="coerce"
            ),
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "description": raw["description"].fillna(""),
            "url": raw["url"],
        }
    )
    # Inserate ohne Preis, Zimmerzahl oder Datum lassen sich nicht auswerten
    df = df.dropna(subset=["price", "bedrooms", "listed"])
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bedroom_label"] = df["bedrooms"].map(bedroom_label)
    df["price_class"] = pd.cut(
        df["price"], bins=PRICE_BINS, labels=PRICE_CLASSES, right=True
    )
    # Alter des Inserats in Monaten, gemessen am Datenstand (jüngstes Inserat)
    ref = df["listed"].max()
    df["age_months"] = (ref.year - df["listed"].dt.year) * 12 + (
        ref.month - df["listed"].dt.month
    )
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Preismodell wird trainiert …")
def load_price_model() -> PriceModel:
    return train_price_model(load_data())


df = load_data()
DATA_AS_OF = month_label(df["listed"].max())
all_cities = df["city"].value_counts().index.tolist()  # häufigste Städte zuerst
all_bedrooms = sorted(df["bedrooms"].unique())
all_months = [pd.Timestamp(m) for m in sorted(df["listed"].unique())]


FILTER_KEYS = ("f_city", "f_bedrooms", "f_price", "f_months")


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header(":material/tune: Filter")

    cities = st.multiselect(
        "Stadt",
        options=all_cities,
        default=[],
        key="f_city",
        placeholder="Alle Städte",
        help="Leer lassen, um alle Städte anzuzeigen.",
    )

    bedrooms_sel = st.pills(
        "Anzahl der Schlafzimmer",
        options=[bedroom_label(b) for b in all_bedrooms],
        selection_mode="multi",
        default=[bedroom_label(b) for b in all_bedrooms],
        key="f_bedrooms",
    )

    price_min, price_max = st.select_slider(
        "Preis pro Nacht",
        options=PRICE_STEPS,
        value=(PRICE_STEPS[0], PRICE_STEPS[-1]),
        key="f_price",
        format_func=usd,
    )

    month_from, month_to = st.select_slider(
        "Inseriert zwischen",
        options=all_months,
        value=(all_months[0], all_months[-1]),
        key="f_months",
        format_func=month_label,
    )

    st.button(
        "Filter zurücksetzen",
        icon=":material/restart_alt:",
        on_click=reset_filters,
        width="stretch",
    )

# ---------------------------------------------------------------- Filtern
mask = (
    df["bedroom_label"].isin(bedrooms_sel)
    & df["price"].between(price_min, price_max)
    & df["listed"].between(month_from, month_to)
)
if cities:
    mask &= df["city"].isin(cities)
filtered = df[mask]

# ---------------------------------------------------------------- Kopf
st.title("🏠 Dashboard zum Immobilienmarkt in Texas")
st.caption(
    f"**{num(len(filtered))}** von {num(len(df))} Airbnb-Inseraten entsprechen den "
    "aktuellen Filtern."
)

if not bedrooms_sel:
    st.info(
        "Bitte wählen Sie in der Seitenleiste mindestens eine Schlafzimmeranzahl aus.",
        icon=":material/bed:",
    )
    st.stop()

if filtered.empty:
    st.info(
        "Für diese Filterauswahl gibt es keine Inserate. Erweitern Sie z. B. die "
        "Preisspanne oder den Zeitraum – oder setzen Sie die Filter zurück.",
        icon=":material/search_off:",
    )
    st.stop()

price_model = load_price_model()

PLOT_LAYOUT = dict(
    margin=dict(l=10, r=10, t=10, b=10),
    separators=",.",  # deutsche Schreibweise: 1.234 und 51,4
)

tab_overview, tab_map, tab_prices, tab_predict, tab_invest, tab_data = st.tabs(
    [
        ":material/dashboard: Übersicht",
        ":material/map: Kartenansicht",
        ":material/payments: Preisanalyse",
        ":material/auto_graph: Preisprognose",
        ":material/real_estate_agent: Renditecheck",
        ":material/table: Daten-Explorer",
    ]
)

# ---------------------------------------------------------------- Übersicht
with tab_overview:
    city_counts = filtered["city"].value_counts()
    top_city = city_counts.index[0]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Angebote gesamt", f"{num(len(filtered))}", border=True)
    kpi2.metric("Durchschnittspreis", usd(filtered["price"].mean()), border=True,
                help="Durchschnittlicher Preis pro Nacht")
    kpi3.metric("Medianpreis", usd(filtered["price"].median()), border=True,
                help="Die Hälfte der Inserate ist günstiger, die andere Hälfte teurer.")
    kpi4.metric("Beliebteste Stadt", top_city, border=True,
                help=f"Stadt mit den meisten Inseraten ({num(city_counts.iloc[0])})")

    col_left, col_right = st.columns([3, 2])

    with col_left, st.container(border=True):
        top_n = min(15, len(city_counts))
        st.subheader(f"Durchschnittspreis pro Nacht – Top {top_n} Städte")
        st.caption("Die Städte mit den meisten Inseraten")
        by_city = (
            filtered[filtered["city"].isin(city_counts.index[:top_n])]
            .groupby("city")
            .agg(avg_price=("price", "mean"), listings=("price", "size"))
            .reset_index()
            .sort_values("avg_price")
        )
        fig = px.bar(
            by_city,
            x="avg_price",
            y="city",
            orientation="h",
            custom_data=["listings"],
            color_discrete_sequence=[ACCENT],
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Ø Preis: $%{x:,.0f} pro Nacht"
            "<br>Inserate: %{customdata[0]:,}<extra></extra>",
            marker_cornerradius=4,
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=max(260, 32 * top_n),
            xaxis=dict(title="Ø Preis pro Nacht", tickprefix="$", tickformat=",.0f"),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig, alt="Durchschnittspreis pro Nacht in den größten Städten")

    with col_right, st.container(border=True):
        st.subheader("Verteilung nach Schlafzimmern")
        st.caption("Anteil der Inserate je Schlafzimmeranzahl")
        bed_groups = filtered["bedrooms"].clip(upper=5).map(
            lambda n: "5 und mehr" if n == 5 else bedroom_label(n)
        )
        order = ["Studio", "1", "2", "3", "4", "5 und mehr"]
        pie_data = (
            bed_groups.value_counts().reindex(order).dropna().rename_axis("group")
            .reset_index(name="count")
        )
        pie_data["group"] = pie_data["group"].map(
            lambda g: g if g in ("Studio", "5 und mehr") else f"{g} Schlafzimmer"
        )
        fig = px.pie(
            pie_data,
            names="group",
            values="count",
            color_discrete_sequence=BLUE_RAMP,
            category_orders={"group": pie_data["group"].tolist()},
        )
        fig.update_traces(
            sort=False,
            direction="clockwise",
            texttemplate="%{percent:.1%}",
            textposition="inside",
            insidetextorientation="horizontal",
            marker_line=dict(color="white", width=2),
            hovertemplate="<b>%{label}</b><br>%{value:,} Inserate (%{percent:.1%})<extra></extra>",
        )
        fig.update_layout(**PLOT_LAYOUT, height=400, legend_title_text=None)
        st.plotly_chart(fig, alt="Anteil der Inserate nach Anzahl der Schlafzimmer")

# ---------------------------------------------------------------- Karte
with tab_map:
    map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        st.info("Für die gefilterten Inserate liegen keine Standortdaten vor.",
                icon=":material/location_off:")
    else:
        color_lookup = {
            cls: [int(c[i : i + 2], 16) for i in (1, 3, 5)] + [210]
            for cls, c in zip(PRICE_CLASSES, PRICE_CLASS_COLORS)
        }
        map_df["color"] = map_df["price_class"].astype(str).map(color_lookup)
        map_df["tip_title"] = map_df["title"].map(escape)
        map_df["tip_city"] = map_df["city"].map(escape)
        map_df["tip_price"] = map_df["price"].map(usd)
        map_df["tip_bedrooms"] = map_df["bedroom_label"]

        lat_span = map_df["latitude"].max() - map_df["latitude"].min()
        lon_span = map_df["longitude"].max() - map_df["longitude"].min()
        span = max(lat_span, lon_span, 0.05)
        zoom = 5.2 if span > 6 else 6 if span > 3 else 7.5 if span > 1 else 9.5

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df[["longitude", "latitude", "color", "tip_title",
                         "tip_city", "tip_price", "tip_bedrooms"]],
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_line_color=[255, 255, 255, 200],
            stroked=True,
            line_width_min_pixels=1,
            get_radius=120,
            radius_min_pixels=4,
            radius_max_pixels=14,
            pickable=True,
            auto_highlight=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(map_df["latitude"].median()),
                longitude=float(map_df["longitude"].median()),
                zoom=zoom,
            ),
            tooltip={
                "html": "<b>{tip_title}</b><br/>"
                        "Preis: {tip_price} pro Nacht<br/>"
                        "Schlafzimmer: {tip_bedrooms}<br/>"
                        "Stadt: {tip_city}",
                "style": {"fontSize": "13px", "maxWidth": "320px"},
            },
        )

        legend = " ".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin-right:18px"><span style="width:12px;height:12px;'
            f'border-radius:50%;background:{c};display:inline-block"></span>'
            f"{cls}</span>"
            for cls, c in zip(PRICE_CLASSES, PRICE_CLASS_COLORS)
        )
        with st.container(border=True):
            st.subheader("Inserate auf der Karte")
            st.caption(
                f"{num(len(map_df))} Inserate mit Standort · Fahren Sie mit der Maus "
                "über einen Punkt für Details."
            )
            st.html(f'<div style="font-size:14px">Preis pro Nacht: {legend}</div>')
            st.pydeck_chart(deck, height=600,
                            alt="Karte der Inserate in Texas, eingefärbt nach Preisklasse")

# ---------------------------------------------------------------- Preise
with tab_prices:
    p99 = filtered["price"].quantile(0.99)
    col_left, col_right = st.columns(2)

    with col_left, st.container(border=True):
        st.subheader("Wie sind die Preise verteilt?")
        hist_data = filtered[filtered["price"] <= p99]
        st.caption(
            f"Anzahl der Inserate je Preisstufe · ohne die teuersten 1 % "
            f"(über {usd(p99)}), damit die Verteilung lesbar bleibt"
            if len(hist_data) < len(filtered)
            else "Anzahl der Inserate je Preisstufe"
        )
        fig = px.histogram(hist_data, x="price", nbins=40,
                           color_discrete_sequence=[ACCENT])
        fig.update_traces(
            hovertemplate="$%{x} pro Nacht<br>%{y:,} Inserate<extra></extra>",
            marker_line=dict(color="white", width=1),
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=380,
            bargap=0.05,
            xaxis=dict(title="Preis pro Nacht", tickprefix="$", tickformat=",.0f"),
            yaxis=dict(title="Anzahl Inserate"),
        )
        st.plotly_chart(fig, alt="Verteilung der Übernachtungspreise")

    with col_right, st.container(border=True):
        st.subheader("Preis nach Anzahl der Schlafzimmer")
        st.caption(
            "Die Box zeigt die mittleren 50 % der Preise, der Strich darin den "
            "Median · einzelne Ausreißer sind ausgeblendet"
        )
        box_order = [bedroom_label(b) for b in sorted(filtered["bedrooms"].unique())]
        fig = px.box(
            filtered,
            x="bedroom_label",
            y="price",
            points=False,
            category_orders={"bedroom_label": box_order},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=380,
            xaxis=dict(title="Schlafzimmer", type="category"),
            yaxis=dict(title="Preis pro Nacht", tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig, alt="Preisspanne je Anzahl der Schlafzimmer")

    with st.container(border=True):
        st.subheader("Neue Inserate pro Monat")
        st.caption("Wann wurden die Unterkünfte erstmals auf Airbnb angeboten?")
        monthly = (
            filtered.groupby("listed").size()
            .reindex(pd.date_range(filtered["listed"].min(),
                                   filtered["listed"].max(), freq="MS"),
                     fill_value=0)
            .rename_axis("month").reset_index(name="count")
        )
        monthly["label"] = monthly["month"].map(month_label)
        fig = px.line(monthly, x="month", y="count", custom_data=["label"],
                      color_discrete_sequence=[ACCENT])
        fig.update_traces(
            line_width=2,
            hovertemplate="%{customdata[0]}<br>%{y:,} neue Inserate<extra></extra>",
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=340,
            hovermode="x unified",
            xaxis=dict(title=None),
            yaxis=dict(title="Neue Inserate", rangemode="tozero"),
        )
        st.plotly_chart(fig, alt="Anzahl neuer Inserate pro Monat im Zeitverlauf")

# ---------------------------------------------------------------- Prognose
@st.fragment
def prediction_section() -> None:
    """Eingaben hier rechnen nur diesen Abschnitt neu, nicht das ganze Dashboard."""
    st.subheader("Was sollte eine Unterkunft pro Nacht kosten?")
    st.caption(
        "Geben Sie die Eckdaten ein – die Schätzung aktualisiert sich sofort. "
        "Das Modell hat aus allen Inseraten gelernt; das Vergleichsdiagramm rechts "
        "folgt Ihren Filtern in der Seitenleiste."
    )
    col_in, col_out = st.columns([1, 2])

    with col_in, st.container(border=True):
        st.markdown("**Eckdaten der Unterkunft**")
        p_city = st.selectbox(
            "Stadt",
            options=price_model.cities + [OTHER_CITY],
            key="p_city",
            help="Die 40 Städte mit den meisten Inseraten; alle übrigen sind unter "
                 f"„{OTHER_CITY}“ zusammengefasst.",
        )
        p_bed = st.select_slider(
            "Schlafzimmer", options=BEDROOM_INPUT_OPTIONS, value=2,
            format_func=bedroom_input_label, key="p_bed",
        )
        p_age = st.slider(
            "Seit wie vielen Monaten ist das Inserat online?",
            min_value=0, max_value=int(df["age_months"].max()), value=12,
            key="p_age",
            help=f"Gezählt bis zum Datenstand ({DATA_AS_OF}).",
        )

    pred, pred_low, pred_high = price_model.predict(p_city, p_bed, p_age)

    with col_out:
        m1, m2 = st.columns(2)
        m1.metric("Geschätzter Preis pro Nacht", usd(pred), border=True)
        m2.metric(
            "Üblicher Preisbereich", f"{usd(pred_low)} – {usd(pred_high)}", border=True,
            help="8 von 10 vergleichbaren Inseraten liegen in diesem Bereich "
                 "(gemessen an Inseraten, die das Modell beim Lernen nicht gesehen hat).",
        )

        with st.container(border=True):
            st.markdown("**Tatsächliche Preise vergleichbarer Inserate**")
            same_bed = filtered["bedrooms"].clip(upper=MAX_BEDROOMS) == p_bed
            same_city = (
                ~filtered["city"].isin(price_model.cities)
                if p_city == OTHER_CITY
                else filtered["city"] == p_city
            )
            comp = filtered[same_bed & same_city]
            if len(comp) < 3:
                st.info(
                    "In den gefilterten Daten gibt es zu wenige vergleichbare Inserate "
                    f"({len(comp)}). Erweitern Sie die Filter in der Seitenleiste, um "
                    "die Schätzung mit echten Preisen zu vergleichen.",
                    icon=":material/filter_alt_off:",
                )
            else:
                x_max = max(comp["price"].quantile(0.95), pred_high * 1.2)
                shown = comp[comp["price"] <= x_max]
                st.caption(
                    f"{num(len(comp))} Inserate · Median {usd(comp['price'].median())} · "
                    "der blaue Bereich zeigt den üblichen Preisbereich der Prognose"
                )
                fig = px.histogram(shown, x="price", nbins=30,
                                   color_discrete_sequence=["#9ec5f4"])
                fig.update_traces(
                    hovertemplate="$%{x} pro Nacht<br>%{y:,} Inserate<extra></extra>",
                    marker_line=dict(color="white", width=1),
                )
                fig.add_vrect(x0=pred_low, x1=pred_high, fillcolor=ACCENT,
                              opacity=0.12, line_width=0)
                fig.add_vline(
                    x=pred, line_color="#0d366b", line_width=2,
                    annotation_text=f"Prognose {usd(pred)}",
                    annotation_position="top right",
                )
                fig.update_layout(
                    **PLOT_LAYOUT, height=320, bargap=0.05,
                    xaxis=dict(title="Preis pro Nacht", tickprefix="$", tickformat=",.0f"),
                    yaxis=dict(title="Anzahl Inserate"),
                )
                st.plotly_chart(fig, alt="Prognose im Vergleich zu echten Preisen ähnlicher Inserate")

    with st.expander("Wie zuverlässig ist die Prognose?", icon=":material/info:"):
        st.markdown(
            f"""
- Das Modell (ein sogenannter *Random Forest*) hat aus **{num(price_model.n_train)}
  Inseraten** gelernt und wurde an **{num(price_model.n_test)} weiteren Inseraten**
  geprüft, die es vorher nicht kannte.
- Bei diesen Prüf-Inseraten lag die Schätzung typischerweise etwa
  **{pct(price_model.median_pct_error, 0)}** neben dem echten Preis.
- Berücksichtigt werden nur **Stadt, Schlafzimmer und Inseratsalter**. Ausstattung,
  genaue Lage, Fotos und Bewertungen fehlen in den Daten – die Prognose ist daher
  ein Richtwert, kein Gutachten.
- Sehr teure Ausreißer (oberstes 1 %) wurden beim Lernen nicht berücksichtigt.
"""
        )


with tab_predict:
    prediction_section()

# ---------------------------------------------------------------- Renditecheck
@st.fragment
def investment_section() -> None:
    """Eingaben hier rechnen nur diesen Abschnitt neu, nicht das ganze Dashboard."""
    st.subheader("Lohnt sich dieses Objekt als Ferienvermietung?")
    st.caption(
        "Schneller Renditecheck für Kaufinteressenten: erzielbare Nachtpreise "
        "vergleichbarer Airbnb-Inserate × Auslastung, abzüglich Kosten. "
        "Alle Annahmen lassen sich anpassen."
    )

    with st.container(border=True):
        st.markdown("**Objekt**")
        c1, c2, c3 = st.columns(3)
        i_city = c1.selectbox("Stadt", options=all_cities, key="i_city")
        i_bed = c2.select_slider(
            "Schlafzimmer", options=BEDROOM_INPUT_OPTIONS, value=3,
            format_func=bedroom_input_label, key="i_bed",
        )
        i_price = c3.number_input(
            "Kaufpreis (USD)", min_value=10_000, max_value=50_000_000,
            value=350_000, step=10_000, key="i_price",
        )
        st.markdown("**Annahmen**")
        c4, c5, c6 = st.columns(3)
        i_occ = c4.slider(
            "Erwartete Auslastung", 10, 95, 55, format="%d %%", key="i_occ",
            help="Anteil der Nächte im Jahr, die gebucht sind. Die Inseratsdaten "
                 "enthalten keine Buchungen – bitte an den lokalen Markt anpassen.",
        )
        i_var = c5.slider(
            "Laufende Kosten (Anteil am Umsatz)", 0, 70, 35, format="%d %%", key="i_var",
            help="Reinigung, Plattformgebühren, Verwaltung, Strom, Internet, "
                 "Verbrauchsmaterial.",
        )
        i_fix = c6.select_slider(
            "Fixkosten pro Jahr (Anteil am Kaufpreis)",
            options=[round(x * 0.1, 1) for x in range(61)], value=2.5,
            format_func=lambda x: f"{num(x, 1)} %", key="i_fix",
            help="Grundsteuer, Versicherung und Instandhaltung.",
        )

    comps = filtered[
        (filtered["city"] == i_city)
        & (filtered["bedrooms"].clip(upper=MAX_BEDROOMS) == i_bed)
    ]
    if len(comps) >= 5:
        nightly = float(comps["price"].median())
        source = f"Median aus {num(len(comps))} vergleichbaren Inseraten in {i_city}"
    else:
        model_city = i_city if i_city in price_model.cities else OTHER_CITY
        nightly = price_model.predict(model_city, i_bed, 12)[0]
        source = (
            f"Schätzung des Preismodells – in den gefilterten Daten gibt es nur "
            f"{len(comps)} vergleichbare Inserate"
        )

    occ, var_share, fix_share = i_occ / 100, i_var / 100, i_fix / 100
    revenue = nightly * 365 * occ
    var_costs = revenue * var_share
    fix_costs = i_price * fix_share
    net = revenue - var_costs - fix_costs
    net_yield = net / i_price
    margin_per_night = nightly * (1 - var_share)
    breakeven = fix_costs / (margin_per_night * 365) if margin_per_night > 0 else float("inf")
    breakeven_text = f"{pct(breakeven, 0)}" if breakeven <= 1 else "nicht erreichbar"
    payback_text = f"{num(i_price / net)} Jahre" if net > 0 else "–"
    net_yield = round(net_yield, 3) or 0.0  # vermeidet die Anzeige „-0.0%“

    st.caption(f"Nachtpreis: {source}.")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Nachtpreis", usd(nightly), border=True)
    k2.metric("Umsatz pro Jahr", usd(revenue), border=True,
              help=f"{usd(nightly)} × 365 Nächte × {i_occ} % Auslastung")
    k3.metric("Nettoertrag pro Jahr", usd(net), border=True,
              help="Umsatz minus laufende Kosten und Fixkosten, vor Finanzierung und "
                   "Einkommensteuer")
    k4.metric("Nettorendite", f"{pct(net_yield)}", border=True,
              help="Nettoertrag pro Jahr geteilt durch den Kaufpreis")
    k5.metric("Break-even-Auslastung", breakeven_text, border=True,
              help="Ab dieser Auslastung decken die Einnahmen alle Kosten.")

    if net <= 0:
        st.error(
            f"Bei {i_occ} % Auslastung deckt die Vermietung die Kosten nicht. "
            f"Kostendeckend wird es erst ab {breakeven_text} Auslastung.",
            title="Rechnet sich nicht", icon=":material/trending_down:",
        )
    elif net_yield >= 0.06:
        st.success(
            f"{pct(net_yield)} Nettorendite – der Kaufpreis hat sich nach etwa "
            f"{payback_text}n wieder eingespielt.",
            title="Starkes Objekt", icon=":material/trending_up:",
        )
    elif net_yield >= 0.03:
        st.warning(
            f"{pct(net_yield)} Nettorendite – solide, aber mit wenig Puffer. "
            "Ein niedrigerer Kaufpreis oder eine höhere Auslastung würde helfen.",
            title="Durchschnittliches Objekt", icon=":material/trending_flat:",
        )
    else:
        st.warning(
            f"Nur {pct(net_yield)} Nettorendite – als reine Ferienvermietung schwach.",
            title="Schwaches Objekt", icon=":material/trending_down:",
        )
    st.caption("Bewertungsschema: ab 6 % stark, 3–6 % durchschnittlich, darunter schwach.")

    col_chart, col_comps = st.columns([3, 2])
    with col_chart, st.container(border=True):
        st.markdown("**Nettoertrag je nach Auslastung**")
        st.caption("Wie robust ist die Rechnung, wenn weniger Nächte gebucht werden?")
        occ_range = pd.DataFrame({"occ": range(10, 96)})
        occ_range["net"] = (
            nightly * 365 * occ_range["occ"] / 100 * (1 - var_share) - fix_costs
        )
        fig = px.line(occ_range, x="occ", y="net", color_discrete_sequence=[ACCENT])
        fig.update_traces(
            line_width=2,
            hovertemplate="%{x} % Auslastung<br>Nettoertrag: $%{y:,.0f}<extra></extra>",
        )
        fig.add_hline(y=0, line_color="#8a8984", line_width=1, line_dash="dot")
        fig.add_scatter(
            x=[i_occ], y=[net], mode="markers", showlegend=False,
            marker=dict(size=11, color="#0d366b", line=dict(color="white", width=2)),
            hovertemplate="Ihre Annahme: %{x} %<br>Nettoertrag: $%{y:,.0f}<extra></extra>",
        )
        fig.update_layout(
            **PLOT_LAYOUT, height=330,
            xaxis=dict(title="Auslastung", ticksuffix=" %"),
            yaxis=dict(title="Nettoertrag pro Jahr", tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig, alt="Nettoertrag pro Jahr in Abhängigkeit von der Auslastung")

    with col_comps, st.container(border=True):
        st.markdown("**Vergleichsobjekte**")
        if comps.empty:
            st.info("Keine vergleichbaren Inserate in den gefilterten Daten.",
                    icon=":material/search_off:")
        else:
            st.caption(f"{num(len(comps))} Inserate in {i_city} mit gleicher Zimmerzahl")
            comp_table = pd.DataFrame({
                "Titel": comps["title"],
                "Preis pro Nacht": comps["price"].round(0).astype(int),
                "Link": comps["url"],
            }).sort_values("Preis pro Nacht")
            st.dataframe(
                comp_table, hide_index=True, height=260,
                column_config={
                    "Preis pro Nacht": st.column_config.NumberColumn(format="$%d"),
                    "Link": st.column_config.LinkColumn(display_text="Ansehen"),
                },
                alt=f"Vergleichbare Inserate in {i_city}",
            )

    report = f"""Renditecheck Ferienvermietung – {i_city}, {bedroom_input_label(i_bed)} Schlafzimmer
Erstellt am {date.today():%d.%m.%Y} · Datenbasis: Airbnb-Inserate in Texas (Stand {DATA_AS_OF})

OBJEKT
  Kaufpreis:                 {usd(i_price)}
  Erwarteter Nachtpreis:     {usd(nightly)} ({source})

ANNAHMEN
  Auslastung:                {i_occ} %
  Laufende Kosten:           {i_var} % vom Umsatz
  Fixkosten:                 {num(i_fix, 1)} % vom Kaufpreis pro Jahr

ERGEBNIS PRO JAHR
  Umsatz:                    {usd(revenue)}
  Laufende Kosten:           {usd(var_costs)}
  Fixkosten:                 {usd(fix_costs)}
  Nettoertrag:               {usd(net)}
  Nettorendite:              {pct(net_yield)}
  Amortisation:              {payback_text}
  Break-even-Auslastung:     {breakeven_text}

Hinweis: Überschlagsrechnung ohne Finanzierung, Kaufnebenkosten und Steuern auf
Erträge. Die Auslastung ist eine Annahme. Keine Anlageberatung.
"""
    slug = f"{i_city}_{bedroom_input_label(i_bed)}".replace(" ", "_").replace("+", "plus")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Kurzbericht für Kunden herunterladen", data=report.encode("utf-8-sig"),
        file_name=f"renditecheck_{slug}.txt", mime="text/plain",
        icon=":material/description:", width="stretch",
    )
    comps_export = pd.DataFrame({
        "Titel": comps["title"], "Stadt": comps["city"],
        "Preis pro Nacht (USD)": comps["price"].round(0).astype(int),
        "Schlafzimmer": comps["bedroom_label"],
        "Inseriert seit": comps["listed"].dt.date, "Link": comps["url"],
    })
    d2.download_button(
        "Vergleichsobjekte als CSV", disabled=comps.empty,
        data=lambda: comps_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"vergleichsobjekte_{slug}.csv", mime="text/csv",
        icon=":material/download:", width="stretch",
    )


with tab_invest:
    investment_section()

# ---------------------------------------------------------------- Daten
with tab_data:
    search = st.text_input(
        "Suche",
        placeholder="Suchbegriff in Titel, Stadt oder Beschreibung, z. B. „Pool“",
        icon=":material/search:",
    )
    table = filtered
    if search.strip():
        term = search.strip()
        hit = (
            table["title"].str.contains(term, case=False, regex=False)
            | table["city"].str.contains(term, case=False, regex=False)
            | table["description"].str.contains(term, case=False, regex=False)
        )
        table = table[hit]

    export = pd.DataFrame(
        {
            "Titel": table["title"],
            "Stadt": table["city"],
            "Preis pro Nacht (USD)": table["price"].round(0).astype(int),
            "Schlafzimmer": table["bedroom_label"],
            "Inseriert seit": table["listed"].dt.date,
            "Breitengrad": table["latitude"],
            "Längengrad": table["longitude"],
            "Beschreibung": table["description"],
            "Link": table["url"],
        }
    )

    header_left, header_right = st.columns([3, 1], vertical_alignment="bottom")
    header_left.caption(
        f"{num(len(export))} Inserate · Klicken Sie auf eine Spaltenüberschrift, "
        "um zu sortieren."
    )
    header_right.download_button(
        "Als CSV herunterladen",
        # Erst beim Klick erzeugen – hält die Seite bei jeder Filteränderung schnell
        data=lambda: export.to_csv(index=False).encode("utf-8-sig"),
        file_name="texas_airbnb_inserate.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
        disabled=export.empty,
    )

    if export.empty:
        st.info("Keine Inserate passen zu Ihrem Suchbegriff.",
                icon=":material/search_off:")
    else:
        st.dataframe(
            export,
            hide_index=True,
            height=560,
            column_config={
                "Titel": st.column_config.TextColumn(width="large", pinned=True),
                "Preis pro Nacht (USD)": st.column_config.NumberColumn(
                    "Preis pro Nacht", format="$%d"
                ),
                "Inseriert seit": st.column_config.DateColumn(format="MMM YYYY"),
                "Breitengrad": None,
                "Längengrad": None,
                "Beschreibung": st.column_config.TextColumn(width="large"),
                "Link": st.column_config.LinkColumn(display_text="Ansehen"),
            },
            alt="Tabelle der gefilterten Airbnb-Inserate",
        )

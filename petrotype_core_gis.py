# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid")

# Загрузка данных
@st.cache_data
def load_data():
    path = r'\\BABAKOV\portal\Северо-Останинское\Вася\Расчёт комплексов литологов\litho_interact.xlsx'
    return pd.read_excel(path, index_col=None)

st.title("Интерактивная визуализация керновых данных")

# Загрузка
core_disc_prop = load_data()

# Проверка нужных колонок
required_columns = ['Литология по описанию керна', 'KP', 'DENS_VL', 'KPR', 'Петротип', 'Литология по ГИС']
gis_features = ['GK_NORM', 'NK', 'DTP_NORM', 'log_BK_NORM']
if not all(col in core_disc_prop.columns for col in required_columns + gis_features):
    st.error(f"Файл должен содержать столбцы: {', '.join(required_columns + gis_features)}")
    st.stop()

# Выбор литологии
unique_lithologies = sorted(core_disc_prop['Литология по описанию керна'].dropna().unique())
selected_lithology = st.selectbox("Выберите литологический комплекс:", unique_lithologies)

# Фильтрация по литологии
df = core_disc_prop[core_disc_prop['Литология по описанию керна'] == selected_lithology].copy()
df.reset_index(drop=True, inplace=True)

# Выбор петротипа
available_petrotips = sorted(df['Петротип'].dropna().unique())
selected_petrotips = st.multiselect(
    "Выберите петротипы:",
    options=available_petrotips,
    default=available_petrotips,
)

# Фильтрация по выбранным петротипам
df = df[df['Петротип'].isin(selected_petrotips)].copy()
df.reset_index(drop=True, inplace=True)
df['Index'] = df.index.astype(str)  # Подписи

# Проверка на пустую выборку
if df.empty:
    st.warning("Нет данных для отображения: выберите другие петротипы.")
    st.stop()

# Настройка размера шрифта
font_size = st.slider("Размер шрифта на графиках:", min_value=8, max_value=36, value=16)

# Кросплоты (4 штуки)
fig, axs = plt.subplots(2, 2, figsize=(18, 14))

# ---------- Верх: DENS_VL ----------
# Слева
sns.scatterplot(
    data=df,
    x='KP',
    y='DENS_VL',
    hue='Петротип',
    s=300,
    alpha=0.6,
    palette='tab10',
    ax=axs[0, 0]
)
for _, row in df.iterrows():
    axs[0, 0].text(row['KP'], row['DENS_VL'], row['Index'], fontsize=font_size)
axs[0, 0].set_title("Пористость vs Плотность (Петротип)", fontsize=font_size)
axs[0, 0].set_xlabel('Пористость, д.ед.', fontsize=font_size)
axs[0, 0].set_ylabel('Плотность (вода), г/см³', fontsize=font_size)
axs[0, 0].tick_params(labelsize=font_size)
axs[0, 0].legend(fontsize=font_size * 0.8, title='Петротип')

# Справа
sns.scatterplot(
    data=df,
    x='KP',
    y='DENS_VL',
    hue='Кластеры ГИС',
    s=300,
    alpha=0.6,
    palette='Set2',
    ax=axs[0, 1]
)
for _, row in df.iterrows():
    axs[0, 1].text(row['KP'], row['DENS_VL'], row['Index'], fontsize=font_size)
axs[0, 1].set_title("Пористость vs Плотность (ГИС)", fontsize=font_size)
axs[0, 1].set_xlabel('Пористость, д.ед.', fontsize=font_size)
axs[0, 1].set_ylabel('Плотность (вода), г/см³', fontsize=font_size)
axs[0, 1].tick_params(labelsize=font_size)
axs[0, 1].legend(fontsize=font_size * 0.8, title='ГИС')

# ---------- Низ: KPR ----------
# Слева
sns.scatterplot(
    data=df,
    x='KP',
    y='KPR',
    hue='Петротип',
    s=300,
    alpha=0.6,
    palette='tab10',
    ax=axs[1, 0]
)
for _, row in df.iterrows():
    axs[1, 0].text(row['KP'], row['KPR'], row['Index'], fontsize=font_size)
axs[1, 0].set_title("Пористость vs Проницаемость (Петротип)", fontsize=font_size)
axs[1, 0].set_xlabel('Пористость, д.ед.', fontsize=font_size)
axs[1, 0].set_ylabel('Проницаемость, мкм²', fontsize=font_size)
axs[1, 0].set_yscale('log')  # Логарифмическая ось Y
axs[1, 0].tick_params(labelsize=font_size)
axs[1, 0].legend(fontsize=font_size * 0.8, title='Петротип')

# Справа
sns.scatterplot(
    data=df,
    x='KP',
    y='KPR',
    hue='Кластеры ГИС',
    s=300,
    alpha=0.6,
    palette='Set2',
    ax=axs[1, 1]
)
for _, row in df.iterrows():
    axs[1, 1].text(row['KP'], row['KPR'], row['Index'], fontsize=font_size)
axs[1, 1].set_title("Пористость vs Проницаемость (ГИС)", fontsize=font_size)
axs[1, 1].set_xlabel('Пористость, д.ед.', fontsize=font_size)
axs[1, 1].set_ylabel('Проницаемость, мкм²', fontsize=font_size)
axs[1, 1].set_yscale('log')  # Логарифмическая ось Y
axs[1, 1].tick_params(labelsize=font_size)
axs[1, 1].legend(fontsize=font_size * 0.8, title='ГИС')

# Отображаем кросплоты
st.pyplot(fig)

# ---------- Дендрограмма ----------
if len(df) >= 2:
    fig2, ax2 = plt.subplots(figsize=(10, max(3, len(df) * 0.3)))
    try:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[gis_features])  # стандартизация
        Z = linkage(X_scaled, method='weighted')
        dendrogram(Z, labels=df['Index'].to_numpy(), orientation='right', ax=ax2)
        ax2.set_title("Дендрограмма по ГИС-признакам (относительные)", fontsize=font_size)
        ax2.set_xlabel("Расстояние", fontsize=font_size)
        ax2.set_ylabel("Объекты (индексы)", fontsize=font_size)
        ax2.tick_params(labelsize=font_size)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Ошибка при построении дендрограммы: {e}")
else:
    st.warning("Недостаточно точек для построения дендрограммы (нужно хотя бы 2).")

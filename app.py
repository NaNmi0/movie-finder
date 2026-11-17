from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("Подборка фильмов.xlsx")
REQUIRED_COLUMNS = ["Название", "Год выпуска", "Страна", "Жанр", "Атмосфера", "Тематика"]
BATCH_SIZE = 15
RATING_OPTIONS = {
    "Не смотрел": None,
    "1 — совсем не понравился": 1,
    "2 — скорее не понравился": 2,
    "3 — нейтрально": 3,
    "4 — понравился": 4,
    "5 — очень понравился": 5,
}

st.set_page_config(page_title="Подбор фильмов", page_icon="🎬", layout="wide")


# ------------------------------------------------------------
# Загрузка и подготовка каталога
# ------------------------------------------------------------
@st.cache_data
def load_movies(file_path: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, engine="openpyxl")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("В Excel не найдены обязательные столбцы: " + ", ".join(missing))

    df = raw[REQUIRED_COLUMNS].copy()

    def clean_text(value):
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()

    for col in ["Название", "Страна", "Жанр", "Атмосфера", "Тематика"]:
        df[col] = df[col].apply(clean_text)

    # Убираем порядковый номер в начале названия: «12. Название»
    df["Название"] = df["Название"].str.replace(r"^\d+\.\s*", "", regex=True)
    df["Год выпуска"] = pd.to_numeric(df["Год выпуска"], errors="coerce")
    df = df.dropna(subset=["Название", "Год выпуска"])
    df["Год выпуска"] = df["Год выпуска"].astype(int)

    # В исходном файле есть повторяющиеся фильмы с частично разными тегами.
    # Объединяем их в одну запись, не теряя перечисленные характеристики.
    def merge_values(series):
        result = []
        for value in series:
            for part in str(value).split(","):
                part = part.strip()
                if part and part.casefold() not in {x.casefold() for x in result}:
                    result.append(part)
        return ", ".join(result)

    grouped_rows = []
    for (title, year), group in df.groupby(["Название", "Год выпуска"], sort=False):
        grouped_rows.append({
            "Название": title,
            "Год выпуска": year,
            "Страна": merge_values(group["Страна"]),
            "Жанр": merge_values(group["Жанр"]),
            "Атмосфера": merge_values(group["Атмосфера"]),
            "Тематика": merge_values(group["Тематика"]),
        })

    result = pd.DataFrame(grouped_rows).reset_index(drop=True)
    result["id"] = result.index.astype(int)
    return result


def build_feature_matrix(movies: pd.DataFrame):
    """Преобразует текстовые характеристики и год выпуска в числовые признаки."""
    text_rows = []
    for _, row in movies.iterrows():
        text_rows.append(
            f"жанры {row['Жанр']} "
            f"страны {row['Страна']} "
            f"атмосфера {row['Атмосфера']} "
            f"тематика {row['Тематика']}"
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    text_matrix = vectorizer.fit_transform(text_rows)

    years = movies["Год выпуска"].to_numpy(dtype=float)
    year_std = years.std() or 1.0
    year_feature = ((years - years.mean()) / year_std * 0.15).reshape(-1, 1)

    # Год имеет небольшой вес: сходство определяется прежде всего содержанием.
    features = hstack([text_matrix, csr_matrix(year_feature)]).tocsr()
    return features


def diverse_selection(features, candidate_ids, count, already_selected=None):
    """
    Жадный farthest-first алгоритм:
    сначала берёт наиболее далёкую пару, затем каждый раз добавляет фильм,
    максимально непохожий на уже выбранные.
    """
    candidates = list(candidate_ids)
    if not candidates or count <= 0:
        return []

    already_selected = list(already_selected or [])
    chosen = []

    if already_selected:
        # Для следующей порции выбираем фильмы, максимально далёкие
        # и от ранее показанных, и от фильмов текущей порции.
        reference = already_selected.copy()
        for _ in range(min(count, len(candidates))):
            distances = cosine_distances(features[candidates], features[reference + chosen])
            min_dist = distances.min(axis=1)
            next_pos = int(np.argmax(min_dist))
            chosen.append(candidates.pop(next_pos))
    else:
        if len(candidates) == 1:
            return candidates
        distances = cosine_distances(features[candidates])
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        first_pair = [candidates[i], candidates[j]]
        chosen.extend(first_pair)
        candidates = [x for x in candidates if x not in first_pair]

        while candidates and len(chosen) < count:
            distances_to_chosen = cosine_distances(features[candidates], features[chosen])
            min_dist = distances_to_chosen.min(axis=1)
            next_pos = int(np.argmax(min_dist))
            chosen.append(candidates.pop(next_pos))

    return chosen[:count]


def rating_weight(rating):
    """Преобразует оценку в вес предпочтения: 1=-1, 3=0, 5=+1."""
    if rating is None:
        return 0.0
    return (float(rating) - 3.0) / 2.0


def explain_match(candidate, reference, columns=("Жанр", "Атмосфера", "Тематика")):
    explanations = []
    for col in columns:
        candidate_values = {x.strip().casefold(): x.strip() for x in str(candidate[col]).split(",") if x.strip()}
        reference_values = {x.strip().casefold(): x.strip() for x in str(reference[col]).split(",") if x.strip()}
        common = [candidate_values[k] for k in candidate_values.keys() & reference_values.keys()]
        if common:
            explanations.append(f"{col.lower()}: {', '.join(sorted(common))}")
    return explanations


def recommend(movies, features, ratings, calibration_ids, top_n=3):
    """Оценивает не показанные в калибровке фильмы по близости к оценённым."""
    rated_ids = [mid for mid, rating in ratings.items() if rating is not None and rating != 3]
    weights = np.array([rating_weight(ratings[mid]) for mid in rated_ids], dtype=float)

    if not rated_ids or np.sum(np.abs(weights)) == 0:
        return []

    similarities = cosine_similarity(features, features[rated_ids])
    candidate_ids = [i for i in range(len(movies)) if i not in set(calibration_ids)]

    scored = []
    for movie_id in candidate_ids:
        sim_row = similarities[movie_id]
        score = float(np.dot(sim_row, weights) / np.sum(np.abs(weights)))
        scored.append((score, movie_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []

    for score, movie_id in scored[:top_n]:
        movie = movies.iloc[movie_id]
        positive_refs = [
            mid for mid in rated_ids if ratings[mid] >= 4
        ]
        nearest_positive = None
        if positive_refs:
            nearest_positive = max(
                positive_refs,
                key=lambda mid: cosine_similarity(
                    features[movie_id], features[mid]
                )[0, 0]
            )

        reasons = []
        if nearest_positive is not None:
            ref_movie = movies.iloc[nearest_positive]
            shared = explain_match(movie, ref_movie)
            reasons.append(f"Ближе всего по набору характеристик к понравившемуся фильму «{ref_movie['Название']}».")
            if shared:
                reasons.append("Общие признаки: " + "; ".join(shared) + ".")

        results.append({
            "movie_id": movie_id,
            "score": score,
            "movie": movie,
            "reasons": reasons,
        })

    return results


# ------------------------------------------------------------
# Интерфейс
# ------------------------------------------------------------
st.title("🎬 Персональные рекомендации фильмов")
st.write(
    "Оцените 15 разнообразных фильмов из вашей подборки. "
    "Алгоритм сравнит их характеристики и предложит три фильма "
    "из оставшегося каталога."
)

if not DATA_FILE.exists():
    st.error(
        "Не найден файл «Подборка фильмов.xlsx». "
        "Положите его в ту же папку, что и app.py."
    )
    st.stop()

try:
    movies = load_movies(str(DATA_FILE))
except Exception as error:
    st.error(f"Не удалось прочитать каталог: {error}")
    st.stop()

if len(movies) < BATCH_SIZE + 3:
    st.error("В каталоге должно быть минимум 18 уникальных фильмов.")
    st.stop()

features = build_feature_matrix(movies)

# Состояние существует только в текущем сеансе браузера.
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "calibration_ids" not in st.session_state:
    initial_ids = diverse_selection(features, list(range(len(movies))), BATCH_SIZE)
    st.session_state.calibration_ids = initial_ids
if "batch_ids" not in st.session_state:
    st.session_state.batch_ids = st.session_state.calibration_ids.copy()
if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False

st.caption(
    f"В каталоге: {len(movies)} уникальных фильмов. "
    f"Для первоначальной настройки показывается по {BATCH_SIZE} максимально разнообразных фильмов."
)

st.subheader("Оцените фильмы")
st.info(
    "Выберите «Не смотрел», если вы не знакомы с фильмом. "
    "Оценки 1–2 считаются отрицательными, 3 — нейтральной, 4–5 — положительными."
)

with st.form("rating_form"):
    for movie_id in st.session_state.batch_ids:
        movie = movies.iloc[movie_id]
        title = movie["Название"]
        st.markdown(f"**{title} ({movie['Год выпуска']})**")
        st.caption(
            f"Жанр: {movie['Жанр']}  \n"
            f"Атмосфера: {movie['Атмосфера']}  \n"
            f"Тематика: {movie['Тематика']}"
        )

        current_rating = st.session_state.ratings.get(movie_id)
        options = list(RATING_OPTIONS.keys())
        default_index = 0
        if current_rating is not None:
            default_index = list(RATING_OPTIONS.values()).index(current_rating)

        selected_label = st.selectbox(
            "Ваша оценка",
            options,
            index=default_index,
            key=f"rating_widget_{movie_id}",
        )
        st.session_state[f"pending_rating_{movie_id}"] = selected_label
        st.divider()

    submitted = st.form_submit_button("Сохранить оценки")

if submitted:
    for movie_id in st.session_state.batch_ids:
        label = st.session_state.get(f"rating_widget_{movie_id}", "Не смотрел")
        st.session_state.ratings[movie_id] = RATING_OPTIONS[label]
    st.session_state.show_recommendations = False
    st.success("Оценки сохранены в текущем сеансе.")

positive_count = sum(
    1 for value in st.session_state.ratings.values()
    if value is not None and value >= 4
)

rated_count = sum(
    1 for value in st.session_state.ratings.values()
    if value is not None
)

st.write(
    f"Сохранено оценок: **{rated_count}**. "
    f"Положительных оценок (4–5): **{positive_count}**."
)

remaining_ids = [
    i for i in range(len(movies))
    if i not in set(st.session_state.calibration_ids)
]

if positive_count == 0 and len(remaining_ids) > 0:
    st.warning(
        "Чтобы определить, что вам нравится, нужна хотя бы одна оценка 4 или 5. "
        "Если среди текущих 15 фильмов нет просмотренного и понравившегося, "
        "добавьте следующую разнообразную порцию."
    )
    if st.button("Показать следующие 15 фильмов"):
        next_ids = diverse_selection(
            features,
            remaining_ids,
            min(BATCH_SIZE, len(remaining_ids)),
            already_selected=st.session_state.calibration_ids,
        )
        st.session_state.calibration_ids.extend(next_ids)
        st.session_state.batch_ids = next_ids
        st.rerun()

if positive_count > 0:
    if st.button("Подобрать три фильма", type="primary"):
        st.session_state.show_recommendations = True

if st.session_state.show_recommendations:
    results = recommend(
        movies,
        features,
        st.session_state.ratings,
        st.session_state.calibration_ids,
        top_n=3,
    )

    st.subheader("🍿 Ваши три рекомендации")
    if not results:
        st.warning("Недостаточно оценок для расчёта. Оцените хотя бы один фильм на 4 или 5.")
    else:
        for rank, result in enumerate(results, start=1):
            movie = result["movie"]
            st.markdown(f"### {rank}. {movie['Название']} ({movie['Год выпуска']})")
            st.write(f"**Жанр:** {movie['Жанр']}")
            st.write(f"**Страна:** {movie['Страна']}")
            st.write(f"**Атмосфера:** {movie['Атмосфера']}")
            st.write(f"**Тематика:** {movie['Тематика']}")
            st.write(f"**Индекс соответствия:** {result['score']:.3f}")
            for reason in result["reasons"]:
                st.write(reason)
            st.divider()

st.caption(
    "Результаты не сохраняются в аккаунте или базе данных. "
    "Оценки существуют только в активном сеансе приложения."
)

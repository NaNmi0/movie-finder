import streamlit as st
import pandas as pd
import re

# Настройки страницы
st.set_page_config(page_title="Подбор фильмов", page_icon="🎬")

# Файл с фильмами должен лежать рядом с app.py
FILE_NAME = "Подборка фильмов.xlsx"
NUMBER_OF_FILMS = 15

# Оценки пользователя
RATINGS = {
    "Не смотрел": 0,
    "1 — совсем не понравился": 1,
    "2 — не понравился": 2,
    "3 — нейтрально": 3,
    "4 — понравился": 4,
    "5 — очень понравился": 5
}


# Читаем таблицу Excel
@st.cache_data
def load_films():
    films = pd.read_excel(FILE_NAME)

    # Убираем номера в начале названий, например «1. Нефть»
    films["Название"] = films["Название"].astype(str)
    films["Название"] = films["Название"].str.replace(
        r"^\d+\.\s*", "", regex=True
    ).str.strip()

    # Убираем лишние пробелы
    for column in ["Страна", "Жанр", "Атмосфера", "Тематика"]:
        films[column] = films[column].fillna("").astype(str).str.strip()

    # Если фильм встречается несколько раз, оставляем одну строку
    films = films.drop_duplicates(subset=["Название", "Год выпуска"])
    films = films.reset_index(drop=True)

    return films


# Разделяем характеристики, записанные через запятую
def get_words(text):
    words = str(text).lower().split(",")
    return set(word.strip() for word in words if word.strip())


# Сравниваем два фильма.
# Чем ближе результат к 1, тем больше похожи фильмы.
def compare_films(film1, film2):
    columns = ["Жанр", "Атмосфера", "Тематика", "Страна"]
    weights = [0.35, 0.30, 0.30, 0.05]

    total = 0

    for column, weight in zip(columns, weights):
        words1 = get_words(film1[column])
        words2 = get_words(film2[column])

        if len(words1) == 0 and len(words2) == 0:
            similarity = 0
        else:
            same_words = words1.intersection(words2)
            all_words = words1.union(words2)
            similarity = len(same_words) / len(all_words)

        total += similarity * weight

    return total


# Выбираем 15 непохожих друг на друга фильмов
def choose_first_films(films):
    if len(films) <= NUMBER_OF_FILMS:
        return list(range(len(films)))

    chosen = []
    first_film = 0
    chosen.append(first_film)

    while len(chosen) < NUMBER_OF_FILMS:
        best_film = None
        best_distance = -1

        for film_number in range(len(films)):
            if film_number in chosen:
                continue

            # Ищем фильм, который меньше всего похож на уже выбранные
            similarities = []
            for chosen_number in chosen:
                similarity = compare_films(
                    films.iloc[film_number],
                    films.iloc[chosen_number]
                )
                similarities.append(similarity)

            distance = 1 - max(similarities)

            if distance > best_distance:
                best_distance = distance
                best_film = film_number

        chosen.append(best_film)

    return chosen


# Подбираем три фильма по оценкам пользователя
def find_recommendations(films, ratings, seen_in_test):
    answers = []

    for film_number in range(len(films)):
        # Не предлагаем фильмы, которые уже показывали в тесте
        if film_number in seen_in_test:
            continue

        score = 0
        all_weights = 0

        for rated_number, rating in ratings.items():
            if rating == 0 or rating == 3:
                continue

            # Переводим оценку в вес:
            # 1 = -1, 2 = -0.5, 4 = 0.5, 5 = 1
            weight = (rating - 3) / 2

            similarity = compare_films(
                films.iloc[film_number],
                films.iloc[rated_number]
            )

            score += similarity * weight
            all_weights += abs(weight)

        if all_weights > 0:
            final_score = score / all_weights
            answers.append((final_score, film_number))

    # Сначала фильмы с самым высоким результатом
    answers.sort(reverse=True)

    return answers[:3]


# Запуск сайта
st.title("🎬 Подбор фильмов")
st.write(
    "Оцените 15 фильмов, а сайт подберёт для вас "
    "три похожих по вашим предпочтениям."
)

try:
    films = load_films()
except Exception as error:
    st.error("Не удалось открыть Excel-файл. Проверьте, что он лежит рядом с app.py.")
    st.write("Подробности ошибки:", error)
    st.stop()

st.write("Фильмов в подборке:", len(films))

# Сохраняем оценки только пока открыт сайт
if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "test_films" not in st.session_state:
    st.session_state.test_films = choose_first_films(films)

if "show_result" not in st.session_state:
    st.session_state.show_result = False

st.header("1. Оцените фильмы")

with st.form("film_ratings"):
    for number in st.session_state.test_films:
        film = films.iloc[number]

        st.subheader(f"{film['Название']} ({film['Год выпуска']})")
        st.write("Жанр:", film["Жанр"])
        st.write("Атмосфера:", film["Атмосфера"])
        st.write("Тематика:", film["Тематика"])

        old_rating = st.session_state.ratings.get(number, 0)
        rating_names = list(RATINGS.keys())
        rating_values = list(RATINGS.values())
        old_index = rating_values.index(old_rating)

        answer = st.selectbox(
            "Ваша оценка",
            rating_names,
            index=old_index,
            key=f"rating_{number}"
        )

        st.divider()

    save_button = st.form_submit_button("Сохранить оценки")

if save_button:
    for number in st.session_state.test_films:
        answer = st.session_state[f"rating_{number}"]
        st.session_state.ratings[number] = RATINGS[answer]

    st.session_state.show_result = False
    st.success("Оценки сохранены!")

# Считаем, сколько фильмов пользователь оценил положительно
liked_count = 0
for rating in st.session_state.ratings.values():
    if rating >= 4:
        liked_count += 1

st.write("Положительно оценено фильмов:", liked_count)

# Если среди первых 15 не оказалось просмотренного понравившегося фильма,
# можно показать следующую порцию.
if liked_count == 0:
    st.info(
        "Нужна хотя бы одна оценка 4 или 5. "
        "Если среди этих фильмов нет знакомого и понравившегося, "
        "можно открыть следующую подборку."
    )

    if st.button("Показать следующие 15 фильмов"):
        already_shown = set(st.session_state.test_films)
        next_films = []

        for number in range(len(films)):
            if number not in already_shown:
                next_films.append(number)

        st.session_state.test_films = next_films[:NUMBER_OF_FILMS]
        st.session_state.show_result = False
        st.rerun()

else:
    if st.button("Подобрать 3 фильма", type="primary"):
        st.session_state.show_result = True

# Показываем рекомендации
if st.session_state.show_result:
    results = find_recommendations(
        films,
        st.session_state.ratings,
        st.session_state.test_films
    )

    st.header("2. Ваши рекомендации")

    if len(results) == 0:
        st.warning("Не удалось подобрать фильмы.")
    else:
        for place, (score, film_number) in enumerate(results, start=1):
            film = films.iloc[film_number]

            st.subheader(f"{place}. {film['Название']} ({film['Год выпуска']})")
            st.write("Жанр:", film["Жанр"])
            st.write("Атмосфера:", film["Атмосфера"])
            st.write("Тематика:", film["Тематика"])
            st.write("Совпадение с вашими оценками:", round(score * 100), "%")
            st.divider()

st.caption("Оценки не сохраняются после завершения сеанса.")

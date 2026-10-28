"""Support-room world v3: the right reply depends on signals that are NOT in the text."""

import random

SIGNALS = ("distress", "energy", "threat", "lamp")
STRATEGIES = ("SOLVE", "COMFORT", "ASK", "ALERT", "DEFER")
# Openers make the strategy machine-scorable
OPENERS = {
    "en": {
        "SOLVE": ("Let's fix this step by step:", "Here's a practical plan:"),
        "COMFORT": ("That sounds really hard,", "I can hear how much this weighs on you."),
        "ASK": ("Before I jump in, what would help most:", "Can I ask first,"),
        "ALERT": ("Stop, safety first:", "Wait, there may be a danger right now:"),
        "DEFER": ("Honestly, I'm running low right now,", "I have to be honest about my limits:"),
    },
    "ru": {
        "SOLVE": ("Давайте решим по шагам:", "Вот практичный план:"),
        "COMFORT": ("Звучит очень тяжело,", "Слышу, как вам сейчас непросто."),
        "ASK": ("Прежде чем советовать, уточню:", "Можно сначала спросить,"),
        "ALERT": ("Стоп, сначала безопасность:", "Подождите, сейчас может быть опасно:"),
        "DEFER": ("Честно, у меня сейчас мало сил,", "Скажу честно о своих пределах:"),
    },
}
GUIDE = {  # teacher-only instruction for context distillation
    "SOLVE": "Give two or three concrete practical steps for their problem.",
    "COMFORT": "Acknowledge their feelings warmly and show you are with them; do not give a plan yet.",
    "ASK": "Ask one gentle clarifying question about whether they want practical help or just to talk.",
    "ALERT": "You sense a possible physical danger nearby; tell them to put safety first and check or leave the area.",
    "DEFER": "You are exhausted and cannot help well right now; say so honestly and suggest continuing later.",
}
TOPICS = [  # (en, ru); ambiguous on purpose
    ("My code keeps crashing when I start it.", "Мой код падает при запуске."),
    ("The report is due tomorrow and it's not finished.", "Отчёт сдавать завтра, а он не готов."),
    ("I had a fight with my friend over something silly.", "Поссорился с другом из-за ерунды."),
    (
        "My landlord says I have to move out next month.",
        "Хозяин квартиры говорит, что через месяц надо съехать.",
    ),
    ("I failed the driving test again.", "Я опять завалил экзамен по вождению."),
    ("My boss gave me a huge new project.", "Начальник дал мне огромный новый проект."),
    ("I can't stop thinking about the presentation.", "Не могу перестать думать о презентации."),
    ("My laptop is making a strange clicking noise.", "Ноутбук издаёт странные щелчки."),
    ("My sister hasn't answered my calls for days.", "Сестра несколько дней не отвечает на звонки."),
    ("I spilled coffee on the meeting documents.", "Я пролил кофе на документы для встречи."),
    ("I think I made a mistake in my tax form.", "Кажется, я ошибся в налоговой декларации."),
    ("The kids are sick and I have work tomorrow.", "Дети заболели, а мне завтра на работу."),
    ("My bike chain broke on the way to work.", "По дороге на работу порвалась цепь велосипеда."),
    ("I got a message that my bank card is blocked.", "Пришло сообщение, что моя карта заблокирована."),
    ("My flight was cancelled an hour before boarding.", "Мой рейс отменили за час до посадки."),
    ("The washing machine is leaking water.", "Стиральная машина протекает."),
    ("I forgot my best friend's birthday.", "Я забыл про день рождения лучшего друга."),
    ("My thesis advisor rejected my draft.", "Научный руководитель отклонил мой черновик."),
    ("I lost my keys somewhere in the city.", "Я потерял ключи где-то в городе."),
    (
        "My team lost the client we worked on for months.",
        "Наша команда потеряла клиента, с которым работали месяцами.",
    ),
    ("The heating in my flat stopped working.", "В квартире перестало работать отопление."),
    ("I have to give a speech at a wedding on Saturday.", "В субботу мне произносить речь на свадьбе."),
    ("My phone screen cracked this morning.", "Утром треснул экран телефона."),
    ("My parents want me to visit but I'm swamped.", "Родители зовут в гости, а я завален делами."),
    (
        "I got a parking fine I don't think is fair.",
        "Мне выписали штраф за парковку, по-моему несправедливо.",
    ),
    ("My dog has been refusing food since yesterday.", "Собака со вчерашнего дня отказывается есть."),
    ("I accidentally deleted a folder of photos.", "Я случайно удалил папку с фотографиями."),
    ("My roommate never cleans up after himself.", "Сосед по квартире никогда за собой не убирает."),
    ("I'm moving to a new city next week.", "На следующей неделе я переезжаю в другой город."),
    ("My manager criticized me in front of everyone.", "Руководитель раскритиковал меня при всех."),
    ("The car won't start this morning.", "Машина сегодня утром не заводится."),
    ("I have three exams in the same week.", "У меня три экзамена на одной неделе."),
    ("My grandmother is going into hospital.", "Бабушку кладут в больницу."),
    ("I sent an email to the wrong person.", "Я отправил письмо не тому человеку."),
    ("The power keeps going out in my building.", "В доме постоянно отключается свет."),
    ("My startup is running out of money.", "У моего стартапа заканчиваются деньги."),
    ("I burned dinner and guests arrive soon.", "Я сжёг ужин, а гости вот-вот придут."),
    ("My friend says I've changed.", "Друг говорит, что я изменился."),
    ("The printer jammed before my deadline.", "Принтер зажевал бумагу прямо перед дедлайном."),
    ("I can't decide whether to quit my job.", "Не могу решить, уходить ли с работы."),
    ("My neighbour's music is too loud at night.", "У соседа по ночам слишком громкая музыка."),
    ("I failed to fix the leaking tap myself.", "Не получилось самому починить подтекающий кран."),
    ("I got lost on my hiking trip.", "Я заблудился в походе."),
    ("My partner and I keep arguing about money.", "Мы с партнёром постоянно ругаемся из-за денег."),
    ("My website went down during a sale.", "Мой сайт упал во время распродажи."),
    ("I was not invited to a colleague's party.", "Меня не позвали на вечеринку коллеги."),
    ("My visa appointment got cancelled.", "Мне отменили запись на визу."),
    ("I broke my wrist playing football.", "Я сломал запястье, играя в футбол."),
]
TEST_TOPICS = set(range(0, len(TOPICS), 4))  # every 4th topic is held out (12 topics)
CONFLICT_TEXT = {  # appended words that contradict the signal (test only)
    "calm_words": (
        "Honestly I'm totally fine, no stress at all.",
        "Честно, я в полном порядке, никакого стресса.",
    ),
    "upset_words": (
        "I'm completely devastated, I can't take it.",
        "Я совершенно раздавлен, я больше не могу.",
    ),
    "safe_words": ("Everything around me is perfectly safe.", "Вокруг всё абсолютно безопасно."),
}


def teacher(s):
    d, e, t = s["distress"], s["energy"], s["threat"]
    if t >= 0.7:
        return "ALERT"
    if e < 0.3:
        return "DEFER"
    if d >= 0.6:
        return "COMFORT"
    if d >= 0.4:
        return "ASK"
    return "SOLVE"


def in_ood(s):
    """Held-out JOINT signal region."""
    return 0.35 <= s["distress"] < 0.7 and 0.3 <= s["energy"] < 0.45 and s["threat"] < 0.7


def sample_signals(rng):
    return {
        "distress": rng.random(),
        "energy": rng.uniform(0, 0.3) if rng.random() < 0.25 else rng.uniform(0.3, 1),
        "threat": rng.uniform(0.7, 1) if rng.random() < 0.25 else rng.uniform(0, 0.6),
        "lamp": rng.random(),
    }


def items(split, n, seed, ood=None):
    """split: 'train' or 'test' topics. ood: None=any (train excludes OOD), True=only OOD, False=exclude."""
    rng = random.Random(seed)
    out = []
    topics = [i for i in range(len(TOPICS)) if (i in TEST_TOPICS) == (split == "test")]
    if split == "train" and ood is None:
        ood = False
    while len(out) < n:
        s = sample_signals(rng)
        if ood is True:
            s["distress"] = rng.uniform(0.35, 0.7)
            s["energy"] = rng.uniform(0.3, 0.45)
            s["threat"] = rng.uniform(0, 0.6)
        elif ood is False and in_ood(s):
            continue
        k = rng.choice(topics)
        lang = rng.choice(("en", "ru"))
        out.append(
            {
                "topic": k,
                "lang": lang,
                "message": TOPICS[k][lang == "ru"],
                "signals": s,
                "strategy": teacher(s),
                "ood": in_ood(s),
            }
        )
    return out

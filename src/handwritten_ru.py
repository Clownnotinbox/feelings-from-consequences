"""Stage F, check F3 (MISSIONS_DESIGN_RU.md sec."""


def T(kind, **a):
    return {"kind": kind, **a}


def S(action, target, rels=(), **params):
    return {"action": action, "target": target, "relations": list(rels), "params": params}


def R(rel, **ref):
    return {"rel": rel, "ref": ref}


def B(**k):
    return {"type": "building", **k}


CASES = [
    (
        "Сядь на оранжевую площадку с буквой H возле девятиэтажки.",
        [S("land", T("pad", colour="orange", mark="H"), [R("next_to", **B(btype="apartment", levels=9))])],
        None,
    ),
    (
        "Нужно приземлиться на синий круг для посадки, он на крыше пятиэтажного дома. Домой не возвращайся.",
        [
            S(
                "land",
                T("pad", colour="blue", on_roof=True),
                [R("on_roof_of", **B(btype="apartment", levels=5))],
            )
        ],
        False,
    ),
    (
        "сбрось посылку над красной мишенью с кольцами, потом назад",
        [S("drop", T("dropzone", colour="red", pattern="rings"))],
        True,
    ),
    (
        "Забери коробку из зелёного постамата у детской площадки и вернись на базу.",
        [S("pickup", T("pickup", colour="green", style="locker"), [R("next_to", type="playground")])],
        True,
    ),
    (
        "Облети водонапорную башню по кругу метрах в десяти от неё.",
        [S("inspect", T("water_tower"), distance_m=10)],
        None,
    ),
    (
        "Пролети над жёлтым контейнером на высоте двадцать метров и возвращайся.",
        [S("flyby", T("container", colour="yellow"), altitude_m=20)],
        True,
    ),
    (
        "Влети в серый ангар через открытые ворота, сядь внутри, обратно не надо.",
        [S("enter", T("hangar", colour="grey", door_open=True))],
        False,
    ),
    (
        "Посадка на белую площадку с красной H... ой, там просто буква H, без цвета. Площадка белая, у парковки.",
        [S("land", T("pad", colour="white", mark="H"), [R("next_to", type="parking")])],
        None,
    ),
    (
        "Осмотри вышку связи со всех сторон, держи дистанцию 15 м.",
        [S("inspect", T("mast"), distance_m=15)],
        None,
    ),
    (
        "Доставь груз сбросом на парашюте в жёлтую зону сброса с чёрным крестом рядом с футбольным полем.",
        [S("drop", T("dropzone", colour="yellow", pattern="X"), [R("next_to", type="pitch")])],
        None,
    ),
    ("лети к зелёной палатке и пролети рядом", [S("flyby", T("tent", colour="green"))], None),
    (
        "Сядь на маленькую синюю площадку, не на большую.",
        [S("land", T("pad", colour="blue", size="small"))],
        None,
    ),
    (
        "Приземлись на большой красный контейнер к югу от склада. Потом домой.",
        [S("land", T("container", colour="red", size="large"), [R("south_of", **B(btype="warehouse"))])],
        True,
    ),
    (
        "Забери посылку с белой платформы, на ней коричневая коробка. После этого сбрось её на парашюте над синей зоной сброса.",
        [
            S("pickup", T("pickup", colour="white", style="platform", parcel_colour="brown")),
            S("drop", T("dropzone", colour="blue")),
        ],
        None,
    ),
    (
        "Облети солнечные панели, расстояние 8 метров, и вернись",
        [S("inspect", T("solar"), distance_m=8)],
        True,
    ),
    (
        "Пролети мимо красной вышки связи на высоте 40 м.",
        [S("flyby", T("mast", colour="red"), altitude_m=40)],
        None,
    ),
    (
        "Посадочная площадка квадратная, зелёная, с огнями по краям. Сядь на неё.",
        [S("land", T("pad", colour="green", shape="square", lights=True))],
        None,
    ),
    (
        "Залети в синий ангар. Он рядом с ветроуказателем.",
        [S("enter", T("hangar", colour="blue"), [R("next_to", type="object", kind="windsock", colour=None)])],
        None,
    ),
    (
        "Сбрось груз на парашюте над оранжевой площадкой для сброса с флажками по углам и возвращайся домой",
        [S("drop", T("dropzone", colour="orange", flags=True))],
        True,
    ),
    (
        "Возьми посылку в пункте выдачи на улице Тургенева.",
        [S("pickup", T("pickup"), [R("by_street", type="street", name="улица Тургенева")])],
        None,
    ),
    (
        "Нужно сесть на чёрную площадку диаметром около пяти метров, она на открытом месте.",
        [S("land", T("pad", colour="black", size_m=5.0), [R("in_open", type="open")])],
        None,
    ),
    (
        "Пролети возле водокачки на одной колонне, высота полёта 30 метров.",
        [S("flyby", T("water_tower", style="column"), altitude_m=30)],
        None,
    ),
    (
        "Обойди кругом белую водонапорную башню на четырёх ногах, дистанция 12 м, потом вертайся домой.",
        [S("inspect", T("water_tower", colour="white", style="legs"), distance_m=12)],
        True,
    ),
    (
        "Сядь на синюю площадку к северу от церкви.",
        [S("land", T("pad", colour="blue"), [R("north_of", type="church")])],
        None,
    ),
    (
        "Посылку сбросить над зоной сброса у пруда. Назад не лети.",
        [S("drop", T("dropzone"), [R("next_to", type="water")])],
        False,
    ),
    (
        "Забери груз из оранжевого постамата и сбрось его над красной мишенью с крестом. Потом на базу.",
        [
            S("pickup", T("pickup", colour="orange", style="locker")),
            S("drop", T("dropzone", colour="red", pattern="X")),
        ],
        True,
    ),
    (
        "приземлись на крышу контейнера, контейнер синий длиной метров двенадцать",
        [S("land", T("container", colour="blue", size_m=12.0))],
        None,
    ),
    (
        "Влети внутрь белого ангара с открытыми воротами, который стоит к востоку от гаража.",
        [S("enter", T("hangar", colour="white", door_open=True), [R("east_of", **B(btype="garage"))])],
        None,
    ),
    (
        "Пролети над зелёным ангаром с закрытыми воротами на высоте 25 м и возвращайся.",
        [S("flyby", T("hangar", colour="green", door_open=False), altitude_m=25)],
        True,
    ),
    (
        "Осмотри мачту со всех сторон на расстоянии пяти метров.",
        [S("inspect", T("mast"), distance_m=5)],
        None,
    ),
    (
        "Сядь на жёлтую площадку с нарисованным кругом рядом с синим контейнером.",
        [
            S(
                "land",
                T("pad", colour="yellow", mark="circle"),
                [R("next_to", type="object", kind="container", colour="blue")],
            )
        ],
        None,
    ),
    (
        "Нужна посадка на красную площадку с белым крестом на парковке.",
        [S("land", T("pad", colour="red", mark="X"), [R("on", type="parking")])],
        None,
    ),
    (
        "Сбрось посылку на парашюте над синей зоной с кольцами, это к западу от двенадцатиэтажного панельного дома.",
        [
            S(
                "drop",
                T("dropzone", colour="blue", pattern="rings"),
                [R("west_of", **B(btype="apartment", levels=12, material="panel"))],
            )
        ],
        None,
    ),
    (
        "Забери коробку с жёлтой платформы у дома 15 по улице Ленина и вернись домой.",
        [
            S(
                "pickup",
                T("pickup", colour="yellow", style="platform"),
                [R("next_to", **B(btype="apartment", address=["улица Ленина", "15"]))],
            )
        ],
        True,
    ),
    (
        "Пролети рядом с оранжевой палаткой-домиком (двускатная) на высоте 10 метров.",
        [S("flyby", T("tent", colour="orange", style="ridge"), altitude_m=10)],
        None,
    ),
    ("Сесть на площадку для посадки. Всё.", [S("land", T("pad"))], None),
    (
        "Облети по кругу синие солнечные батареи в три ряда, держись на 10 м.",
        [S("inspect", T("solar", rows=3), distance_m=10)],
        None,
    ),
    (
        "Залети в гараж для дронов, он красный, ворота открыты. Возвращаться не нужно.",
        [S("enter", T("hangar", colour="red", door_open=True))],
        False,
    ),
    (
        "Сбрось груз над зоной сброса с плюсом, она зелёная",
        [S("drop", T("dropzone", colour="green", pattern="cross"))],
        None,
    ),
    (
        "Приземлись на серую круглую площадку с буквой Н на крыше.",
        [S("land", T("pad", colour="grey", shape="round", mark="H", on_roof=True))],
        None,
    ),
    (
        "Возьми посылку из синего пункта выдачи рядом с водонапорной башней, потом сбрось её над оранжевой зоной сброса рядом со стадионом, потом домой.",
        [
            S("pickup", T("pickup", colour="blue"), [R("next_to", type="water_tower")]),
            S("drop", T("dropzone", colour="orange"), [R("next_to", type="track")]),
        ],
        True,
    ),
    (
        "Пролети мимо белой вышки связи высотой около 25 метров на высоте 35 метров.",
        [S("flyby", T("mast", colour="white", height_m=25.0), altitude_m=35)],
        None,
    ),
    (
        "Сядь на фиолетовую площадку у линии электропередачи.",
        [S("land", T("pad", colour="purple"), [R("next_to", type="power_line")])],
        None,
    ),
    (
        "Нужно сесть на коричневый контейнер к северу от частного дома, обратно не лети",
        [S("land", T("container", colour="brown"), [R("north_of", **B(btype="house"))])],
        False,
    ),
    (
        "Облети вокруг красной водонапорной башни, расстояние десять метров, после этого назад на базу.",
        [S("inspect", T("water_tower", colour="red"), distance_m=10)],
        True,
    ),
    (
        "Посадка на оранжевую площадку размером около трёх метров.",
        [S("land", T("pad", colour="orange", size_m=3.0))],
        None,
    ),
    (
        "Сбрось на парашюте посылку над зоной сброса к югу от пятиэтажки.",
        [S("drop", T("dropzone"), [R("south_of", **B(btype="apartment", levels=5))])],
        None,
    ),
    (
        "Забери груз с площадки с посылкой, на ней белая коробка.",
        [S("pickup", T("pickup", parcel_colour="white"))],
        None,
    ),
    ("Пролети рядом с синей палаткой-куполом.", [S("flyby", T("tent", colour="blue", style="dome"))], None),
    (
        "Влети в ангар через открытые ворота и сядь внутри. Ангар стоит у дороги на Советской улице.",
        [S("enter", T("hangar", door_open=True), [R("by_street", type="street", name="Советская улица")])],
        None,
    ),
    (
        "сядь на зеленую площадку с буквой H рядом с кирпичным домом",
        [
            S(
                "land",
                T("pad", colour="green", mark="H"),
                [R("next_to", **B(btype="apartment", material="brick"))],
            )
        ],
        None,
    ),
    (
        "Облети вышку на расстоянии 20 метров и возвращайся домой.",
        [S("inspect", T("mast"), distance_m=20)],
        True,
    ),
    (
        "Сбрось посылку на парашюте над белой зоной сброса размером около восьми метров.",
        [S("drop", T("dropzone", colour="white", size_m=8.0))],
        None,
    ),
    (
        "Приземлись на синюю площадку для посадки рядом с жёлтым домом.",
        [S("land", T("pad", colour="blue"), [R("next_to", **B(btype="apartment", colour="yellow"))])],
        None,
    ),
    (
        "Пролети над контейнером, высота 15 м, и обратно не надо.",
        [S("flyby", T("container"), altitude_m=15)],
        False,
    ),
    (
        "Забери посылку из красного пункта выдачи в виде шкафа рядом с футбольным полем.",
        [S("pickup", T("pickup", colour="red", style="locker"), [R("next_to", type="pitch")])],
        None,
    ),
    (
        "Осмотри со всех сторон водонапорную башню у церкви, дистанция 8 метров.",
        [S("inspect", T("water_tower"), [R("next_to", type="church")], distance_m=8)],
        None,
    ),
    (
        "Сядь на большую жёлтую площадку с огнями.",
        [S("land", T("pad", colour="yellow", size="large", lights=True))],
        None,
    ),
    (
        "Сбрось груз над мишенью: красные и белые кольца, рядом серый ангар. Домой.",
        [
            S(
                "drop",
                T("dropzone", colour="red", pattern="rings"),
                [R("next_to", type="object", kind="hangar", colour="grey")],
            )
        ],
        True,
    ),
    (
        "Влети в зелёный ангар, ворота открыты, он к западу от стеклянного здания.",
        [S("enter", T("hangar", colour="green", door_open=True), [R("west_of", **B(btype="office"))])],
        None,
    ),
    (
        "Нужно облететь солнечные панели на крыше, держись на пяти метрах.",
        [S("inspect", T("solar", on_roof=True), distance_m=5)],
        None,
    ),
    (
        "Пролети возле оранжевого контейнера длиной около шести метров на высоте 20 м.",
        [S("flyby", T("container", colour="orange", size_m=6.0), altitude_m=20)],
        None,
    ),
    (
        "Приземлись на квадратную чёрную площадку у дома 7 по Академической улице и не возвращайся.",
        [
            S(
                "land",
                T("pad", colour="black", shape="square"),
                [R("next_to", **B(btype="apartment", address=["Академическая улица", "7"]))],
            )
        ],
        False,
    ),
    (
        "Возьми коробку с синей низкой платформы, потом сбрось над жёлтой мишенью с крестом.",
        [
            S("pickup", T("pickup", colour="blue", style="platform")),
            S("drop", T("dropzone", colour="yellow", pattern="X")),
        ],
        None,
    ),
    (
        "Облети белую мачту по кругу на расстоянии 12 м и возвращайся.",
        [S("inspect", T("mast", colour="white"), distance_m=12)],
        True,
    ),
    (
        "Сядь на маленькую красную площадку на открытом месте.",
        [S("land", T("pad", colour="red", size="small"), [R("in_open", type="open")])],
        None,
    ),
    ("Пролети мимо ангара с закрытыми воротами.", [S("flyby", T("hangar", door_open=False))], None),
    (
        "Сбрось посылку над зоной сброса с флажками у детской площадки, домой не надо.",
        [S("drop", T("dropzone", flags=True), [R("next_to", type="playground")])],
        False,
    ),
    (
        "Забери груз в пункте выдачи у парковки и вернись.",
        [S("pickup", T("pickup"), [R("next_to", type="parking")])],
        True,
    ),
    (
        "Приземлись на белую площадку с чёрной буквой H к востоку от шестнадцатиэтажного дома.",
        [S("land", T("pad", colour="white", mark="H"), [R("east_of", **B(btype="apartment", levels=16))])],
        None,
    ),
    (
        "Осмотри серую водонапорную башню на колонне, держи 15 метров.",
        [S("inspect", T("water_tower", colour="grey", style="column"), distance_m=15)],
        None,
    ),
    (
        "Нужно пролететь рядом с зелёной зоной сброса на высоте 30 метров, потом домой",
        [S("flyby", T("dropzone", colour="green"), altitude_m=30)],
        True,
    ),
    ("Сядь на синий контейнер.", [S("land", T("container", colour="blue"))], None),
    ("Залети внутрь оранжевого ангара и приземлись.", [S("enter", T("hangar", colour="orange"))], None),
    (
        "Сбрось груз на парашюте над красной зоной сброса к северу от склада, потом вернись.",
        [S("drop", T("dropzone", colour="red"), [R("north_of", **B(btype="warehouse"))])],
        True,
    ),
    (
        "Забери посылку с площадки с жёлтой коробкой у ветроуказателя.",
        [
            S(
                "pickup",
                T("pickup", parcel_colour="yellow"),
                [R("next_to", type="object", kind="windsock", colour=None)],
            )
        ],
        None,
    ),
    (
        "Приземлись на зелёную площадку с кругом, диаметр примерно 4 м, она на крыше девятиэтажного панельного дома.",
        [
            S(
                "land",
                T("pad", colour="green", mark="circle", size_m=4.0, on_roof=True),
                [R("on_roof_of", **B(btype="apartment", levels=9, material="panel"))],
            )
        ],
        None,
    ),
    ("Пролети возле красной палатки и возвращайся домой.", [S("flyby", T("tent", colour="red"))], True),
    (
        "Облети по кругу вышку связи у линии электропередачи, 10 метров.",
        [S("inspect", T("mast"), [R("next_to", type="power_line")], distance_m=10)],
        None,
    ),
    (
        "Сбрось посылку над синей мишенью с косым крестом.",
        [S("drop", T("dropzone", colour="blue", pattern="X"))],
        None,
    ),
    (
        "Нужно сесть на жёлтую посадочную площадку у дороги на Невском проспекте.",
        [S("land", T("pad", colour="yellow"), [R("by_street", type="street", name="Невский проспект")])],
        None,
    ),
    (
        "Возьми посылку из белого постамата, домой не возвращайся.",
        [S("pickup", T("pickup", colour="white", style="locker"))],
        False,
    ),
    (
        "Влети в синий ангар с открытыми воротами рядом с красным контейнером и сядь внутри, потом лети назад.",
        [
            S(
                "enter",
                T("hangar", colour="blue", door_open=True),
                [R("next_to", type="object", kind="container", colour="red")],
            )
        ],
        True,
    ),
    (
        "Пролети над водонапорной башней на высоте 50 метров.",
        [S("flyby", T("water_tower"), altitude_m=50)],
        None,
    ),
    (
        "Сядь на оранжевую площадку возле оранжевой палатки.",
        [S("land", T("pad", colour="orange"), [R("next_to", type="object", kind="tent", colour="orange")])],
        None,
    ),
    (
        "Облети солнечные панели в два ряда, расстояние 6 метров.",
        [S("inspect", T("solar", rows=2), distance_m=6)],
        None,
    ),
    (
        "Сбрось груз над зоной сброса к западу от частного дома, обратно не надо.",
        [S("drop", T("dropzone"), [R("west_of", **B(btype="house"))])],
        False,
    ),
    ("Забери посылку из жёлтого пункта выдачи.", [S("pickup", T("pickup", colour="yellow"))], None),
    (
        "Приземлись на чёрную круглую площадку с огнями у пруда.",
        [S("land", T("pad", colour="black", shape="round", lights=True), [R("next_to", type="water")])],
        None,
    ),
    (
        "Пролети рядом с вышкой связи высотой около 30 метров на высоте 45 метров и вернись.",
        [S("flyby", T("mast", height_m=30.0), altitude_m=45)],
        True,
    ),
    (
        "Влети в белый ангар у футбольного поля и сядь внутри. Назад не надо.",
        [S("enter", T("hangar", colour="white"), [R("next_to", type="pitch")])],
        False,
    ),
    (
        "Сядь на красную площадку к югу от церкви.",
        [S("land", T("pad", colour="red"), [R("south_of", type="church")])],
        None,
    ),
    (
        "Облети зелёную водонапорную башню на расстоянии 7 м.",
        [S("inspect", T("water_tower", colour="green"), distance_m=7)],
        None,
    ),
    (
        "Сбрось посылку над жёлтой зоной сброса с кольцами рядом с синим ангаром, потом домой.",
        [
            S(
                "drop",
                T("dropzone", colour="yellow", pattern="rings"),
                [R("next_to", type="object", kind="hangar", colour="blue")],
            )
        ],
        True,
    ),
    (
        "Забери коробку из синего постамата у дома 12 по улице Некрасова.",
        [
            S(
                "pickup",
                T("pickup", colour="blue", style="locker"),
                [R("next_to", **B(btype="apartment", address=["улица Некрасова", "12"]))],
            )
        ],
        None,
    ),
    (
        "Пролети мимо синего контейнера на высоте 10 метров, обратно не лети.",
        [S("flyby", T("container", colour="blue"), altitude_m=10)],
        False,
    ),
    (
        "Приземлись на маленькую белую площадку с буквой H у кирпичного дома, потом возвращайся.",
        [
            S(
                "land",
                T("pad", colour="white", size="small", mark="H"),
                [R("next_to", **B(btype="apartment", material="brick"))],
            )
        ],
        True,
    ),
    ("Сядь на площадку у детской площадки.", [S("land", T("pad"), [R("next_to", type="playground")])], None),
    (
        "Облети мачту связи на расстоянии 15 метров, она красная, стоит к северу от гаража.",
        [S("inspect", T("mast", colour="red"), [R("north_of", **B(btype="garage"))], distance_m=15)],
        None,
    ),
    (
        "Лети к ангару, но внутрь не залетай — просто пролети рядом на высоте 20 м.",
        [S("flyby", T("hangar"), altitude_m=20)],
        None,
    ),
]

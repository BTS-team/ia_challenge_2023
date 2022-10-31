language = {
    'austrian': 0,
    'belgian': 1,
    'bulgarian': 2,
    'croatian': 3,
    'cypriot': 4,
    'czech': 5,
    'danish': 6,
    'dutch': 7,
    'estonian': 8,
    'finnish': 9,
    'french': 10,
    'german': 11,
    'greek': 12,
    'hungarian': 13,
    'irish': 14,
    'italian': 15,
    'latvian': 16,
    'lithuanian': 17,
    'luxembourgish': 18,
    'maltese': 19,
    'polish': 20,
    'portuguese': 21,
    'romanian': 22,
    'slovakian': 23,
    'slovene': 24,
    'spanish': 25,
    'swedish': 26
}

city = {
    'amsterdam': 0,
    'copenhagen': 1,
    'madrid': 2,
    'paris': 3,
    'rome': 4,
    'sofia': 5,
    'valletta': 6,
    'vienna': 7,
    'vilnius': 8
}

brand = {
    'Independant': 0,
    'Royal Lotus': 1,
    'Boss Western': 2,
    'Tripletree': 3,
    'Quadrupletree': 4,
    'Corlton': 5,
    'Ibas': 6,
    '8 Premium': 7,
    'Ardisson': 8,
    'Chill Garden Inn': 9,
    'Marcure': 10,
    'Navatel': 11,
    'CourtYord': 12,
    'J.Halliday Inn': 13,
    'Morriot': 14,
    'Safitel': 15
}

group = {
    'Independant': 0,
    'Yin Yang': 1,
    'Boss Western': 2,
    'Chillton Worldwide': 3,
    'Morriott International': 4,
    'Accar Hotels': 5
}


def apply(x):
    if x in group.keys():
        return group[x]
    elif x in city.keys():
        return city[x]
    elif x in brand.keys():
        return brand[x]
    elif x in language.keys():
        return language[x]
    else:
        return x
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
    'Royal Lotus': 0,
    'Independant': 1,
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
    'Yin Yang': 0,
    'Independant': 1,
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


response_test = {
    'prices': [
        {'hotel_id': 12, 'price': 76, 'stock': 0},
        {'hotel_id': 567, 'price': 89, 'stock': 0},
        {'hotel_id': 630, 'price': 150, 'stock': 0},
        {'hotel_id': 307, 'price': 123, 'stock': 0},
        {'hotel_id': 807, 'price': 396, 'stock': 1},
        {'hotel_id': 317, 'price': 182, 'stock': 1},
        {'hotel_id': 809, 'price': 396, 'stock': 1},
        {'hotel_id': 269, 'price': 148, 'stock': 0},
        {'hotel_id': 821, 'price': 107, 'stock': 0},
        {'hotel_id': 441, 'price': 190, 'stock': 1},
        {'hotel_id': 603, 'price': 349, 'stock': 1},
        {'hotel_id': 557, 'price': 109, 'stock': 0},
        {'hotel_id': 588, 'price': 169, 'stock': 0},
        {'hotel_id': 211, 'price': 144, 'stock': 0},
        {'hotel_id': 657, 'price': 250, 'stock': 1},
        {'hotel_id': 517, 'price': 168, 'stock': 0},
        {'hotel_id': 85, 'price': 153, 'stock': 0},
        {'hotel_id': 271, 'price': 79, 'stock': 0},
        {'hotel_id': 103, 'price': 176, 'stock': 7},
        {'hotel_id': 163, 'price': 283, 'stock': 1},
        {'hotel_id': 450, 'price': 153, 'stock': 16},
        {'hotel_id': 158, 'price': 159, 'stock': 1},
        {'hotel_id': 745, 'price': 107, 'stock': 0},
        {'hotel_id': 172, 'price': 128, 'stock': 0},
        {'hotel_id': 925, 'price': 121, 'stock': 0},
        {'hotel_id': 960, 'price': 397, 'stock': 1},
        {'hotel_id': 515, 'price': 397, 'stock': 1},
        {'hotel_id': 706, 'price': 123, 'stock': 0},
        {'hotel_id': 86, 'price': 109, 'stock': 0},
        {'hotel_id': 762, 'price': 170, 'stock': 1},
        {'hotel_id': 752, 'price': 392, 'stock': 1},
        {'hotel_id': 892, 'price': 271, 'stock': 1},
        {'hotel_id': 373, 'price': 175, 'stock': 0},
        {'hotel_id': 779, 'price': 290, 'stock': 1},
        {'hotel_id': 326, 'price': 349, 'stock': 1},
        {'hotel_id': 658, 'price': 173, 'stock': 0},
        {'hotel_id': 283, 'price': 103, 'stock': 1},
        {'hotel_id': 917, 'price': 394, 'stock': 1},
        {'hotel_id': 442, 'price': 171, 'stock': 1},
        {'hotel_id': 674, 'price': 131, 'stock': 0},
        {'hotel_id': 193, 'price': 144, 'stock': 0},
        {'hotel_id': 967, 'price': 76, 'stock': 0},
        {'hotel_id': 303, 'price': 207, 'stock': 0},
        {'hotel_id': 460, 'price': 81, 'stock': 0},
        {'hotel_id': 223, 'price': 147, 'stock': 0},
        {'hotel_id': 369, 'price': 73, 'stock': 0},
        {'hotel_id': 835, 'price': 92, 'stock': 25},
        {'hotel_id': 778, 'price': 106, 'stock': 0},
        {'hotel_id': 760, 'price': 266, 'stock': 1},
        {'hotel_id': 891, 'price': 110, 'stock': 4},
        {'hotel_id': 750, 'price': 101, 'stock': 7},
        {'hotel_id': 581, 'price': 145, 'stock': 0},
        {'hotel_id': 10, 'price': 148, 'stock': 0},
        {'hotel_id': 429, 'price': 223, 'stock': 9},
        {'hotel_id': 230, 'price': 178, 'stock': 0},
        {'hotel_id': 410, 'price': 114, 'stock': 1},
        {'hotel_id': 117, 'price': 89, 'stock': 0},
        {'hotel_id': 585, 'price': 70, 'stock': 1},
        {'hotel_id': 724, 'price': 361, 'stock': 1},
        {'hotel_id': 878, 'price': 106, 'stock': 0},
        {'hotel_id': 806, 'price': 261, 'stock': 4},
        {'hotel_id': 971, 'price': 173, 'stock': 0},
        {'hotel_id': 214, 'price': 288, 'stock': 1},
        {'hotel_id': 256, 'price': 76, 'stock': 0},
        {'hotel_id': 604, 'price': 126, 'stock': 0},
        {'hotel_id': 218, 'price': 286, 'stock': 1},
        {'hotel_id': 680, 'price': 147, 'stock': 0},
        {'hotel_id': 16, 'price': 173, 'stock': 0},
        {'hotel_id': 660, 'price': 168, 'stock': 0},
        {'hotel_id': 902, 'price': 188, 'stock': 1},
        {'hotel_id': 157, 'price': 361, 'stock': 1},
        {'hotel_id': 196, 'price': 107, 'stock': 0},
        {'hotel_id': 916, 'price': 131, 'stock': 1},
        {'hotel_id': 921, 'price': 89, 'stock': 0},
        {'hotel_id': 969, 'price': 82, 'stock': 0},
        {'hotel_id': 311, 'price': 92, 'stock': 13},
        {'hotel_id': 880, 'price': 126, 'stock': 0},
        {'hotel_id': 676, 'price': 89, 'stock': 0},
        {'hotel_id': 9, 'price': 82, 'stock': 0},
        {'hotel_id': 264, 'price': 77, 'stock': 0},
        {'hotel_id': 82, 'price': 106, 'stock': 0},
        {'hotel_id': 507, 'price': 113, 'stock': 1},
        {'hotel_id': 510, 'price': 417, 'stock': 1},
        {'hotel_id': 52, 'price': 170, 'stock': 0},
        {'hotel_id': 45, 'price': 171, 'stock': 0},
        {'hotel_id': 788, 'price': 106, 'stock': 0},
        {'hotel_id': 366, 'price': 76, 'stock': 0},
        {'hotel_id': 945, 'price': 175, 'stock': 0},
        {'hotel_id': 587, 'price': 148, 'stock': 0},
        {'hotel_id': 505, 'price': 106, 'stock': 0},
        {'hotel_id': 594, 'price': 410, 'stock': 1},
        {'hotel_id': 455, 'price': 157, 'stock': 8},
        {'hotel_id': 140, 'price': 180, 'stock': 0},
        {'hotel_id': 111, 'price': 142, 'stock': 2},
        {'hotel_id': 591, 'price': 307, 'stock': 5},
        {'hotel_id': 475, 'price': 103, 'stock': 0},
        {'hotel_id': 482, 'price': 78, 'stock': 0},
        {'hotel_id': 715, 'price': 170, 'stock': 0},
        {'hotel_id': 22, 'price': 170, 'stock': 0},
        {'hotel_id': 105, 'price': 170, 'stock': 0},
        {'hotel_id': 570, 'price': 105, 'stock': 0},
        {'hotel_id': 387, 'price': 406, 'stock': 1},
        {'hotel_id': 867, 'price': 144, 'stock': 0},
        {'hotel_id': 493, 'price': 171, 'stock': 0},
        {'hotel_id': 234, 'price': 170, 'stock': 0},
        {'hotel_id': 628, 'price': 147, 'stock': 0},
        {'hotel_id': 679, 'price': 147, 'stock': 0},
        {'hotel_id': 375, 'price': 103, 'stock': 0},
        {'hotel_id': 83, 'price': 171, 'stock': 0},
        {'hotel_id': 146, 'price': 207, 'stock': 0},
        {'hotel_id': 34, 'price': 76, 'stock': 0},
        {'hotel_id': 417, 'price': 173, 'stock': 0},
        {'hotel_id': 101, 'price': 388, 'stock': 1},
        {'hotel_id': 743, 'price': 168, 'stock': 0},
        {'hotel_id': 933, 'price': 147, 'stock': 0},
        {'hotel_id': 352, 'price': 89, 'stock': 15},
        {'hotel_id': 777, 'price': 175, 'stock': 0},
        {'hotel_id': 548, 'price': 406, 'stock': 1},
        {'hotel_id': 627, 'price': 103, 'stock': 0},
        {'hotel_id': 224, 'price': 156, 'stock': 1},
        {'hotel_id': 783, 'price': 97, 'stock': 0}
    ],
    'request': {
        'city': 'amsterdam',
        'date': 1,
        'language': 'hungarian',
        'mobile': 1,
        'avatar_id': 6
    }
}

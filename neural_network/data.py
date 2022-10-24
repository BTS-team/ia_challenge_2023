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


response_test_1 = {
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

response_test_2 = {
    'prices': [
        {'hotel_id': 720, 'price': 467, 'stock': 2},
        {'hotel_id': 434, 'price': 114, 'stock': 0},
        {'hotel_id': 349, 'price': 317, 'stock': 1},
        {'hotel_id': 367, 'price': 96, 'stock': 0},
        {'hotel_id': 952, 'price': 94, 'stock': 0},
        {'hotel_id': 528, 'price': 432, 'stock': 2},
        {'hotel_id': 990, 'price': 168, 'stock': 0},
        {'hotel_id': 60, 'price': 94, 'stock': 0},
        {'hotel_id': 834, 'price': 91, 'stock': 0},
        {'hotel_id': 757, 'price': 151, 'stock': 1},
        {'hotel_id': 531, 'price': 204, 'stock': 2},
        {'hotel_id': 772, 'price': 327, 'stock': 0},
        {'hotel_id': 664, 'price': 472, 'stock': 2},
        {'hotel_id': 332, 'price': 333, 'stock': 1},
        {'hotel_id': 113, 'price': 89, 'stock': 0},
        {'hotel_id': 445, 'price': 89, 'stock': 0},
        {'hotel_id': 695, 'price': 202, 'stock': 2},
        {'hotel_id': 142, 'price': 390, 'stock': 4},
        {'hotel_id': 547, 'price': 202, 'stock': 2},
        {'hotel_id': 793, 'price': 167, 'stock': 0},
        {'hotel_id': 994, 'price': 193, 'stock': 2},
        {'hotel_id': 300, 'price': 145, 'stock': 1},
        {'hotel_id': 818, 'price': 173, 'stock': 0},
        {'hotel_id': 562, 'price': 327, 'stock': 1},
        {'hotel_id': 1, 'price': 95, 'stock': 0},
        {'hotel_id': 518, 'price': 194, 'stock': 1},
        {'hotel_id': 497, 'price': 269, 'stock': 4},
        {'hotel_id': 771, 'price': 478, 'stock': 2},
        {'hotel_id': 539, 'price': 351, 'stock': 3},
        {'hotel_id': 622, 'price': 213, 'stock': 1},
        {'hotel_id': 829, 'price': 202, 'stock': 2},
        {'hotel_id': 401, 'price': 196, 'stock': 0},
        {'hotel_id': 96, 'price': 194, 'stock': 1},
        {'hotel_id': 425, 'price': 308, 'stock': 1},
        {'hotel_id': 43, 'price': 79, 'stock': 16},
        {'hotel_id': 669, 'price': 322, 'stock': 1},
        {'hotel_id': 358, 'price': 306, 'stock': 1},
        {'hotel_id': 698, 'price': 231, 'stock': 6},
        {'hotel_id': 961, 'price': 229, 'stock': 7},
        {'hotel_id': 405, 'price': 97, 'stock': 0},
        {'hotel_id': 348, 'price': 198, 'stock': 2},
        {'hotel_id': 407, 'price': 317, 'stock': 1},
        {'hotel_id': 402, 'price': 128, 'stock': 0},
        {'hotel_id': 249, 'price': 87, 'stock': 0},
        {'hotel_id': 900, 'price': 235, 'stock': 0},
        {'hotel_id': 463, 'price': 97, 'stock': 0},
        {'hotel_id': 318, 'price': 141, 'stock': 0},
        {'hotel_id': 6, 'price': 120, 'stock': 0},
        {'hotel_id': 194, 'price': 118, 'stock': 0},
        {'hotel_id': 805, 'price': 202, 'stock': 2},
        {'hotel_id': 898, 'price': 92, 'stock': 0},
        {'hotel_id': 443, 'price': 145, 'stock': 1},
        {'hotel_id': 855, 'price': 311, 'stock': 1},
        {'hotel_id': 978, 'price': 335, 'stock': 4},
        {'hotel_id': 391, 'price': 191, 'stock': 2},
        {'hotel_id': 100, 'price': 87, 'stock': 0},
        {'hotel_id': 958, 'price': 239, 'stock': 0},
        {'hotel_id': 844, 'price': 469, 'stock': 2},
        {'hotel_id': 959, 'price': 221, 'stock': 4},
        {'hotel_id': 556, 'price': 175, 'stock': 0},
        {'hotel_id': 335, 'price': 90, 'stock': 0},
        {'hotel_id': 989, 'price': 108, 'stock': 0},
        {'hotel_id': 981, 'price': 110, 'stock': 0},
        {'hotel_id': 523, 'price': 104, 'stock': 3},
        {'hotel_id': 58, 'price': 322, 'stock': 1},
        {'hotel_id': 251, 'price': 165, 'stock': 0},
        {'hotel_id': 53, 'price': 94, 'stock': 0},
        {'hotel_id': 770, 'price': 336, 'stock': 1},
        {'hotel_id': 707, 'price': 335, 'stock': 5},
        {'hotel_id': 766, 'price': 467, 'stock': 2},
        {'hotel_id': 508, 'price': 339, 'stock': 1},
        {'hotel_id': 761, 'price': 145, 'stock': 0},
        {'hotel_id': 872, 'price': 97, 'stock': 0},
        {'hotel_id': 110, 'price': 202, 'stock': 2},
        {'hotel_id': 153, 'price': 333, 'stock': 1},
        {'hotel_id': 546, 'price': 108, 'stock': 1},
        {'hotel_id': 78, 'price': 461, 'stock': 2},
        {'hotel_id': 532, 'price': 342, 'stock': 1},
        {'hotel_id': 596, 'price': 170, 'stock': 0},
        {'hotel_id': 983, 'price': 91, 'stock': 0},
        {'hotel_id': 36, 'price': 193, 'stock': 0},
        {'hotel_id': 243, 'price': 97, 'stock': 0},
        {'hotel_id': 726, 'price': 325, 'stock': 3},
        {'hotel_id': 506, 'price': 148, 'stock': 0},
        {'hotel_id': 280, 'price': 117, 'stock': 12},
        {'hotel_id': 876, 'price': 141, 'stock': 0},
        {'hotel_id': 716, 'price': 143, 'stock': 0},
        {'hotel_id': 559, 'price': 299, 'stock': 1},
        {'hotel_id': 422, 'price': 481, 'stock': 2},
        {'hotel_id': 826, 'price': 196, 'stock': 8},
        {'hotel_id': 727, 'price': 121, 'stock': 0},
        {'hotel_id': 365, 'price': 94, 'stock': 0},
        {'hotel_id': 908, 'price': 481, 'stock': 2},
        {'hotel_id': 267, 'price': 173, 'stock': 0},
        {'hotel_id': 930, 'price': 120, 'stock': 0},
        {'hotel_id': 487, 'price': 166, 'stock': 8},
        {'hotel_id': 91, 'price': 128, 'stock': 0},
        {'hotel_id': 744, 'price': 170, 'stock': 0},
        {'hotel_id': 23, 'price': 198, 'stock': 2},
        {'hotel_id': 90, 'price': 200, 'stock': 2},
        {'hotel_id': 922, 'price': 342, 'stock': 1},
        {'hotel_id': 865, 'price': 277, 'stock': 7},
        {'hotel_id': 435, 'price': 464, 'stock': 2},
        {'hotel_id': 572, 'price': 231, 'stock': 0},
        {'hotel_id': 329, 'price': 239, 'stock': 0},
        {'hotel_id': 691, 'price': 89, 'stock': 0},
        {'hotel_id': 831, 'price': 108, 'stock': 5},
        {'hotel_id': 973, 'price': 171, 'stock': 0},
        {'hotel_id': 291, 'price': 256, 'stock': 1},
        {'hotel_id': 938, 'price': 121, 'stock': 0},
        {'hotel_id': 202, 'price': 198, 'stock': 2},
        {'hotel_id': 32, 'price': 175, 'stock': 0},
        {'hotel_id': 812, 'price': 200, 'stock': 0},
        {'hotel_id': 431, 'price': 123, 'stock': 0},
        {'hotel_id': 289, 'price': 140, 'stock': 0},
        {'hotel_id': 688, 'price': 87, 'stock': 0},
        {'hotel_id': 92, 'price': 84, 'stock': 3},
        {'hotel_id': 690, 'price': 241, 'stock': 4},
        {'hotel_id': 611, 'price': 204, 'stock': 0},
        {'hotel_id': 638, 'price': 153, 'stock': 6},
        {'hotel_id': 995, 'price': 214, 'stock': 3},
        {'hotel_id': 541, 'price': 155, 'stock': 1},
        {'hotel_id': 711, 'price': 140, 'stock': 2},
        {'hotel_id': 433, 'price': 143, 'stock': 2},
        {'hotel_id': 309, 'price': 94, 'stock': 0},
        {'hotel_id': 893, 'price': 229, 'stock': 0},
        {'hotel_id': 830, 'price': 89, 'stock': 0},
        {'hotel_id': 849, 'price': 198, 'stock': 2},
        {'hotel_id': 481, 'price': 95, 'stock': 0},
        {'hotel_id': 50, 'price': 149, 'stock': 1},
        {'hotel_id': 869, 'price': 469, 'stock': 1},
        {'hotel_id': 951, 'price': 87, 'stock': 0},
        {'hotel_id': 99, 'price': 202, 'stock': 0},
        {'hotel_id': 616, 'price': 198, 'stock': 0}
    ],
    'request': {
        'city': 'copenhagen',
        'date': 2,
        'language': 'hungarian',
        'mobile': 0,
        'avatar_id': 5
    }
}

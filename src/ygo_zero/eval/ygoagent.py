"""ygo-agent 0546 checkpoint as a gauntlet opponent.

Adapts our observation/action spec (``docs/obs_spec.md``, ``docs/action_spec.md``) to the feature
packing ygo-agent's TFLite model expects. All lookup tables below are copied from
``../ygo-agent/ygoinf/ygoinf/features.py`` (and the column layouts from
``../ygo-agent/docs/feature_engineering.md``); ygo-agent is never imported at runtime.

Key differences bridged here:
- Card codes: ours are raw ``code`` ints; theirs are indices into the checkpoint's *vintage*
  code list (``_VINTAGE_CODES`` below; unknown/out-of-vocab → 0). This is NOT the order of today's
  ``data/models/code_list.txt`` (see NOTEBOOK 2026-10-02): the exported embedding table holds
  exactly 1000 rows and gathers with ids ≥ 1000 crash the interpreter (measured), so feeding
  current-list ids is both wrong and fatal.
- Effect descriptions: ours are edo ``code << 20 | idx`` (already split into effect_code/effect_idx
  columns); theirs are classic ``code << 4 | idx`` packed into one effect id.
- Our card rows keep our own order; each encoded action's spec-index column points at the right
  (1-based) row of the encoded card array.
- History: ours is most-recent-first with turns-ago; theirs is oldest-first with the same turns-ago
  clamp, built here directly from our ``history`` buffer each call (no separate ring needed).
- Recurrent LSTM state is kept per (game id, acting seat); a new game id starts from zeros.
"""

from pathlib import Path

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind

N_CARD_FEATURES = 41
N_CARDS = 160
N_ACTION_FEATURES = 12
MAX_ACTIONS = 24
N_GLOBAL_FEATURES = 23
N_HISTORY_ACTIONS = 32
H_ACTION_FEATURES = 14
N_RNN_CHANNELS = 512

MODEL_PATH = Path("data/models/0546_26550M.tflite")
CODE_LIST_PATH = Path("data/models/code_list.txt")

# --- lookup tables copied from ../ygo-agent/ygoinf/ygoinf/features.py ---

# features.py:73-82 (`system_string_to_id = {s: i + 16 ...}`).
SYSTEM_STRINGS = [
    1050,
    1051,
    1052,
    1054,
    1055,
    1056,
    1057,
    1058,
    1059,
    1060,
    1061,
    1062,
    1063,
    1064,
    1066,
    1067,
    1068,
    1069,
    1070,
    1071,
    1072,
    1073,
    1074,
    1075,
    1076,
    1080,
    1081,
    1150,
    1151,
    1152,
    1153,
    1154,
    1155,
    1156,
    1157,
    1158,
    1159,
    1160,
    1161,
    1162,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1190,
    1191,
    1192,
    1193,
    1,
    30,
    31,
    80,
    81,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    200,
    203,
    210,
    218,
    219,
    220,
    221,
    222,
    221,
    1621,
    1622,
]
SYSTEM_STRING_TO_ID = {s: i + 16 for i, s in enumerate(SYSTEM_STRINGS)}

# features.py:109-117 (location_to_id). Our location ids (obs_spec.md) already match except overlay
# (ours 8), which has no counterpart: overlay units ride on a holder that is virtually always in
# mzone, so they encode as mzone with the overlay flag set.
LOCATION_TO_ID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 3}

# features.py:124-134 (position_to_id), keyed by our position encoding (obs_spec.md col 5).
# Ours: mzone 1 FU-ATK, 2 FD-ATK, 3 FU-DEF, 4 FD-DEF; other zones 5 face-up, 6 face-down; else 0.
POSITION_TO_ID = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 5, 6: 7}

# features.py:141-150 (attribute_to_id), keyed by ATTRIBUTE_* bit.
ATTRIBUTE_TO_ID = {
    C.ATTRIBUTE_EARTH: 1, C.ATTRIBUTE_WATER: 2, C.ATTRIBUTE_FIRE: 3, C.ATTRIBUTE_WIND: 4,
    C.ATTRIBUTE_LIGHT: 5, C.ATTRIBUTE_DARK: 6, C.ATTRIBUTE_DIVINE: 7,
}  # fmt: skip

# features.py:152-180 (race_to_id), keyed by RACE_* bit. Our obs folds RACE_YOKAI onto bit 31
# (shared with RACE_GALAXY, unused by any TCG card); neither is in their table, so both map to 0
# (likewise the non-TCG races CYBORG/MAGICALKNIGHT/HIGHDRAGON/OMEGAPSYCHIC/CELESTIALWARRIOR).
RACE_TO_ID = {
    C.RACE_WARRIOR: 1, C.RACE_SPELLCASTER: 2, C.RACE_FAIRY: 3, C.RACE_FIEND: 4,
    C.RACE_ZOMBIE: 5, C.RACE_MACHINE: 6, C.RACE_AQUA: 7, C.RACE_PYRO: 8,
    C.RACE_ROCK: 9, C.RACE_WINGEDBEAST: 10, C.RACE_PLANT: 11, C.RACE_INSECT: 12,
    C.RACE_THUNDER: 13, C.RACE_DRAGON: 14, C.RACE_BEAST: 15, C.RACE_BEASTWARRIOR: 16,
    C.RACE_DINOSAUR: 17, C.RACE_FISH: 18, C.RACE_SEASERPENT: 19, C.RACE_REPTILE: 20,
    C.RACE_PSYCHIC: 21, C.RACE_DIVINE: 22, C.RACE_CREATORGOD: 23, C.RACE_WYRM: 24,
    C.RACE_CYBERSE: 25, C.RACE_ILLUSION: 26,
}  # fmt: skip

# features.py:187-213 (type_to_id), keyed by TYPE_* bit (TYPE_GEMINI is their "dual",
# TYPE_SPSUMMON their "special"; TYPE_MAXIMUM has no counterpart).
TYPE_TO_OFFSET = {
    C.TYPE_MONSTER: 0, C.TYPE_SPELL: 1, C.TYPE_TRAP: 2, C.TYPE_NORMAL: 3, C.TYPE_EFFECT: 4,
    C.TYPE_FUSION: 5, C.TYPE_RITUAL: 6, C.TYPE_TRAPMONSTER: 7, C.TYPE_SPIRIT: 8,
    C.TYPE_UNION: 9, C.TYPE_GEMINI: 10, C.TYPE_TUNER: 11, C.TYPE_SYNCHRO: 12,
    C.TYPE_TOKEN: 13, C.TYPE_QUICKPLAY: 14, C.TYPE_CONTINUOUS: 15, C.TYPE_EQUIP: 16,
    C.TYPE_FIELD: 17, C.TYPE_COUNTER: 18, C.TYPE_FLIP: 19, C.TYPE_TOON: 20,
    C.TYPE_XYZ: 21, C.TYPE_PENDULUM: 22, C.TYPE_SPSUMMON: 23, C.TYPE_LINK: 24,
}  # fmt: skip

# features.py:215-226 (phase_to_id), keyed by PHASE_* bit.
PHASE_TO_ID = {
    C.PHASE_DRAW: 0, C.PHASE_STANDBY: 1, C.PHASE_MAIN1: 2, C.PHASE_BATTLE_START: 3,
    C.PHASE_BATTLE_STEP: 4, C.PHASE_DAMAGE: 5, C.PHASE_DAMAGE_CAL: 6, C.PHASE_BATTLE: 7,
    C.PHASE_MAIN2: 8, C.PHASE_END: 9,
}  # fmt: skip

# features.py:228-244 (msg_to_id), keyed by our MSG_* int. Messages their model never saw
# (ANNOUNCE_RACE/CARD, SORT_*, SELECT_COUNTER, RPS) map to 0 (N/A); the choice stays legal because
# we map the winning probability slot back to our option index.
MSG_TO_ID = {
    C.MSG_SELECT_IDLECMD: 1, C.MSG_SELECT_CHAIN: 2, C.MSG_SELECT_CARD: 3, C.MSG_SELECT_TRIBUTE: 4,
    C.MSG_SELECT_POSITION: 5, C.MSG_SELECT_EFFECTYN: 6, C.MSG_SELECT_YESNO: 7,
    C.MSG_SELECT_BATTLECMD: 8, C.MSG_SELECT_UNSELECT_CARD: 9, C.MSG_SELECT_OPTION: 10,
    C.MSG_SELECT_PLACE: 11, C.MSG_SELECT_SUM: 12, C.MSG_SELECT_DISFIELD: 13,
    C.MSG_ANNOUNCE_ATTRIB: 14, C.MSG_ANNOUNCE_NUMBER: 15,
}  # fmt: skip

# Our option kind -> (their act id, their phase id). features.py:247-286
# (action_act_to_id / action_phase_to_id). TO_* moves carry no act, only a phase; PASS and NO are
# their "cancel"; SELECT-ish and placement options carry no act.
KIND_TO_ACT_PHASE = {
    Kind.SUMMON: (4, 0), Kind.SPSUMMON: (3, 0), Kind.REPOS: (2, 0), Kind.MSET: (5, 0),
    Kind.SSET: (1, 0), Kind.ACTIVATE: (8, 0), Kind.ATTACK: (6, 0), Kind.TO_BATTLE: (0, 1),
    Kind.TO_MAIN2: (0, 2), Kind.TO_END: (0, 3), Kind.PASS: (9, 0), Kind.YES: (8, 0),
    Kind.NO: (9, 0), Kind.OPTION: (8, 0), Kind.SELECT: (0, 0), Kind.UNSELECT: (0, 0),
    Kind.FINISH: (0, 0), Kind.CANCEL: (9, 0), Kind.PLACE: (0, 0), Kind.POSITION: (0, 0),
    Kind.NUMBER: (0, 0), Kind.ATTRIBUTE: (0, 0), Kind.RACE: (0, 0), Kind.DECLARE: (0, 0),
    Kind.SORT: (0, 0), Kind.COUNTER: (0, 0), Kind.RPS: (0, 0),
}  # fmt: skip

# Our POSITION arg is a POS_* bit; theirs is a position id (features.py:632 posotion_to_response).
POS_BIT_TO_ID = {
    C.POS_FACEUP_ATTACK: 1,
    C.POS_FACEDOWN_ATTACK: 2,
    C.POS_FACEUP_DEFENSE: 4,
    C.POS_FACEDOWN_DEFENSE: 6,
}

# Kinds whose (effect_code, effect_idx) == (0, 0) still means "an effect with no description"
# (their effect 0 -> encoded 1) rather than "no effect" (their -1 -> encoded 0).
_EFFECTFUL_KINDS = frozenset({Kind.ACTIVATE, Kind.YES, Kind.NO, Kind.OPTION})


def load_code_ids(path: str | Path) -> dict[int, int]:
    """Card code -> 1-based id (line order, like features.py:init_code_list). Unknown codes -> 0.

    Accepts both the single-column format init_code_list parses and the two-column
    "<code> <has_script>" format this repo's code_list.txt is stored in (first token wins).
    This is the *reference* mapping; the policy itself uses the checkpoint's vintage order
    (vintage_code_ids), which is what the baked-in embedding table was trained against.
    """
    ids = {0: 0}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids[int(line.split()[0])] = len(ids)
    return ids


# The checkpoint's card-id table, recovered 2026-10-02 by matching the TFLite embedding matrix
# (tensor 272, shape (1000, 1024)) against ../ygorl/refs/assets/embed864.pkl (864 voyage vectors):
# row 0 == the embeddings' mean (their "unknown" row, cf. cleanba.py), rows 1..864 == a permutation
# of the 864 vectors (exact float match, each key used once), rows 865..999 == 0 (padding).
# Order below is that permutation: id i (1-based) <-> _VINTAGE_CODES[i - 1]. Ids were verified
# in-vocab by probing the interpreter (0..999 run, >= 1000 raise gather_nd OOB).
_VINTAGE_CODES = (
    82255873,
    91512836,
    21770261,
    64213018,
    3659803,
    7610395,
    89907228,
    10045474,
    95440946,
    59822133,
    65810490,
    71645243,
    62849088,
    94224458,
    58921041,
    55273560,
    1142880,
    10604644,
    28674153,
    1984618,
    1906812,
    46659709,
    67489920,
    79814787,
    29491335,
    90673288,
    84058253,
    99092625,
    78917791,
    52900001,
    53303460,
    42956964,
    32446631,
    31801517,
    69550260,
    5402805,
    55410871,
    26077387,
    43096270,
    82340057,
    53129443,
    33698022,
    15341822,
    15341823,
    28053764,
    35834120,
    75147529,
    176393,
    44095762,
    70465811,
    35268888,
    44689689,
    45484331,
    8491308,
    60242223,
    70095154,
    90884404,
    74889525,
    60514626,
    64753988,
    20357457,
    98462037,
    38041941,
    56910167,
    22411610,
    77672445,
    84816245,
    5370235,
    23837055,
    29948294,
    58069384,
    6498706,
    2625940,
    90448279,
    31480216,
    13935002,
    89631139,
    60764582,
    60764583,
    75622825,
    48800175,
    9396663,
    50954680,
    22110648,
    75119041,
    81439173,
    64332231,
    53334471,
    74983882,
    48048590,
    98875864,
    63013339,
    8198621,
    78836195,
    78836196,
    76589547,
    17418745,
    75452921,
    93104633,
    94145021,
    53971455,
    18973184,
    14821890,
    14821891,
    39973386,
    35052053,
    26556950,
    44026394,
    3410461,
    68815402,
    3285552,
    30765616,
    27198002,
    86682165,
    68829754,
    56465981,
    12421694,
    23331401,
    72272462,
    79229522,
    63941210,
    83755611,
    2978414,
    85969518,
    31986289,
    22493812,
    28062326,
    9464441,
    5141117,
    67007102,
    18027139,
    38120068,
    18027140,
    6853254,
    18027141,
    31533705,
    82324106,
    36577931,
    66200211,
    30069399,
    16943771,
    20001443,
    2857636,
    20001444,
    49881766,
    73915051,
    73915052,
    73915053,
    73915054,
    73915055,
    66788016,
    78394033,
    97268402,
    23431858,
    9929399,
    36637374,
    34822851,
    26096328,
    29301450,
    35263181,
    63767246,
    19280590,
    21082832,
    27204311,
    27204312,
    49752796,
    93039339,
    78789357,
    51409648,
    59900656,
    40633085,
    75500286,
    79387393,
    32056071,
    21179144,
    72291079,
    37630732,
    85771020,
    85771021,
    67922703,
    97452818,
    56597273,
    41329459,
    62481204,
    70875956,
    83239740,
    60406592,
    5043010,
    49808197,
    99111753,
    43422537,
    82738008,
    35984222,
    42427231,
    6728559,
    33676147,
    25451383,
    9726840,
    30327675,
    30327676,
    1041278,
    14957441,
    44092305,
    39701395,
    69890968,
    38030233,
    71039903,
    32807846,
    14558127,
    41999284,
    11654068,
    69247929,
    56984514,
    31036355,
    62962630,
    31833038,
    86686671,
    15590356,
    83104731,
    76524507,
    74157028,
    38517737,
    85289965,
    30811117,
    40551411,
    72860663,
    904186,
    34767866,
    20612097,
    23116809,
    69868556,
    26412047,
    42632209,
    21830680,
    60025884,
    27450401,
    56495147,
    49032236,
    56495148,
    18494512,
    42671152,
    24224830,
    93850690,
    8240199,
    46052429,
    92892239,
    45644898,
    96633955,
    94973029,
    98978921,
    17228909,
    43664495,
    44330099,
    51987572,
    98159737,
    87116928,
    44362883,
    36629636,
    28355719,
    67284108,
    95515789,
    22850702,
    87669905,
    19096726,
    90846359,
    82556059,
    53855410,
    53855411,
    10443957,
    70534340,
    86066372,
    5041348,
    67949764,
    12958920,
    41456842,
    65330383,
    63995093,
    14089429,
    30190809,
    51227866,
    50005218,
    3814632,
    23893227,
    73594093,
    40703223,
    18144506,
    22953212,
    63288573,
    56364287,
    69248256,
    40844553,
    35269904,
    54537490,
    1426715,
    39972130,
    15394084,
    93130022,
    1799465,
    23434538,
    66661679,
    35261759,
    73964868,
    82994510,
    84815190,
    48411997,
    9925983,
    13536607,
    63184228,
    65500516,
    98338152,
    2819436,
    74440056,
    11738490,
    20368764,
    62125439,
    25415053,
    73082255,
    39030163,
    23068051,
    51611042,
    9047461,
    99550630,
    94703022,
    11091375,
    77542832,
    11050416,
    11050417,
    11050418,
    11050419,
    44586427,
    46724542,
    86801872,
    81767889,
    70342110,
    41620959,
    7392746,
    15629802,
    83764718,
    46173680,
    46173681,
    27882994,
    69811711,
    38053382,
    26326542,
    44097051,
    23516703,
    62543394,
    43898403,
    39730727,
    35726888,
    34479659,
    17751597,
    41373230,
    25419324,
    86871615,
    14532163,
    22404676,
    37129797,
    48115278,
    40908371,
    58371672,
    84797028,
    44052075,
    70368879,
    55326323,
    44146295,
    93191801,
    6631034,
    12580477,
    59160189,
    24874631,
    87746184,
    63166095,
    2295440,
    75732623,
    29843092,
    29843093,
    29843094,
    10389143,
    82732705,
    16946850,
    93490856,
    93490857,
    63442605,
    87240372,
    43140792,
    43202238,
    38342335,
    14470846,
    14470847,
    10000080,
    63356631,
    52340444,
    37351133,
    52340445,
    45883110,
    17000166,
    68468459,
    35514097,
    99137266,
    99137267,
    74997493,
    98596597,
    80701178,
    74659583,
    11321089,
    93912846,
    48068379,
    25311006,
    3987233,
    30650148,
    70391589,
    70950699,
    45467446,
    88923964,
    60600126,
    83994433,
    72302403,
    59438930,
    32335698,
    83308376,
    70369116,
    44308318,
    8025951,
    62541668,
    46647145,
    51208047,
    24010609,
    64382840,
    4417408,
    12965762,
    31600514,
    85243785,
    34690954,
    1546123,
    60303245,
    93224849,
    31434645,
    60465049,
    32731036,
    28776350,
    87052196,
    64583601,
    20590515,
    56051649,
    38811586,
    37576645,
    74627017,
    47658965,
    95492061,
    24915933,
    645088,
    47710198,
    13764603,
    75524093,
    32519092,
    89870349,
    56733747,
    19324993,
    18094166,
    52947044,
    93347961,
    75047173,
    30757127,
    40044918,
    21143940,
    16605586,
    45906428,
    59392529,
    40854197,
    60461804,
    9411399,
    58004362,
    58481572,
    1948619,
    46759931,
    32828466,
    90590303,
    83965310,
    14124483,
    27780618,
    50720316,
    22865492,
    89943723,
    8949584,
    22908820,
    24094653,
    55990317,
    52445243,
    93229151,
    94445733,
    31562086,
    28373620,
    3734202,
    33198837,
    92107604,
    68957034,
    42566602,
    29942771,
    7478431,
    30430448,
    66712905,
    87188910,
    80666118,
    20618850,
    34813545,
    67835547,
    93454062,
    8728498,
    24299458,
    66011101,
    56399890,
    77058170,
    27572350,
    15397015,
    51452091,
    73642296,
    43262273,
    25774450,
    8233522,
    43455065,
    31849106,
    34267821,
    25789292,
    82385847,
    70902743,
    3679218,
    5821478,
    83326048,
    23002292,
    8267140,
    4031928,
    11109820,
    84749824,
    5758500,
    15693423,
    75732622,
    19613556,
    12058741,
    61245672,
    80845034,
    90241276,
    9742784,
    72270339,
    27548199,
    53639887,
    38511382,
    48452496,
    26700718,
    65741786,
    89023486,
    79606837,
    20665527,
    50091196,
    45112597,
    9674034,
    2772337,
    45663742,
    24081957,
    27381364,
    87871125,
    48815792,
    98127546,
    35405755,
    4280258,
    73347079,
    2009101,
    81983656,
    85215458,
    73652465,
    58820853,
    81105204,
    91351370,
    82633039,
    16195942,
    17377751,
    27243130,
    95040215,
    1475311,
    53582587,
    75498415,
    78156759,
    52687916,
    86848580,
    23338098,
    49003716,
    14785765,
    69031175,
    59839761,
    76913983,
    16051717,
    72930878,
    53567095,
    14087893,
    33236860,
    73580471,
    22835145,
    5318639,
    11827244,
    4939890,
    3717252,
    84013237,
    50907446,
    48130397,
    23912837,
    84433295,
    19261966,
    97518132,
    31924889,
    74822425,
    20366274,
    56832966,
    77505534,
    69764158,
    34710660,
    6417578,
    59546797,
    51023024,
    94977269,
    40605147,
    37445295,
    4904633,
    30328508,
    44394295,
    77723643,
    48424886,
    24635329,
    91703676,
    66431519,
    2526224,
    18621798,
    65305978,
    57554544,
    90681088,
    85106525,
    64182380,
    8264361,
    57134592,
    4810828,
    51296484,
    49238328,
    94259633,
    13048472,
    73898890,
    52472775,
    25801745,
    80532587,
    26866984,
    9839945,
    98477480,
    88284599,
    92919429,
    71818935,
    10774240,
    39114494,
    86310763,
    84211599,
    10804018,
    15982593,
    56111151,
    98095162,
    27541563,
    84330567,
    11765832,
    22024279,
    34773082,
    68431965,
    72167543,
    38436986,
    82489470,
    63436931,
    38264974,
    33407125,
    51697825,
    69087397,
    32909498,
    11110587,
    21887175,
    19271881,
    66730191,
    43227,
    55204071,
    37961969,
    84478195,
    51405049,
    80611581,
    10019086,
    35146019,
    6351147,
    1225009,
    89252153,
    22993208,
    60303688,
    77543769,
    53143898,
    33878367,
    86188410,
    54757758,
    1769875,
    5818798,
    59388357,
    71832012,
    2511,
    55521751,
    97698279,
    27132400,
    72444406,
    71858682,
    25862681,
    29587993,
    81674782,
    65681983,
    5560911,
    77765207,
    21844576,
    69540484,
    2347656,
    74920585,
    24361622,
    73542331,
    73468603,
    48626373,
    77202120,
    84271823,
    91800273,
    36975314,
    36668118,
    21123811,
    63136489,
    27552504,
    63533837,
    52038441,
    18940725,
    72971064,
    4160316,
    26268488,
    77103950,
    59185998,
    10963799,
    73628505,
    5380979,
    69680031,
    71607202,
    34995106,
    56099748,
    73218989,
    69327790,
    28226490,
    82570174,
    54334420,
    50277355,
    81003500,
    91810826,
    6763530,
    65326118,
    4928565,
    30336082,
    32756828,
    58143852,
    30983281,
    6767771,
    33854624,
    93854893,
    69946549,
    42790071,
    81497285,
    55063751,
    25533642,
    91880660,
    54498517,
    42493140,
    40139997,
    42009836,
    41420027,
    53804307,
    1686814,
    18969888,
    1508649,
    39402797,
    73956664,
    68304193,
    60362066,
    53329234,
    17827173,
    58932615,
    34755994,
    572850,
    38798785,
    81767888,
    98506199,
    15005145,
    28954097,
    97682931,
    28126717,
    73539069,
    61470213,
    92714517,
    48608796,
    4931121,
    75730490,
    47355498,
    39931513,
    77207191,
    9012916,
    4796100,
    47961808,
    53212882,
    78661338,
    87639778,
    98645731,
    58699500,
    92907248,
    69601012,
    45819647,
    37629703,
    6637331,
    80117527,
    23076639,
    99585850,
    30748475,
    93503294,
    41215808,
    24508238,
    35059553,
    46565218,
    55461744,
    61173621,
    40155014,
    92565383,
    7436169,
    80433039,
    34090915,
    62015408,
    54126514,
    82956214,
    78888899,
    41371602,
    34541543,
    89538537,
    35809262,
    92731385,
    74018812,
    92907249,
    49959355,
    41562624,
    74586817,
    98558751,
    59843383,
    40364916,
    99423156,
    83283063,
    92826944,
    57288064,
    26326541,
    52467217,
    4333086,
    91420202,
    86926989,
    55623480,
    52711246,
    36630403,
    13965201,
    94801854,
    79783880,
)


def vintage_code_ids(code_list_path: str | Path | None = None) -> dict[int, int]:
    """The checkpoint's id table: 0 -> 0, vintage code i -> id i + 1; everything else -> 0.

    If ``code_list_path`` is given, validates that the staged reference list contains every
    vintage code (it must be a superset of the training list, else the staging is mixed up).
    """
    if code_list_path is not None:
        staged = set(load_code_ids(code_list_path)) - {0}
        missing = [c for c in _VINTAGE_CODES if c not in staged]
        if missing:
            raise ValueError(f"{len(missing)} vintage codes missing from {code_list_path}")
    return {0: 0} | {code: i + 1 for i, code in enumerate(_VINTAGE_CODES)}


def int_transform(x: int) -> tuple[int, int]:
    """Card-id uint16 split (features.py:453)."""
    return x // 256, x % 256


def float_transform(x: int) -> tuple[int, int]:
    """LP/ATK/DEF transform: value mod 65536 split to 2 bytes (features.py:457)."""
    x = int(x) % 65536
    return x // 256, x % 256


def effect_to_id(kind: int, effect_code: int, effect_idx: int) -> int:
    """Map our (effect_code, effect_idx) columns to their packed effect id (encode_action rule).

    Ours is edo ``code << 20 | idx`` pre-split; theirs packs classic ``code << 4 | idx`` descs as:
    -1 (no effect) -> 0, 0 -> 1, card effect (offset + idx) -> idx + 2, else a system-string id.
    """
    if effect_code == 0 and effect_idx == 0:
        return 1 if kind in _EFFECTFUL_KINDS else 0
    if effect_code == 0:
        return SYSTEM_STRING_TO_ID.get(effect_idx, 0)
    return (effect_idx + 2) & 0xFF


def encode_cards(cards: np.ndarray, code_to_id: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Our cards rows [160, 21] -> their cards [160, 41] uint8 plus index map.

    Row order is kept; ``index_map[i]`` is the 1-based encoded row for our card row ``i``, so an
    action's spec-index column is ``index_map[card_row]`` (0 when the action names no card).
    """
    out = np.zeros((N_CARDS, N_CARD_FEATURES), dtype=np.uint8)
    for i in range(min(N_CARDS, len(cards))):
        r = cards[i]
        code, loc = int(r[0]), int(r[1])
        seq, ctrl = int(r[2]), int(r[3])
        overlay = loc == 8
        cid = code_to_id.get(code, 0)
        out[i, 0], out[i, 1] = int_transform(cid)
        out[i, 2] = LOCATION_TO_ID.get(loc, 0)
        if loc in (3, 4, 5) or overlay:
            out[i, 3] = min(max(seq, 0), 255)
        out[i, 4] = ctrl if ctrl in (0, 1) else 0
        out[i, 5] = 5 if overlay else POSITION_TO_ID.get(int(r[5]), 0)
        out[i, 6] = 1 if overlay else 0
        out[i, 7] = ATTRIBUTE_TO_ID.get(int(r[8]), 0)
        out[i, 8] = RACE_TO_ID.get(int(r[9]) & 0xFFFFFFFF, 0)
        level = int(r[10]) or int(r[11]) or int(r[12])  # their level folds rank/link in
        out[i, 9] = min(max(level, 0), 13)
        # col 10 (counters) is not in our obs; col 11 negated = disabled or forbidden.
        out[i, 11] = 1 if int(r[19]) & (C.STATUS_DISABLED | C.STATUS_FORBIDDEN) else 0
        out[i, 12:14] = float_transform(int(r[14]))
        out[i, 14:16] = float_transform(int(r[15]))
        types = int(r[7]) & 0xFFFFFFFF
        for bit, off in TYPE_TO_OFFSET.items():
            if types & bit:
                out[i, 16 + off] = 1
    index_map = np.arange(1, N_CARDS + 1, dtype=np.int64)
    return out, index_map


def encode_global(g: np.ndarray) -> np.ndarray:
    """Our global row [24] -> their global [23] uint8 (counts copied straight across)."""
    x = np.zeros(N_GLOBAL_FEATURES, dtype=np.uint8)
    x[0:2] = float_transform(int(g[0]))
    x[2:4] = float_transform(int(g[1]))
    x[4] = min(max(int(g[2]), 0), 16)
    x[5] = PHASE_TO_ID.get(int(g[3]), 0)
    x[6] = 1 if g[4] else 0
    x[7] = 1 if g[5] else 0
    for j in range(14):
        x[8 + j] = min(max(int(g[6 + j]), 0), 255)
    # x[22] (is_end) stays 0, as in their encode_global.
    return x


def encode_option(
    kind: int,
    code: int,
    effect_code: int,
    effect_idx: int,
    arg: int,
    msg: int,
    spec_index: int,
    code_to_id: dict[int, int],
) -> np.ndarray:
    """One of our options -> one of their 12-col legal-action rows."""
    x = np.zeros(N_ACTION_FEATURES, dtype=np.uint8)
    x[0] = spec_index & 0xFF
    cid = code_to_id.get(code, 0)
    x[1], x[2] = int_transform(cid)
    x[3] = MSG_TO_ID.get(msg, 0)
    act, phase = KIND_TO_ACT_PHASE.get(kind, (0, 0))
    if kind == Kind.ATTACK and arg == 1:  # directly attackable (their direct_attack)
        act = 7
    x[4] = act
    x[5] = 1 if kind == Kind.FINISH else 0
    x[6] = effect_to_id(kind, effect_code, effect_idx)
    x[7] = phase
    if kind == Kind.POSITION:
        x[8] = POS_BIT_TO_ID.get(arg & 0xFFFFFFFF, 0)
    elif kind == Kind.NUMBER:
        x[9] = min(max(arg, 0), 12)
    elif kind == Kind.PLACE:
        rel, z = arg >> 4, arg & 15
        if z < 8:
            place = (16 if rel else 1) + z  # my/opp mzone (their m1-m7 / om1-om7)
            x[10] = place if place <= (22 if rel else 7) else 0
        else:
            place = (23 if rel else 8) + (z - 8)  # my/opp szone (their s1-s8 / os1-os8)
            x[10] = place if place <= (30 if rel else 15) else 0
    elif kind == Kind.ATTRIBUTE:
        x[11] = ATTRIBUTE_TO_ID.get(arg & 0xFFFFFFFF, 0)
    return x


def encode_actions(
    actions: np.ndarray, n_options: int, index_map: np.ndarray, code_to_id: dict[int, int]
) -> np.ndarray:
    """Our action rows [64, 10] -> their legal actions [24, 12] uint8 (first 24 options)."""
    x = np.zeros((MAX_ACTIONS, N_ACTION_FEATURES), dtype=np.uint8)
    for i in range(min(n_options, MAX_ACTIONS)):
        r = actions[i]
        row = int(r[1])
        spec = int(index_map[row]) if 0 <= row < len(index_map) else 0
        x[i] = encode_option(
            int(r[0]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), spec, code_to_id
        )
    return x


def encode_history(history: np.ndarray, history_mask: np.ndarray, code_to_id: dict[int, int]) -> np.ndarray:
    """Our history [32, 13] (most-recent-first) -> their history actions [32, 14] (oldest-first).

    Their turn column is turns-ago clamped to 16 (features.py:encode_history_actions); the spec
    index cannot be rebuilt (card rows move between decisions), so it is 0 (N/A). Like theirs,
    rows whose mapped msg is 0 carry turn 0.
    """
    x = np.zeros((N_HISTORY_ACTIONS, H_ACTION_FEATURES), dtype=np.uint8)
    valid = [k for k in range(min(len(history), N_HISTORY_ACTIONS)) if history_mask[k]]
    for j, k in enumerate(reversed(valid)):
        r = history[k]
        row = encode_option(int(r[0]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), 0, code_to_id)
        x[j, :12] = row
        x[j, 12] = min(max(int(r[11]), 0), 16) if x[j, 3] != 0 else 0
        x[j, 13] = PHASE_TO_ID.get(int(r[12]), 0)
    return x


def scorable_options(msg: int, kinds: np.ndarray, n_options: int) -> np.ndarray:
    """Which of the first min(n_options, 24) options the model may pick.

    Mirrors ygo-agent's get_legal_actions (features.py:646), whose builders never emit these:
    - SELECT_CARD / SELECT_TRIBUTE / SELECT_UNSELECT_CARD ignore the core's cancelable flag, so
      our CANCEL option has no counterpart in their training distribution;
    - SELECT_UNSELECT_CARD additionally offers only selectable cards (no unselectable-card
      options, which our UNSELECT kind covers).
    Scoring those OOD rows lets a deterministic argmax sit in cancel-reselect loops forever
    (measured: SPSUMMON -> CANCEL -> same IDLECMD, never terminating). Everything else scores.
    If filtering would leave nothing (degenerate), everything scores.
    """
    k = min(n_options, MAX_ACTIONS)
    ok = np.ones(k, bool)
    if msg in (C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_UNSELECT_CARD):
        ok &= np.asarray(kinds[:k]) != int(Kind.CANCEL)
        if msg == C.MSG_SELECT_UNSELECT_CARD:
            ok &= np.asarray(kinds[:k]) != int(Kind.UNSELECT)
    if not ok.any():
        ok[:] = True
    return ok


class YgoAgentPolicy:
    """The 0546 TFLite checkpoint as a gauntlet Policy (``name = "ygoagent0546"``).

    Stateless w.r.t. history (rebuilt from our buffer each call); the LSTM (h, c) is kept per
    (game id, acting seat) and starts at zeros for each new game. One row at a time through the
    interpreter (batch 1); deterministic argmax over the scorable slots (scorable_options) of the
    first min(n_options, 24) unless ``sample=True``.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        code_list_path: str | Path = CODE_LIST_PATH,
        seed: int = 0,
        sample: bool = False,
    ) -> None:
        from ai_edge_litert.interpreter import Interpreter  # lazy: unit tests must not need it

        self.name = "ygoagent0546"
        self.rng = np.random.default_rng(seed)
        self.sample = sample
        self.code_to_id = vintage_code_ids(code_list_path)
        with open(model_path, "rb") as f:
            blob = f.read()
        self._interp = Interpreter(model_content=blob)
        self._interp.allocate_tensors()
        in_details = sorted(self._interp.get_input_details(), key=lambda d: d["index"])
        by_shape: dict[tuple[int, ...], list[dict]] = {}
        for d in in_details:
            by_shape.setdefault(tuple(int(s) for s in d["shape"]), []).append(d)
        hiddens = by_shape.get((1, N_RNN_CHANNELS), [])
        if len(hiddens) < 2 or any(
            tuple(int(s) for s in d["shape"]) != shape
            for d, shape in zip(
                in_details,
                [(1, 512), (1, 512), (1, 24, 12), (1, 160, 41), (1, 23), (1, 32, 14)],
            )
        ):
            raise RuntimeError(f"unexpected 0546 model inputs: {[d['shape'] for d in in_details]}")
        # (h, c) share a shape; index order disambiguates.
        self._in_h, self._in_c = hiddens[0]["index"], hiddens[1]["index"]
        self._in_actions = by_shape[(1, MAX_ACTIONS, N_ACTION_FEATURES)][0]["index"]
        self._in_cards = by_shape[(1, N_CARDS, N_CARD_FEATURES)][0]["index"]
        self._in_global = by_shape[(1, N_GLOBAL_FEATURES)][0]["index"]
        self._in_hist = by_shape[(1, N_HISTORY_ACTIONS, H_ACTION_FEATURES)][0]["index"]
        out_details = sorted(self._interp.get_output_details(), key=lambda d: d["index"])
        by_shape = {}
        for d in out_details:
            by_shape.setdefault(tuple(int(s) for s in d["shape"]), []).append(d)
        hiddens = by_shape.get((1, N_RNN_CHANNELS), [])
        probs = by_shape.get((1, MAX_ACTIONS), [])
        if len(hiddens) < 2 or not probs:
            raise RuntimeError(f"unexpected 0546 model outputs: {[d['shape'] for d in out_details]}")
        self._out_h, self._out_c = hiddens[0]["index"], hiddens[1]["index"]
        self._out_probs = probs[0]["index"]
        self._states: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _state(self, key: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        state = self._states.get(key)
        if state is None:
            state = (
                np.zeros((1, N_RNN_CHANNELS), np.float32),
                np.zeros((1, N_RNN_CHANNELS), np.float32),
            )
            self._states[key] = state
        return state

    def reset_keys(self, game_ids) -> None:
        """Drop LSTM states whose game token is in ``game_ids``.

        RL rollouts run the VecEnv without a schedule, so ``buf["game"]`` is -1 for
        every env and episode: callers there pass their own per-episode tokens via
        ``keys`` and must call this when an episode ends, or states leak across
        episodes (and envs, which share the -1 token).
        """
        drop = {int(g) for g in game_ids}
        for key in [k for k in self._states if k[0] in drop]:
            del self._states[key]

    def __call__(self, buf: dict[str, np.ndarray], rows: np.ndarray,
                 keys: np.ndarray | None = None) -> np.ndarray:
        rows = np.asarray(rows)
        if keys is None:
            # Schedule mode (eval/match play): buf["game"] is the schedule index.
            live = {int(g) for g in np.asarray(buf["game"]).tolist()}
            for key in [k for k in self._states if k[0] not in live]:
                del self._states[key]
            game_of = lambda _n, r: int(buf["game"][r])  # noqa: E731
        else:
            # Schedule-free mode (RL rollouts): buf["game"] is -1 for every row,
            # so the caller passes one game token per row and owns cleanup via
            # reset_keys. No auto-pruning here: tokens of episodes still in
            # flight on other envs are not visible in this batch.
            keys = np.asarray(keys)
            assert len(keys) == len(rows), (len(keys), len(rows))
            game_of = lambda n, _r: int(keys[n])  # noqa: E731
        out = np.empty(len(rows), np.int32)
        for n, r in enumerate(rows):
            r = int(r)
            key = (game_of(n, r), int(buf["player"][r]))
            h, c = self._state(key)
            cards_enc, index_map = encode_cards(buf["cards"][r], self.code_to_id)
            n_options = int(buf["action_mask"][r].sum())
            self._interp.set_tensor(self._in_h, h)
            self._interp.set_tensor(self._in_c, c)
            self._interp.set_tensor(
                self._in_actions,
                np.ascontiguousarray(
                    encode_actions(buf["actions"][r], n_options, index_map, self.code_to_id).reshape(
                        1, MAX_ACTIONS, N_ACTION_FEATURES
                    )
                ),
            )
            self._interp.set_tensor(
                self._in_cards, np.ascontiguousarray(cards_enc.reshape(1, N_CARDS, N_CARD_FEATURES))
            )
            self._interp.set_tensor(
                self._in_global,
                np.ascontiguousarray(encode_global(buf["global"][r]).reshape(1, N_GLOBAL_FEATURES)),
            )
            self._interp.set_tensor(
                self._in_hist,
                np.ascontiguousarray(
                    encode_history(buf["history"][r], buf["history_mask"][r], self.code_to_id).reshape(
                        1, N_HISTORY_ACTIONS, H_ACTION_FEATURES
                    )
                ),
            )
            self._interp.invoke()
            h2 = np.array(self._interp.get_tensor(self._out_h), np.float32)
            c2 = np.array(self._interp.get_tensor(self._out_c), np.float32)
            self._states[key] = (h2, c2)
            k = min(n_options, MAX_ACTIONS)
            p = np.asarray(self._interp.get_tensor(self._out_probs), np.float32)[0, :k]
            ok = scorable_options(int(buf["global"][r][21]), buf["actions"][r][:, 0], n_options)
            if self.sample and np.isfinite(p).all() and float(p[ok].sum()) > 0:
                q = p.copy()
                q[~ok] = 0
                out[n] = int(self.rng.choice(k, p=q / float(q.sum())))
            else:
                q = np.where(ok, p, -np.inf)
                out[n] = int(np.argmax(q)) if k else 0
        return out

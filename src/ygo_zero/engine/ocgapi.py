"""ctypes bindings for the edo9300 ygopro-core C API (`ocgapi.h` / `ocgapi_types.h`, API 11.0)."""

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path

API_VERSION = (11, 0)

OCG_Duel = c_void_p


class OCG_CardData(Structure):
    _fields_ = [
        ("code", c_uint32),
        ("alias", c_uint32),
        ("setcodes", POINTER(c_uint16)),  # 0-terminated; the core copies it before cardReaderDone
        ("type", c_uint32),
        ("level", c_uint32),
        ("attribute", c_uint32),
        ("race", c_uint64),
        ("attack", c_int32),
        ("defense", c_int32),
        ("lscale", c_uint32),
        ("rscale", c_uint32),
        ("link_marker", c_uint32),
    ]


class OCG_Player(Structure):
    _fields_ = [("startingLP", c_uint32), ("startingDrawCount", c_uint32), ("drawCountPerTurn", c_uint32)]


OCG_DataReader = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(OCG_CardData))
OCG_DataReaderDone = CFUNCTYPE(None, c_void_p, POINTER(OCG_CardData))
OCG_ScriptReader = CFUNCTYPE(c_int, c_void_p, c_void_p, c_char_p)
OCG_LogHandler = CFUNCTYPE(None, c_void_p, c_char_p, c_int)


class OCG_DuelOptions(Structure):
    _fields_ = [
        ("seed", c_uint64 * 4),  # xoshiro256**; must not be all zero
        ("flags", c_uint64),
        ("team1", OCG_Player),
        ("team2", OCG_Player),
        ("cardReader", OCG_DataReader),
        ("payload1", c_void_p),
        ("scriptReader", OCG_ScriptReader),
        ("payload2", c_void_p),
        ("logHandler", OCG_LogHandler),
        ("payload3", c_void_p),
        ("cardReaderDone", OCG_DataReaderDone),
        ("payload4", c_void_p),
        ("enableUnsafeLibraries", c_uint8),
    ]


class OCG_NewCardInfo(Structure):
    _fields_ = [
        ("team", c_uint8),
        ("duelist", c_uint8),
        ("code", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("pos", c_uint32),
    ]


class OCG_QueryInfo(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("overlay_seq", c_uint32),
    ]


CREATION_STATUS = [
    "SUCCESS",
    "NO_OUTPUT",
    "NOT_CREATED",
    "NULL_DATA_READER",
    "NULL_SCRIPT_READER",
    "INCOMPATIBLE_LUA_API",
    "NULL_RNG_SEED",
]

DUEL_STATUS_END, DUEL_STATUS_AWAITING, DUEL_STATUS_CONTINUE = 0, 1, 2


def load_library(path: str | Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    sigs = {
        "OCG_GetVersion": (None, [POINTER(c_int), POINTER(c_int)]),
        "OCG_CreateDuel": (c_int, [POINTER(OCG_Duel), POINTER(OCG_DuelOptions)]),
        "OCG_DestroyDuel": (None, [OCG_Duel]),
        "OCG_DuelNewCard": (None, [OCG_Duel, POINTER(OCG_NewCardInfo)]),
        "OCG_StartDuel": (None, [OCG_Duel]),
        "OCG_DuelProcess": (c_int, [OCG_Duel]),
        "OCG_DuelGetMessage": (c_void_p, [OCG_Duel, POINTER(c_uint32)]),
        "OCG_DuelSetResponse": (None, [OCG_Duel, c_void_p, c_uint32]),
        "OCG_LoadScript": (c_int, [OCG_Duel, c_char_p, c_uint32, c_char_p]),
        "OCG_DuelQueryCount": (c_uint32, [OCG_Duel, c_uint8, c_uint32]),
        "OCG_DuelQuery": (c_void_p, [OCG_Duel, POINTER(c_uint32), POINTER(OCG_QueryInfo)]),
        "OCG_DuelQueryLocation": (c_void_p, [OCG_Duel, POINTER(c_uint32), POINTER(OCG_QueryInfo)]),
        "OCG_DuelQueryField": (c_void_p, [OCG_Duel, POINTER(c_uint32)]),
    }
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    major, minor = c_int(), c_int()
    lib.OCG_GetVersion(ctypes.byref(major), ctypes.byref(minor))
    if (major.value, minor.value) != API_VERSION:
        raise RuntimeError(f"ocgcore API {major.value}.{minor.value}, bindings expect {API_VERSION}")
    return lib

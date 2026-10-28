//! A C-ABI shim (`ocgcore_rs`) over the Rust port (`ygo-core` + `ygo-cards`).
//!
//! It exports the OCG C API our C++ env (`cpp/ygoenv`) talks to (see
//! `third_party/ygopro-core/ocgapi.h`), plus one new function `OCG_DuelClone`.
//! Card data comes from one process-wide [`CardDb`] (`$YGO_RS_CDB_DIR`, default
//! `<repo>/third_party/BabelCDB`); scripts are compiled into the port, so
//! `OCG_LoadScript` is a no-op returning success.
//!
//! Every entry point catches panics: the error reaches the duel's `logHandler`
//! as `OCG_LOG_TYPE_ERROR`, and that duel's `OCG_DuelProcess` returns END
//! afterwards. Never unwinds into C++.

use std::ffi::{CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ygo_core::api::{NewCardInfo, QueryInfo};
use ygo_core::carddb::CardDb;
use ygo_core::duel::{Duel, DuelOptions, PlayerOptions};
use ygo_core::processor1::Status;
use ygo_core::script::Scripts;

#[allow(non_camel_case_types)] // the name is the OCG C API's
pub type OCG_Duel = *mut c_void;

/// `OCG_Player` (`third_party/ygopro-core/ocgapi_types.h`): field order and types as in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OCG_Player {
    pub starting_lp: u32,
    pub starting_draw_count: u32,
    pub draw_count_per_turn: u32,
}

/// `OCG_DuelOptions`: function pointers are opaque here (`cardReader`/`scriptReader`
/// are ignored: card data comes from the global [`CardDb`], scripts are built in).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OCG_DuelOptions {
    pub seed: [u64; 4],
    pub flags: u64,
    pub team1: OCG_Player,
    pub team2: OCG_Player,
    pub card_reader: *const c_void,
    pub payload1: *mut c_void,
    pub script_reader: *const c_void,
    pub payload2: *mut c_void,
    pub log_handler: Option<unsafe extern "C" fn(*mut c_void, *const c_char, c_int)>,
    pub payload3: *mut c_void,
    pub card_reader_done: *const c_void,
    pub payload4: *mut c_void,
    pub enable_unsafe_libraries: u8,
}

/// `OCG_NewCardInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OCG_NewCardInfo {
    pub team: u8,
    pub duelist: u8,
    pub code: u32,
    pub con: u8,
    pub loc: u32,
    pub seq: u32,
    pub pos: u32,
}

/// `OCG_QueryInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OCG_QueryInfo {
    pub flags: u32,
    pub con: u8,
    pub loc: u32,
    pub seq: u32,
    pub overlay_seq: u32,
}

fn cdb_dir() -> PathBuf {
    if let Ok(d) = std::env::var("YGO_RS_CDB_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/BabelCDB")
}

fn global_db() -> Arc<CardDb> {
    static CELL: OnceLock<Arc<CardDb>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(CardDb::load(&cdb_dir()).expect("ocgshim: cannot load card DB")))
        .clone()
}

fn global_scripts() -> Arc<Scripts> {
    static CELL: OnceLock<Arc<Scripts>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(ygo_cards::scripts())).clone()
}

struct DuelHandle {
    duel: Duel,
    log_handler: Option<unsafe extern "C" fn(*mut c_void, *const c_char, c_int)>,
    payload: *mut c_void,
    failed: bool,
    msg_buf: Vec<u8>,
    query_buf: Vec<u8>,
}

fn report(handle: &DuelHandle, msg: &str, typ: c_int) {
    if let Some(f) = handle.log_handler {
        if let Ok(c) = CString::new(msg) {
            // The handler is synchronous: `c` outlives the call.
            unsafe { f(handle.payload, c.as_ptr(), typ) };
        }
    }
}

/// Pass the port's queued log messages to the stored `logHandler` (script errors
/// have type 0 = `OCG_LOG_TYPE_ERROR`).
fn drain_logs(handle: &mut DuelHandle) {
    for (typ, msg) in std::mem::take(&mut handle.duel.logs) {
        report(handle, &msg, typ as c_int);
    }
}

fn fail(handle: &mut DuelHandle, msg: &str) {
    handle.failed = true;
    report(handle, msg, 0);
}

unsafe fn handle_of(duel: OCG_Duel) -> Option<&'static mut DuelHandle> {
    if duel.is_null() {
        return None;
    }
    // SAFETY: every non-null OCG_Duel came from Box::into_raw below and is
    // destroyed exactly once by OCG_DestroyDuel; callers never touch a duel
    // from two threads at once (same contract as the C++ core).
    unsafe { (duel as *mut DuelHandle).as_mut() }
}

/// Run `f` on the handle; on a panic mark the duel failed and report it.
/// Returns `dflt` on a null duel or a panic.
fn with_handle<R>(duel: OCG_Duel, dflt: R, f: impl FnOnce(&mut DuelHandle) -> R) -> R {
    let h = match unsafe { handle_of(duel) } {
        Some(h) => h,
        None => return dflt,
    };
    match catch_unwind(AssertUnwindSafe(|| f(h))) {
        Ok(r) => r,
        Err(_) => {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                if let Some(h) = unsafe { handle_of(duel) } {
                    fail(h, "ocgshim: engine panic");
                }
            }));
            dflt
        }
    }
}

fn player_of(p: &OCG_Player) -> PlayerOptions {
    PlayerOptions {
        starting_lp: p.starting_lp,
        starting_draw_count: p.starting_draw_count,
        draw_count_per_turn: p.draw_count_per_turn,
    }
}

unsafe fn read_new_card(info: *const OCG_NewCardInfo) -> Option<NewCardInfo> {
    if info.is_null() {
        return None;
    }
    let i = unsafe { &*info };
    Some(NewCardInfo {
        team: i.team,
        duelist: i.duelist,
        code: i.code,
        con: i.con,
        loc: i.loc,
        seq: i.seq,
        pos: i.pos,
    })
}

unsafe fn read_query(info: *const OCG_QueryInfo) -> Option<QueryInfo> {
    if info.is_null() {
        return None;
    }
    let i = unsafe { &*info };
    Some(QueryInfo {
        flags: i.flags,
        con: i.con,
        loc: i.loc,
        seq: i.seq,
        overlay_seq: i.overlay_seq,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_GetVersion(major: *mut c_int, minor: *mut c_int) {
    let _ = catch_unwind(|| {
        // Same version as the C++ core this API mirrors (ocgapi_types.h).
        if !major.is_null() {
            unsafe { *major = 11 };
        }
        if !minor.is_null() {
            unsafe { *minor = 0 };
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_CreateDuel(out: *mut OCG_Duel, options: *const OCG_DuelOptions) -> c_int {
    if out.is_null() || options.is_null() {
        return 2; // OCG_DUEL_CREATION_NOT_CREATED
    }
    let opts = unsafe { &*options };
    let (seed, flags, t1, t2, log_handler, payload) =
        (opts.seed, opts.flags, opts.team1, opts.team2, opts.log_handler, opts.payload3);
    match catch_unwind(AssertUnwindSafe(|| {
        let options = DuelOptions {
            seed,
            flags,
            team1: player_of(&t1),
            team2: player_of(&t2),
        };
        let mut duel = Duel::new(&options, global_db(), global_scripts());
        ygo_cards::load_global_scripts(&mut duel);
        let mut h = DuelHandle {
            duel,
            log_handler,
            payload,
            failed: false,
            msg_buf: Vec::new(),
            query_buf: Vec::new(),
        };
        drain_logs(&mut h);
        unsafe { *out = Box::into_raw(Box::new(h)) as OCG_Duel };
        0 // OCG_DUEL_CREATION_SUCCESS
    })) {
        Ok(r) => r,
        Err(_) => 2,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DestroyDuel(duel: OCG_Duel) {
    if duel.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(duel as *mut DuelHandle)) };
    }));
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelNewCard(duel: OCG_Duel, info: *const OCG_NewCardInfo) {
    with_handle(duel, (), |h| {
        if let Some(i) = unsafe { read_new_card(info) } {
            h.duel.ocg_new_card(&i);
            drain_logs(h);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_StartDuel(duel: OCG_Duel) {
    with_handle(duel, (), |h| {
        h.duel.ocg_start_duel();
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelProcess(duel: OCG_Duel) -> c_int {
    with_handle(duel, 0, |h| {
        if h.failed {
            return 0; // OCG_DUEL_STATUS_END after a failure
        }
        let status = h.duel.ocg_process();
        drain_logs(h);
        match status {
            Status::End => 0,
            Status::Awaiting => 1,
            Status::Continue => 2,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelGetMessage(duel: OCG_Duel, length: *mut u32) -> *const c_void {
    with_handle(duel, std::ptr::null(), |h| {
        h.msg_buf = h.duel.ocg_get_message().to_vec();
        if !length.is_null() {
            unsafe { *length = h.msg_buf.len() as u32 };
        }
        h.msg_buf.as_ptr() as *const c_void
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelSetResponse(duel: OCG_Duel, buffer: *const c_void, length: u32) {
    with_handle(duel, (), |h| {
        if buffer.is_null() {
            h.duel.ocg_set_response(&[]);
            return;
        }
        let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, length as usize) };
        h.duel.ocg_set_response(bytes);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_LoadScript(
    _duel: OCG_Duel,
    _buffer: *const c_char,
    _length: u32,
    _name: *const c_char,
) -> c_int {
    // The port compiles its scripts in: nothing to load.
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelQueryCount(duel: OCG_Duel, team: u8, loc: u32) -> u32 {
    with_handle(duel, 0, |h| h.duel.ocg_query_count(team, loc))
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelQuery(
    duel: OCG_Duel,
    length: *mut u32,
    info: *const OCG_QueryInfo,
) -> *const c_void {
    with_handle(duel, std::ptr::null(), |h| {
        match unsafe { read_query(info) } {
            Some(qi) => h.query_buf = h.duel.ocg_query(&qi),
            None => h.query_buf.clear(),
        }
        if !length.is_null() {
            unsafe { *length = h.query_buf.len() as u32 };
        }
        h.query_buf.as_ptr() as *const c_void
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelQueryLocation(
    duel: OCG_Duel,
    length: *mut u32,
    info: *const OCG_QueryInfo,
) -> *const c_void {
    with_handle(duel, std::ptr::null(), |h| {
        match unsafe { read_query(info) } {
            Some(qi) => h.query_buf = h.duel.ocg_query_location(&qi),
            None => h.query_buf.clear(),
        }
        if !length.is_null() {
            unsafe { *length = h.query_buf.len() as u32 };
        }
        h.query_buf.as_ptr() as *const c_void
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelQueryField(duel: OCG_Duel, length: *mut u32) -> *const c_void {
    with_handle(duel, std::ptr::null(), |h| {
        h.query_buf = h.duel.ocg_query_field();
        if !length.is_null() {
            unsafe { *length = h.query_buf.len() as u32 };
        }
        h.query_buf.as_ptr() as *const c_void
    })
}

/// A new, independent handle holding `duel.clone()` (destroy each with
/// `OCG_DestroyDuel`). `payload3` is the clone's own `logHandler` payload: the
/// original's points at the original C++ `Duel`, which may die first.
/// Declared for C++ in `cpp/ygoenv/ocg_clone.h`.
#[unsafe(no_mangle)]
pub extern "C" fn OCG_DuelClone(duel: OCG_Duel, payload3: *mut c_void) -> OCG_Duel {
    with_handle(duel, std::ptr::null_mut(), |h| {
        let h2 = DuelHandle {
            duel: h.duel.clone(),
            log_handler: h.log_handler,
            payload: payload3,
            failed: h.failed,
            msg_buf: Vec::new(),
            query_buf: Vec::new(),
        };
        Box::into_raw(Box::new(h2)) as OCG_Duel
    })
}

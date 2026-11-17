//! Builds, for the cargo target (laptop, Android arm64/x86_64, ...):
//! - Lua 5.4 as C++, each file prefixed with EDOPro's `luaconf-customize.h` (as `third_party/ygopro-core/lua/meson.build`
//!   does with `cat`), from the same 26 sources;
//! - ygopro-core (the 16 sources of its meson.build, C++17, no RTTI), statically;
//! - our C++ env (`cpp/ygoenv`: engine, decision, env, vecenv) and its C ABI (`capi.cpp`), C++20.
//! The core's tree must already carry our patches (`scripts/build_core.sh` applies them); the build fails loudly if not.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LUA: &[&str] = &[
    "lapi", "lauxlib", "lbaselib", "lcode", "lctype", "ldebug", "ldo", "ldump", "lfunc", "lgc", "liolib", "llex",
    "lmathlib", "lmem", "lobject", "lopcodes", "lparser", "lstate", "lstring", "lstrlib", "ltable", "ltablib", "ltm",
    "lundump", "lvm", "lzio",
];
const CORE: &[&str] = &[
    "card", "duel", "effect", "field", "interpreter", "libcard", "libdebug", "libduel", "libeffect", "libgroup",
    "ocgapi", "operations", "playerop", "processor", "processor_visit", "scriptlib",
];
const ENV: &[&str] = &["engine", "decision", "env", "vecenv", "capi"];

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..").canonicalize().unwrap();
    let core = root.join("third_party/ygopro-core");
    let lua_src = core.join("lua/src");
    let envdir = root.join("cpp/ygoenv");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let android = target.contains("android");

    let constants = fs::read_to_string(core.join("ocgapi_constants.h")).expect("third_party/ygopro-core missing: run git submodule update --init --recursive");
    assert!(
        constants.contains("QUERY_CARDID"),
        "ygopro-core is not patched: run scripts/patch_core.sh (it applies third_party/patches)"
    );

    // Lua: prepend the customisation header and compile as C++, like EDOPro.
    let prefix = fs::read_to_string(core.join("lua/luaconf-customize.h")).unwrap();
    let lua_gen = out.join("lua");
    fs::create_dir_all(&lua_gen).unwrap();
    let mut lua = cc::Build::new();
    lua.cpp(true).include(&lua_src).warnings(false).flag_if_supported("-w");
    for name in LUA {
        let src = fs::read_to_string(lua_src.join(format!("{name}.c"))).unwrap();
        let path = lua_gen.join(format!("{name}.cpp"));
        write_if_changed(&path, &format!("{prefix}{src}"));
        lua.file(path);
    }
    lua.compile("lua");

    let mut ocg = cc::Build::new();
    ocg.cpp(true).std("c++17").include(&lua_src).include(&core).warnings(false).flag("-fno-rtti");
    for name in CORE {
        ocg.file(core.join(format!("{name}.cpp")));
    }
    ocg.compile("ocgcore");

    let mut ygo = cc::Build::new();
    ygo.cpp(true).std("c++20").include(&envdir).flag("-isystem").flag(core.to_str().unwrap()).flag("-isystem")
        .flag(lua_src.to_str().unwrap()).warnings(false);
    for name in ENV {
        ygo.file(envdir.join(format!("{name}.cpp")));
    }
    if android {
        ygo.cpp_link_stdlib(None); // linked below, statically
    }
    ygo.compile("ygoenv");

    // Link order: env -> core -> lua (static archives resolve left to right).
    println!("cargo:rustc-link-lib=static=ygoenv");
    println!("cargo:rustc-link-lib=static=ocgcore");
    println!("cargo:rustc-link-lib=static=lua");
    if android {
        // The NDK's static C++ runtime sits in the sysroot next to the compiler scripts/app/cargo_android.sh picks.
        let cxx = env::var(format!("CXX_{}", target.replace('-', "_"))).expect("CXX_<target> (scripts/app/cargo_android.sh)");
        let sysroot = Path::new(&cxx).parent().unwrap().join("../sysroot/usr/lib").join(&target);
        // Copy only these two archives: that sysroot dir also holds a static libc.a (bundling a Rust std in NDK 30),
        // which a link-search on it would pull in instead of the API-level libc.so.
        let rt = out.join("cxxrt");
        fs::create_dir_all(&rt).unwrap();
        for lib in ["libc++_static.a", "libc++abi.a"] {
            fs::copy(sysroot.join(lib), rt.join(lib)).unwrap_or_else(|e| panic!("{}: {e}", sysroot.join(lib).display()));
        }
        println!("cargo:rustc-link-search=native={}", rt.display());
        println!("cargo:rustc-link-lib=static=c++_static");
        println!("cargo:rustc-link-lib=static=c++abi");
    }
    for d in [&envdir, &core, &core.join("lua")] {
        println!("cargo:rerun-if-changed={}", d.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn write_if_changed(path: &Path, text: &str) {
    if fs::read_to_string(path).ok().as_deref() != Some(text) {
        fs::write(path, text).unwrap();
    }
}

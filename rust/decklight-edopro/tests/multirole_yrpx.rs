//! A `.yrpX` made by a real Multirole (A2b-2 gate 4, room creator seated second) imports exactly: every main-deck
//! card added at sequence 0 (`deck_add: "edopro"`), team 1's cards first, no select messages recorded. Same file
//! as `tests/unit/test_yrp.py::test_real_multirole_replay_imports` (NOTEBOOK 2026-11-01 "A2b-2").

use std::path::{Path, PathBuf};

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards};
use decklight_edopro::{record, yrp};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn real_multirole_replay_imports() {
    let root = repo_root();
    let cards = load_cards(&default_cdbs(&root.join("third_party/BabelCDB"), &[])).unwrap();
    let engine = build_engine(&cards, &index_scripts(&root.join("third_party/CardScripts"), &[]).unwrap());
    let data = std::fs::read(root.join("tests/fixtures/yrpx/multirole-order10.yrpx")).unwrap();
    let x = yrp::parse_yrpx(&data).unwrap();
    assert_eq!(yrp::recorded(&x.messages).len(), x.messages.len(), "Multirole records no select messages");
    let (rec, problems) = yrp::to_record(&x, &engine, &root, None, "t", None, None);
    assert!(problems.is_empty(), "{problems:?}");
    let rec = rec.unwrap();
    assert_eq!(rec["deck_add"], "edopro");
    assert_eq!(rec["add_order"], serde_json::json!([1, 0]));
    assert_eq!(rec["result"]["turns"], 35);
    assert!(record::check_record(&engine, &rec).is_empty());
}

#[test]
fn edopro_main_is_its_own_inverse() {
    for n in 0..5u32 {
        let m: Vec<u32> = (10..10 + n).collect();
        assert_eq!(yrp::edopro_main(&yrp::edopro_main(&m)), m);
    }
    assert_eq!(yrp::edopro_main(&[1, 2, 3]), vec![2, 1, 3]);
}

// pybind11 module `ygo_zero._ygoenv`. Thin: the Python wrapper is src/ygo_zero/engine/cpp_env.py.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ygoenv.h"

namespace py = pybind11;
using namespace ygo;

namespace {

template <typename T, size_t N>
py::array_t<T> arr(const std::array<T, N>& a, std::vector<py::ssize_t> shape) {
  py::array_t<T> out(shape);
  std::memcpy(out.mutable_data(), a.data(), N * sizeof(T));
  return out;
}

py::object to_dict(const Obs& o) {
  py::dict d;
  d["cards"] = arr(o.cards, {MAX_CARDS, CARD_COLS});
  d["card_mask"] = arr(o.card_mask, {MAX_CARDS}).attr("astype")("bool");
  d["global"] = arr(o.global, {GLOBAL_COLS});
  d["chain"] = arr(o.chain, {MAX_CHAIN, CHAIN_COLS});
  d["chain_mask"] = arr(o.chain_mask, {MAX_CHAIN}).attr("astype")("bool");
  d["actions"] = arr(o.actions, {MAX_OPTIONS, ACTION_COLS});
  d["action_mask"] = arr(o.action_mask, {MAX_OPTIONS}).attr("astype")("bool");
  d["history"] = arr(o.history, {HISTORY, HISTORY_COLS});
  d["history_mask"] = arr(o.history_mask, {HISTORY}).attr("astype")("bool");
  return d;
}

// Zero-copy view of one field across all envs: shape (N, *shape), rows strided by sizeof(Obs).
template <typename T>
py::array view(py::object owner, std::vector<Obs>& obs, const T* first, std::vector<py::ssize_t> shape, const char* dtype) {
  std::vector<py::ssize_t> full{static_cast<py::ssize_t>(obs.size())}, strides{sizeof(Obs)};
  py::ssize_t inner = sizeof(T);
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    strides.insert(strides.begin() + 1, inner);
    inner *= *it;
  }
  full.insert(full.end(), shape.begin(), shape.end());
  return py::array(py::dtype(dtype), full, strides, first, owner);
}

}  // namespace

PYBIND11_MODULE(_ygoenv, m) {
  m.attr("OBS_VERSION") = OBS_VERSION;

  py::class_<Engine>(m, "Engine")
      .def(py::init<>())
      .def("add_card",
           [](Engine& e, uint32_t code, uint32_t alias, std::vector<uint16_t> setcodes, uint32_t type, uint32_t level,
              uint32_t attribute, uint64_t race, int32_t attack, int32_t defense, uint32_t lscale, uint32_t rscale,
              uint32_t link_marker) {
             e.add_card({code, alias, std::move(setcodes), type, level, attribute, race, attack, defense, lscale,
                         rscale, link_marker});
           })
      .def("set_scripts", [](Engine& e, std::unordered_map<std::string, std::string> paths) { e.script_paths = std::move(paths); })
      .def("finalize", &Engine::finalize)
      .def_readwrite("precompile", &Engine::precompile)
      .def_property_readonly("num_cards", [](Engine& e) { return e.cards.size(); });

  py::class_<DuelConfig>(m, "DuelConfig")
      .def(py::init([](std::vector<uint32_t> main0, std::vector<uint32_t> extra0, std::vector<uint32_t> main1,
                       std::vector<uint32_t> extra1, uint64_t seed, uint64_t flags, uint32_t lp, uint32_t hand,
                       uint32_t draw, std::vector<std::tuple<uint8_t, uint32_t, uint32_t, uint32_t, uint32_t>> setup) {
             DuelConfig c{std::move(main0), std::move(extra0), std::move(main1), std::move(extra1), seed, flags, lp, hand, draw, {}};
             for (auto& [team, code, loc, seq, pos] : setup) c.setup.push_back({team, code, loc, seq, pos});
             return c;
           }),
           py::arg("main0"), py::arg("extra0"), py::arg("main1"), py::arg("extra1"), py::arg("seed"), py::arg("flags"),
           py::arg("lp"), py::arg("hand"), py::arg("draw"), py::arg("setup") = std::vector<std::tuple<uint8_t, uint32_t, uint32_t, uint32_t, uint32_t>>{});

  py::class_<Env>(m, "Env")
      .def(py::init<Engine&, DuelConfig, bool>(), py::keep_alive<1, 2>(), py::arg("engine"), py::arg("config"),
           py::arg("auto_single") = true)
      .def("reset",
           [](Env& env) -> py::object {
             Obs obs;
             bool ok;
             {
               py::gil_scoped_release nogil;
               ok = env.reset(obs);
             }
             return ok ? to_dict(obs) : py::none();
           })
      .def("step",
           [](Env& env, int index) -> py::object {
             Obs obs;
             bool ok;
             {
               py::gil_scoped_release nogil;
               ok = env.step(index, obs);
             }
             return ok ? to_dict(obs) : py::none();
           })
      .def_property_readonly("done", &Env::done)
      .def_property_readonly("winner", [](const Env& e) -> py::object {
        if (e.winner() < 0) return py::none();
        return py::int_(e.winner());
      })
      .def_property_readonly("player", &Env::player)
      .def_property_readonly("turn", [](const Env& e) { return e.duel().turn; })
      .def_property_readonly("errors", [](const Env& e) { return e.duel().errors; })
      .def_readonly("choices", &Env::choices)
      .def_readonly("decisions", &Env::decisions)
      .def_readonly("truncated", &Env::truncated);

  py::class_<Episode>(m, "Episode")
      .def_readonly("deck0", &Episode::deck0)
      .def_readonly("deck1", &Episode::deck1)
      .def_readonly("seed", &Episode::seed)
      .def_readonly("index", &Episode::index)
      .def_readonly("winner", &Episode::winner)
      .def_readonly("turns", &Episode::turns)
      .def_readonly("choices", &Episode::choices);

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init([](Engine& engine, std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> decks,
                       int num_envs, int num_threads, uint64_t seed, bool auto_single) {
             std::vector<Deck> ds;
             for (auto& [main, extra] : decks) ds.push_back({main, extra});
             return std::make_unique<VecEnv>(engine, std::move(ds), num_envs, num_threads, seed, auto_single);
           }),
           py::keep_alive<1, 2>(), py::arg("engine"), py::arg("decks"), py::arg("num_envs"), py::arg("num_threads"),
           py::arg("seed"), py::arg("auto_single") = true)
      .def("reset", &VecEnv::reset, py::call_guard<py::gil_scoped_release>())
      .def("step",
           [](VecEnv& v, py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
             if (actions.size() != v.num_envs) throw std::invalid_argument("actions must have num_envs entries");
             const int32_t* a = actions.data();
             py::gil_scoped_release nogil;
             v.step(a);
           })
      .def("pop_finished", &VecEnv::pop_finished)
      .def("set_schedule",
           [](VecEnv& v, const std::vector<std::tuple<int, int, uint64_t>>& games) {
             std::vector<Episode> eps;
             for (auto& [d0, d1, seed] : games) eps.push_back(Episode{d0, d1, seed, -1, -1, 0, {}});
             v.set_schedule(std::move(eps));
           })
      .def_property_readonly("remaining", &VecEnv::remaining)
      .def_readonly("num_envs", &VecEnv::num_envs)
      .def("buffers", [](py::object self) {
        auto& v = self.cast<VecEnv&>();
        auto& o = v.obs;
        py::dict d;
        d["cards"] = view(self, o, o[0].cards.data(), {MAX_CARDS, CARD_COLS}, "int32");
        d["card_mask"] = view(self, o, o[0].card_mask.data(), {MAX_CARDS}, "bool");
        d["global"] = view(self, o, o[0].global.data(), {GLOBAL_COLS}, "int32");
        d["chain"] = view(self, o, o[0].chain.data(), {MAX_CHAIN, CHAIN_COLS}, "int32");
        d["chain_mask"] = view(self, o, o[0].chain_mask.data(), {MAX_CHAIN}, "bool");
        d["actions"] = view(self, o, o[0].actions.data(), {MAX_OPTIONS, ACTION_COLS}, "int32");
        d["action_mask"] = view(self, o, o[0].action_mask.data(), {MAX_OPTIONS}, "bool");
        d["history"] = view(self, o, o[0].history.data(), {HISTORY, HISTORY_COLS}, "int32");
        d["history_mask"] = view(self, o, o[0].history_mask.data(), {HISTORY}, "bool");
        d["player"] = py::array(py::dtype("int32"), {v.num_envs}, {4}, v.player.data(), self);
        d["winner"] = py::array(py::dtype("int32"), {v.num_envs}, {4}, v.winner.data(), self);
        d["done"] = py::array(py::dtype("bool"), {v.num_envs}, {1}, v.done.data(), self);
        d["game"] = py::array(py::dtype("int64"), {v.num_envs}, {8}, v.game.data(), self);
        return d;
      });
}

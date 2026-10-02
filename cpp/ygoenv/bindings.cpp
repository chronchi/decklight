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
      .def_property_readonly("num_cards", [](Engine& e) { return e.cards.size(); });

  py::class_<DuelConfig>(m, "DuelConfig")
      .def(py::init([](std::vector<uint32_t> main0, std::vector<uint32_t> extra0, std::vector<uint32_t> main1,
                       std::vector<uint32_t> extra1, uint64_t seed, uint64_t flags, uint32_t lp, uint32_t hand,
                       uint32_t draw) {
             return DuelConfig{std::move(main0), std::move(extra0), std::move(main1), std::move(extra1), seed, flags, lp, hand, draw};
           }),
           py::arg("main0"), py::arg("extra0"), py::arg("main1"), py::arg("extra1"), py::arg("seed"), py::arg("flags"),
           py::arg("lp"), py::arg("hand"), py::arg("draw"));

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
}

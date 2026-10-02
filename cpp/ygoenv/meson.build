# Python extension ygo_zero/_ygoenv, linked against the core built by scripts/build_core.sh.
project('ygoenv', 'cpp', default_options : ['cpp_std=c++20', 'buildtype=release', 'b_ndebug=true', 'warning_level=2'])

py = import('python').find_installation(pure : false)
core_src = meson.project_source_root() / '../../third_party/ygopro-core'
core_build = meson.project_source_root() / '../../build/core'
ocgcore = meson.get_compiler('cpp').find_library('ocgcore', dirs : [core_build])
pybind11_inc = run_command(py, '-c', 'import pybind11; print(pybind11.get_include())', check : true).stdout().strip()

py.extension_module('_ygoenv',
  ['engine.cpp', 'decision.cpp', 'env.cpp', 'vecenv.cpp', 'bindings.cpp'],
  cpp_args : ['-isystem', core_src, '-isystem', pybind11_inc],  # both live inside the source tree
  dependencies : [ocgcore, py.dependency(), dependency('threads')],
  install_rpath : core_build,
  build_rpath : core_build,
)

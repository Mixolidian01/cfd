#pragma once
// Portability shim: prefers std::mdspan (C++23 stdlib) when available;
// falls back to the vendored Kokkos P0009 reference implementation (C++20).
//
// After the shim, all mdspan types live in namespace `md`:
//   md::mdspan, md::extents, md::layout_left, md::layout_right,
//   md::layout_stride, md::default_accessor
//
// Source: Kokkos v4.0 reference implementation (Apache-2.0 / LLVM exception)
//         vendored from https://github.com/kokkos/mdspan (include/vendor/).

#if defined(__cpp_lib_mdspan)
#  include <mdspan>
   namespace md = std;
#else
// Force the reference impl into the `Kokkos` namespace.
#  ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#    define MDSPAN_IMPL_STANDARD_NAMESPACE Kokkos
#  endif
#  include "mdspan/mdspan.hpp"
   namespace md = Kokkos;
#endif

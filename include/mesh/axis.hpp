#pragma once
// Layer P — Axis tag (extracted so physics headers can use it without
// pulling in the full operators.hpp / block_tree.hpp hierarchy).
// operators.hpp includes this header and redeclares nothing; callers that
// previously included operators.hpp for Axis continue to work unchanged.

enum class Axis : int { X = 0, Y = 1, Z = 2 };

// Convert face coordinates (n, a, b) to natural cell indices (xi, yi, zi)
// of the left cell. Shared by convective and adjoint RHS loops.
template<Axis DIR>
inline void face_to_ijk(int n, int a, int b,
                         int& xi, int& yi, int& zi) noexcept {
    if constexpr (DIR == Axis::X) { xi = n; yi = a; zi = b; }
    else if constexpr (DIR == Axis::Y) { xi = a; yi = n; zi = b; }
    else                               { xi = a; yi = b; zi = n; }
}

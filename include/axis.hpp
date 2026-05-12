#pragma once
// Layer P — Axis tag (extracted so physics headers can use it without
// pulling in the full operators.hpp / block_tree.hpp hierarchy).
// operators.hpp includes this header and redeclares nothing; callers that
// previously included operators.hpp for Axis continue to work unchanged.

enum class Axis : int { X = 0, Y = 1, Z = 2 };

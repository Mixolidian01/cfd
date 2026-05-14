#include "solver/ns_solver.hpp"
#include <cassert>
#include <stdexcept>

static void test_valid_config_passes() {
    SolverConfig cfg;
    cfg.validate();  // should not throw
}

static void test_bad_cfl_throws() {
    SolverConfig cfg;
    cfg.time.cfl = 1.5;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}

    SolverConfig cfg2;
    cfg2.time.cfl = 0.0;
    try { cfg2.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_max_level_throws() {
    SolverConfig cfg;
    cfg.amr.max_level = -1;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_gamma_throws() {
    SolverConfig cfg;
    cfg.acdi.gamma_a = 0.5;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_lts_ratio_throws() {
    SolverConfig cfg;
    cfg.amr.lts_ratio = 3;  // must be 1, 2, or 4
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

int main() {
    test_valid_config_passes();
    test_bad_cfl_throws();
    test_bad_max_level_throws();
    test_bad_gamma_throws();
    test_bad_lts_ratio_throws();
    return 0;
}

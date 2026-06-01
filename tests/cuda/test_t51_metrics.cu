// Gate t51 — Metrics & Monitoring system (M1-M4 sub-gates added incrementally)
#include "metrics/metrics_bus.hpp"
#include "metrics/field_dumper.hpp"
#include "solver/ns_solver.hpp"
#include <cassert>

int main() {
    // M0: config structs compile and have correct defaults
    SurfaceConfig sc;
    assert(sc.rho_ref == 0.0);
    ProbeConfig pc;
    assert(pc.type == ProbeConfig::Type::POINT);
    assert(pc.n_slabs == 32);
    MetricsConfig mc;
    assert(mc.global_interval == 10);
    assert(mc.residual_interval == 0);

    // CsvWriter smoke-test (writes to /dev/null)
    {
        CsvWriter w("/dev/null", "a,b,c");
        w.append(1, 2.0, "x");
    }
    printf("M0 PASS\n");
    return 0;
}

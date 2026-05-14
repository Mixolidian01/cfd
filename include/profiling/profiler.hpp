#pragma once
// Profiling annotation guards — zero-cost no-ops by default.
//
// Activate NVTX (Nsight Systems/Compute):
//   add -DENABLE_NVTX to compile flags; link nvtx3 (shipped with CUDA Toolkit).
//
// Activate Tracy:
//   add -DENABLE_TRACY and link TracyClient.cpp.
//
// Usage (RAII scope — exits on scope end):
//   PROFILE_SCOPE("label");
//   PROFILE_SCOPE_COLOR("label", 0xFFRRGGBB);

#if defined(ENABLE_NVTX)
#  include <nvtx3/nvToolsExt.h>
   namespace _profiler_detail {
       struct NvtxRange {
           explicit NvtxRange(const char* n) noexcept { nvtxRangePushA(n); }
           ~NvtxRange() noexcept { nvtxRangePop(); }
       };
       struct NvtxRangeColor {
           NvtxRangeColor(const char* n, uint32_t argb) noexcept {
               nvtxEventAttributes_t ea{};
               ea.version     = NVTX_VERSION;
               ea.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
               ea.colorType   = NVTX_COLOR_ARGB;  ea.color = argb;
               ea.messageType = NVTX_MESSAGE_TYPE_ASCII; ea.message.ascii = n;
               nvtxRangePushEx(&ea);
           }
           ~NvtxRangeColor() noexcept { nvtxRangePop(); }
       };
   }
#  define PROFILE_SCOPE(name) \
        ::_profiler_detail::NvtxRange _prof_##__LINE__{name}
#  define PROFILE_SCOPE_COLOR(name, argb) \
        ::_profiler_detail::NvtxRangeColor _prof_##__LINE__{name, argb}

#elif defined(ENABLE_TRACY)
#  include <tracy/Tracy.hpp>
#  define PROFILE_SCOPE(name)             ZoneScopedN(name)
#  define PROFILE_SCOPE_COLOR(name, argb) ZoneScopedNC(name, ((argb) & 0x00FFFFFFu))

#else
#  define PROFILE_SCOPE(name)             ((void)0)
#  define PROFILE_SCOPE_COLOR(name, argb) ((void)0)
#endif

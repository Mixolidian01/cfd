#pragma once
// R9-E1: Wire-format serialisers for 2-D slice and 3-D volume frames.
//
// serialize_frame  / serialize_volume are extracted from LiveStreamer::serialize_*
// because they access only the passed-in FrameBuffer / FrameBuffer3D structs
// (no LiveStreamer member state) and carry all the wire-format knowledge in
// one place.
//
// The push_*() byte-packing helpers are internal to frame_packer.cpp.

#include "io/live_streamer.hpp"   // FrameBuffer, FrameBuffer3D, BlockDesc2D
#include <cstdint>
#include <vector>

// Serialise a 2-D slice frame into a length-prefixed binary blob.
// Wire format: [4-byte LE length][header 32 B][block descs n×16 B][data].
void serialize_frame (const FrameBuffer&   fb, std::vector<uint8_t>& out);

// Serialise a 3-D volume frame into a length-prefixed binary blob.
// Wire format: [4-byte LE length][header 40 B][data (LZ4-compressed if available)].
void serialize_volume(const FrameBuffer3D& fb, std::vector<uint8_t>& out);

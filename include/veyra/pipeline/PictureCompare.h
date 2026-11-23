#pragma once
// Repeated-picture detection with a capture-noise band (2026-10-12) and an
// outlier tolerance (2026-11-02).
//
// Content dedup (ContentDedup.h) has to know whether a sample shows the same
// picture as the last new one. Two earlier answers failed on a real card:
//  - 64x36 sampled points with an 8-level gate (2026-10-11) missed small real
//    motion seen from a still position and dropped real pictures (6 fps).
//  - Byte-identical whole pictures never matched: the MCS T911 at 2560x1440@60
//    NV12 changes ~18% of the pixels of a repeated console frame by 1 level, a
//    few hundred by 2 and at most one by 3, while a new 30 fps game frame seen
//    from a still position has at least 240 pixels beyond 2 levels and peaks at
//    6-16 (tests/claude-cadence-exact-20261012/harness/card_diff.py, 173 pairs).
// A component differs when it is more than 3 levels off on an 8-bit scale
// (bandFor(): the same fraction of a deeper component). Packed layouts whose
// bytes are not components (v210, RGB10, RGB555/565) compare exactly. The
// reference is the last NEW picture, so a slow change accumulates past the band
// instead of slipping through it one small step at a time.
//
// One differing component used to make a new picture. At 3840x2160 NV12 the
// T911 now and then moves a single component of a repeated frame 4 levels, at
// bright HUD text edges: 5 of 2700 repeats in 90 s of a 30 fps PS5 game
// (tests/claude-dedup-precise-20261102). Each became a 16.7 ms frame
// generation pair that VFG High could not fit; the user's session kept up to
// 18 extra samples a second (field log 2026-11-02).
// So a sample repeats the reference while at most toleranceFor() components
// differ, about five per million (63 in 4K NV12). In that session every second
// that kept extra samples had one with at most 47 differing components, and
// the smallest real new picture, a still screen, had 156; a moving 30 fps game
// changes 100 000 and more. A small element that changes on every sample (17 s
// of 60 kept per second whose smallest change was 14 components) no longer
// makes every sample a new picture either.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <intrin.h>
#include <smmintrin.h>
#include <vector>

namespace veyra::pipeline {

struct PicturePlane {
    const uint8_t* data = nullptr;
    size_t pitch = 0, rowBytes = 0;
    unsigned rows = 0;
};
enum class PictureComponents { Bytes, Words, Exact };

class PictureReference {
public:
    static constexpr unsigned kBand8 = 3; // levels on an 8-bit scale
    // The band for components of `depth` bits stored `shift` bits up (P010: 10, 6 -> 768).
    static constexpr unsigned bandFor(unsigned depth, unsigned shift) {
        return depth <= 8 ? kBand8 : ((kBand8 * ((1u << depth) - 1) + 127) / 255) << shift;
    }
    // Components a repeat may have beyond the band: 63 in 3840x2160 NV12, 15 in
    // 1920x1080 NV12, none below 196608 components.
    static constexpr uint64_t toleranceFor(uint64_t components) { return components / 196608; }
    // A sample this far beyond the band is a new picture whatever the rest
    // holds, so the comparison stops there: `over` then saturates at this bound
    // and max8 covers the compared part. Repeats (at most toleranceFor(): 63 in
    // 4K NV12) and the smallest real new picture seen (156) are far below it,
    // so their counts stay exact; a moving game frame reaches it in its first
    // rows instead of after all 12 MB (T911 4K60: 0.9 ms of each new frame's
    // capture callback, 1.8 ms at P95, went to comparing and copying).
    static constexpr uint64_t kDecidedOver = 4096;
    struct Result {
        uint64_t id = 0;       // picture number; equal ids show the same picture
        bool repeat = false;   // at most toleranceFor() components beyond the band of the reference
        uint64_t over = 0;     // components beyond the band (saturates at kDecidedOver for a new picture)
        unsigned max8 = 0;     // largest difference, 8-bit scale
        bool commit = false;   // deferred: commit() must copy this new picture into the reference
    };
    // Ids of different references never collide (space in the top byte).
    explicit PictureReference(unsigned space = 0) : space_(uint64_t(space & 0xFF) << 56) {}
    void reset() { valid_ = false; pending_ = false; }

    // band: per-component tolerance in the stored unit (bandFor(); ignored for Exact).
    // deferCopy: a new picture is numbered at once but becomes the reference only
    // in commit(), which the caller runs with the same planes after it handed the
    // frame on (the capture callback publishes first). Until then the reference
    // is invalid, so a missed commit makes the next sample new, never a repeat.
    Result observe(const PicturePlane* planes, unsigned count, PictureComponents kind, unsigned band = kBand8, bool deferCopy = false) {
        if (kind == PictureComponents::Exact) band = 0;
        Result r;
        size_t total = 0;
        for (unsigned p = 0; p < count; ++p) total += planes[p].rowBytes * planes[p].rows;
        const uint64_t tolerance = toleranceFor(kind == PictureComponents::Words ? total / 2 : total);
        if (valid_ && !pending_ && total == bytes_ && kind == kind_ && band == band_) {
            const uint64_t decided = std::max<uint64_t>(tolerance + 1, kDecidedOver);
            const uint8_t* ref = reference_.data(); unsigned max = 0;
            for (unsigned p = 0; p < count && r.over < decided; ++p)
                for (unsigned y = 0; y < planes[p].rows && r.over < decided; ++y, ref += planes[p].rowBytes) {
                    const uint8_t* row = planes[p].data + size_t(y) * planes[p].pitch;
                    if (kind == PictureComponents::Words) compareWords(row, ref, planes[p].rowBytes, band, r.over, max);
                    else compareBytes(row, ref, planes[p].rowBytes, band, r.over, max);
                }
            r.max8 = band ? (max * kBand8 + band / 2) / band : max;
            if (r.over <= tolerance) { r.id = space_ | id_; r.repeat = true; return r; }
        }
        // A new picture becomes the reference.
        bytes_ = total; kind_ = kind; band_ = band; id_ = (id_ + 1) & 0x00FFFFFFFFFFFFFFull;
        r.id = space_ | id_;
        if (deferCopy) { valid_ = false; pending_ = true; r.commit = true; return r; }
        copyReference(planes, count, total);
        return r;
    }
    // Copies the picture a deferred observe() numbered; no-op otherwise.
    void commit(const PicturePlane* planes, unsigned count) {
        if (!pending_) return;
        pending_ = false;
        size_t total = 0;
        for (unsigned p = 0; p < count; ++p) total += planes[p].rowBytes * planes[p].rows;
        if (total == bytes_) copyReference(planes, count, total);
    }

private:
    void copyReference(const PicturePlane* planes, unsigned count, size_t total) {
        if (reference_.size() < total) reference_.resize(total);
        uint8_t* ref = reference_.data();
        for (unsigned p = 0; p < count; ++p)
            for (unsigned y = 0; y < planes[p].rows; ++y, ref += planes[p].rowBytes)
                std::memcpy(ref, planes[p].data + size_t(y) * planes[p].pitch, planes[p].rowBytes);
        valid_ = true;
    }
    static bool detectAvx2() {
        int info[4]{};
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        __cpuid(info, 1);
        constexpr int kOsxsaveAvx = (1 << 27) | (1 << 28);
        if ((info[2] & kOsxsaveAvx) != kOsxsaveAvx) return false;
        if ((_xgetbv(0) & 0x6ull) != 0x6ull) return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;
    }
    static bool cpuSupportsAvx2() {
        static const bool supported = detectAvx2();
        return supported;
    }
    __declspec(noinline) static void compareBytesAvx2(const uint8_t* a, const uint8_t* b, size_t n, unsigned band, uint64_t& over, unsigned& max) {
        const __m256i limit = _mm256_set1_epi8(static_cast<char>(band)), zero = _mm256_setzero_si256();
        __m256i peak = zero;
        size_t i = 0;
        uint64_t count = 0;
        for (; i + 128 <= n; i += 128) {
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
            const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
            const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
            const __m256i x3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
            const __m256i y3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
            const __m256i d0 = _mm256_or_si256(_mm256_subs_epu8(x0, y0), _mm256_subs_epu8(y0, x0));
            const __m256i d1 = _mm256_or_si256(_mm256_subs_epu8(x1, y1), _mm256_subs_epu8(y1, x1));
            const __m256i d2 = _mm256_or_si256(_mm256_subs_epu8(x2, y2), _mm256_subs_epu8(y2, x2));
            const __m256i d3 = _mm256_or_si256(_mm256_subs_epu8(x3, y3), _mm256_subs_epu8(y3, x3));
            peak = _mm256_max_epu8(peak, _mm256_max_epu8(_mm256_max_epu8(d0, d1), _mm256_max_epu8(d2, d3)));
            const __m256i e0 = _mm256_subs_epu8(d0, limit);
            const __m256i e1 = _mm256_subs_epu8(d1, limit);
            const __m256i e2 = _mm256_subs_epu8(d2, limit);
            const __m256i e3 = _mm256_subs_epu8(d3, limit);
            const __m256i any = _mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3));
            if (!_mm256_testz_si256(any, any)) {
                const uint32_t m0 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(e0, zero)));
                const uint32_t m1 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(e1, zero)));
                const uint32_t m2 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(e2, zero)));
                const uint32_t m3 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(e3, zero)));
                count += uint64_t(__popcnt(m0)) + uint64_t(__popcnt(m1)) + uint64_t(__popcnt(m2)) + uint64_t(__popcnt(m3));
            }
        }
        for (; i + 32 <= n; i += 32) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            const __m256i d = _mm256_or_si256(_mm256_subs_epu8(x, y), _mm256_subs_epu8(y, x));
            peak = _mm256_max_epu8(peak, d);
            const __m256i e = _mm256_subs_epu8(d, limit);
            if (!_mm256_testz_si256(e, e)) {
                const uint32_t m = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(e, zero)));
                count += uint64_t(__popcnt(m));
            }
        }
        const __m128i peak128 = _mm_max_epu8(_mm256_castsi256_si128(peak), _mm256_extracti128_si256(peak, 1));
        alignas(16) uint8_t lanes[16];
        _mm_store_si128(reinterpret_cast<__m128i*>(lanes), peak128);
        for (const uint8_t v : lanes) max = std::max<unsigned>(max, v);
        for (; i < n; ++i) {
            const unsigned d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
            max = std::max(max, d);
            count += (d > band) ? 1u : 0u;
        }
        over += count;
    }
    __declspec(noinline) static void compareWordsAvx2(const uint8_t* a, const uint8_t* b, size_t n, unsigned band, uint64_t& over, unsigned& max) {
        const __m256i limit = _mm256_set1_epi16(static_cast<short>(band)), zero = _mm256_setzero_si256();
        __m256i peak = zero;
        size_t i = 0;
        uint64_t count = 0;
        for (; i + 128 <= n; i += 128) {
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
            const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64));
            const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64));
            const __m256i x3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96));
            const __m256i y3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96));
            const __m256i d0 = _mm256_or_si256(_mm256_subs_epu16(x0, y0), _mm256_subs_epu16(y0, x0));
            const __m256i d1 = _mm256_or_si256(_mm256_subs_epu16(x1, y1), _mm256_subs_epu16(y1, x1));
            const __m256i d2 = _mm256_or_si256(_mm256_subs_epu16(x2, y2), _mm256_subs_epu16(y2, x2));
            const __m256i d3 = _mm256_or_si256(_mm256_subs_epu16(x3, y3), _mm256_subs_epu16(y3, x3));
            peak = _mm256_max_epu16(peak, _mm256_max_epu16(_mm256_max_epu16(d0, d1), _mm256_max_epu16(d2, d3)));
            const __m256i e0 = _mm256_subs_epu16(d0, limit);
            const __m256i e1 = _mm256_subs_epu16(d1, limit);
            const __m256i e2 = _mm256_subs_epu16(d2, limit);
            const __m256i e3 = _mm256_subs_epu16(d3, limit);
            const __m256i any = _mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3));
            if (!_mm256_testz_si256(any, any)) {
                const uint32_t m0 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(e0, zero)));
                const uint32_t m1 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(e1, zero)));
                const uint32_t m2 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(e2, zero)));
                const uint32_t m3 = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(e3, zero)));
                count += uint64_t(__popcnt(m0) + __popcnt(m1) + __popcnt(m2) + __popcnt(m3)) >> 1;
            }
        }
        for (; i + 32 <= n; i += 32) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            const __m256i d = _mm256_or_si256(_mm256_subs_epu16(x, y), _mm256_subs_epu16(y, x));
            peak = _mm256_max_epu16(peak, d);
            const __m256i e = _mm256_subs_epu16(d, limit);
            if (!_mm256_testz_si256(e, e)) {
                const uint32_t m = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(e, zero)));
                count += uint64_t(__popcnt(m)) >> 1;
            }
        }
        const __m128i peak128 = _mm_max_epu16(_mm256_castsi256_si128(peak), _mm256_extracti128_si256(peak, 1));
        alignas(16) uint16_t lanes[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(lanes), peak128);
        for (const uint16_t v : lanes) max = std::max<unsigned>(max, v);
        for (; i + 2 <= n; i += 2) {
            const unsigned x = a[i] | (unsigned(a[i + 1]) << 8), y = b[i] | (unsigned(b[i + 1]) << 8);
            const unsigned d = x > y ? x - y : y - x;
            max = std::max(max, d);
            count += (d > band) ? 1u : 0u;
        }
        over += count;
    }
    static void compareBytes(const uint8_t* a, const uint8_t* b, size_t n, unsigned band, uint64_t& over, unsigned& max) {
        if (cpuSupportsAvx2()) { compareBytesAvx2(a, b, n, band, over, max); return; }
        const __m128i limit = _mm_set1_epi8(char(band)), zero = _mm_setzero_si128();
        __m128i peak = zero; size_t i = 0; uint64_t count = 0;
        for (; i + 16 <= n; i += 16) {
            const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)), y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
            const __m128i d = _mm_or_si128(_mm_subs_epu8(x, y), _mm_subs_epu8(y, x));
            peak = _mm_max_epu8(peak, d);
            count += __popcnt(~unsigned(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_subs_epu8(d, limit), zero))) & 0xFFFFu);
        }
        for (; i < n; ++i) { const unsigned d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]; max = std::max(max, d); count += d > band; }
        alignas(16) uint8_t lanes[16]; _mm_store_si128(reinterpret_cast<__m128i*>(lanes), peak);
        for (const uint8_t v : lanes) max = std::max<unsigned>(max, v);
        over += count;
    }
    // Little-endian 16-bit components (P010/P016 and >8-bit planar frames).
    static void compareWords(const uint8_t* a, const uint8_t* b, size_t n, unsigned band, uint64_t& over, unsigned& max) {
        if (cpuSupportsAvx2()) { compareWordsAvx2(a, b, n, band, over, max); return; }
        const __m128i limit = _mm_set1_epi16(short(band)), zero = _mm_setzero_si128();
        __m128i peak = zero; size_t i = 0; uint64_t count = 0;
        for (; i + 16 <= n; i += 16) {
            const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)), y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
            const __m128i d = _mm_or_si128(_mm_subs_epu16(x, y), _mm_subs_epu16(y, x));
            peak = _mm_max_epu16(peak, d);
            count += __popcnt(~unsigned(_mm_movemask_epi8(_mm_cmpeq_epi16(_mm_subs_epu16(d, limit), zero))) & 0xFFFFu) / 2;
        }
        for (; i + 2 <= n; i += 2) {
            const unsigned x = a[i] | (unsigned(a[i + 1]) << 8), y = b[i] | (unsigned(b[i + 1]) << 8), d = x > y ? x - y : y - x;
            max = std::max(max, d); count += d > band;
        }
        alignas(16) uint16_t lanes[8]; _mm_store_si128(reinterpret_cast<__m128i*>(lanes), peak);
        for (const uint16_t v : lanes) max = std::max<unsigned>(max, v);
        over += count;
    }

    std::vector<uint8_t> reference_;
    size_t bytes_ = 0;
    uint64_t id_ = 0, space_ = 0;
    unsigned band_ = 0;
    bool valid_ = false, pending_ = false;
    PictureComponents kind_ = PictureComponents::Bytes;
};

} // namespace veyra::pipeline

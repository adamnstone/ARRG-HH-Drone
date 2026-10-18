// Laptop harness: run the DSP core over a WAV file (PCM16, 48 kHz, 1 or N_MICS channels;
// a file with more than N_MICS channels uses the first N_MICS, fewer uses channel 0 only) and print one CSV row per frame.
// Build:  g++ -O2 -std=c++11 -o test_core test_core.cpp dsp_core.cpp
// Run:    ./test_core input.wav [active, e.g. 1110 [F: channel 0 at FRONT_POS]] > out.csv
#include "dsp_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include <math.h>

static bool read_wav(const char* path, std::vector<float>& data, int& channels, int& rate) {
    FILE* f = fopen(path, "rb"); if (!f) return false;
    char id[4]; uint32_t sz; fread(id, 1, 4, f); fread(&sz, 4, 1, f); fread(id, 1, 4, f);
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t sr = 0;
    while (fread(id, 1, 4, f) == 4) {
        fread(&sz, 4, 1, f);
        if (!memcmp(id, "fmt ", 4)) {
            fread(&fmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&sr, 4, 1, f);
            uint32_t br; uint16_t ba; fread(&br, 4, 1, f); fread(&ba, 2, 1, f); fread(&bits, 2, 1, f);
            fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            if (bits != 16) { fprintf(stderr, "need 16-bit PCM\n"); return false; }
            size_t n = sz / 2; data.resize(n);
            std::vector<int16_t> buf(n); fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) data[i] = buf[i] / 32768.0f;
            channels = (int)ch; rate = (int)sr; fclose(f); return true;
        } else fseek(f, sz, SEEK_CUR);
    }
    fclose(f); return false;
}

int main(int argc, char** argv) {
    // active: one 0/1 per channel (B, L, R, T), the connected mics (mic_layout at MIC_POS); F: channel 0 at the front
    if (argc < 2) { fprintf(stderr, "usage: %s file.wav [active e.g. 1110 [F]]\n", argv[0]); return 1; }
    std::vector<float> data; int ch = 0, sr = 0;
    if (!read_wav(argv[1], data, ch, sr)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    if (sr != FS) fprintf(stderr, "warning: sample rate %d != %d\n", sr, FS);
    size_t n = data.size() / (size_t)ch;
    if (n < (size_t)NFFT) { fprintf(stderr, "need at least %d samples per channel\n", NFFT); return 1; }
    static dsp::Spectrum S[N_MICS];
    static float frame[NFFT], mono[NBINS];
    dsp::Detector det; dsp::BearingEstimator bear; dsp::FrameEvidence ev; dsp::Bearing b; dsp::ClsWindow win;
    static float lm[NMEL];
    // theta / theta_raw / elev print "nan" when unknown (theta_ok / raw_ok / elev_ok false)
    printf("frame,f0,score,flat,n_harm,hits,detected,strength,theta_raw,theta,elev,conf");
    for (int p = 0; p < N_PAIRS; p++) printf(",tau%d", p);
    printf(",overhead,flat_local,gate,gate_ok,bstat,mused,status,rank,support");
    for (int p = 0; p < N_PAIRS; p++) printf(",coh%d", p);
    printf(",bear_us\n");
    int nframes = 1 + (int)((n - NFFT) / HOP);
    int nch = ch >= N_MICS ? N_MICS : 1;
    bool active[N_MICS] = {true, true, true, true};
    if (argc >= 3 && nch == N_MICS) {
        if (strlen(argv[2]) != N_MICS) { fprintf(stderr, "active: one 0/1 per mic (B, L, R, T)\n"); return 1; }
        for (int c = 0; c < N_MICS; c++) active[c] = argv[2][c] == '1';
        float pos[N_MICS][3];
        memcpy(pos, MIC_POS, sizeof(pos));
        if (argc >= 4 && argv[3][0] == 'F') memcpy(pos[0], FRONT_POS, sizeof(pos[0]));
        dsp::MicLayout lay;
        dsp::mic_layout(active, pos, lay);
        bear.set_layout(lay);
        fprintf(stderr, "layout rank %d level %d azimuth_ok %d elevation_ok %d\n", lay.rank, lay.level, lay.azimuth_ok,
                lay.elevation_ok);
    }
    int n_on = 0;
    for (int c = 0; c < nch; c++) n_on += active[c];
    double bear_total = 0;
    for (int fi = 0; fi < nframes; fi++) {
        for (int c = 0; c < nch; c++) {
            if (!active[c]) continue;
            for (int i = 0; i < NFFT; i++) frame[i] = data[((size_t)fi * HOP + (size_t)i) * (size_t)ch + (size_t)c];
            dsp::spectrum_from_frame(frame, S[c]);
        }
        for (int k = 0; k < NBINS; k++) {                 // mean over the connected channels
            float p = 0;
            for (int c = 0; c < nch; c++) if (active[c]) p += dsp::power_at(S[c], k);
            mono[k] = n_on ? p / n_on : 0.0f;
        }
        dsp::logmel_frame(mono, lm); win.push(lm);
        det.feed(mono, false, ev);
        double us = 0;
        if (nch == N_MICS) {
            auto t0 = std::chrono::steady_clock::now();
            bear.feed(S, ev.mask, ev.detected, b);
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            bear_total += us;
        } else {
            memset(&b, 0, sizeof(b));
            for (int p = 0; p < N_PAIRS; p++) b.taus[p] = NAN;
        }
        printf("%d,%.3f,%.4f,%.5f,%d,%d,%d,%.4f,%.3f,%.3f,%.3f,%.4f", fi, ev.f0, ev.score, ev.flat, ev.n_harm,
               ev.hits, ev.detected ? 1 : 0, ev.strength, b.raw_ok ? b.theta_raw : NAN, b.theta_ok ? b.theta : NAN,
               b.elev_ok ? b.elev : NAN, b.conf);
        for (int p = 0; p < N_PAIRS; p++) printf(",%.7e", b.taus[p]);
        printf(",%d,%.5f,%.4f,%d,%d,%d,%s,%d,%.4f", b.overhead ? 1 : 0, ev.flat_local, ev.gate, ev.gate_ok ? 1 : 0,
               b.bstat, b.mused, dsp::BEARING_STATUS_NAME[b.status], b.rank, b.support);
        for (int p = 0; p < N_PAIRS; p++) printf(",%.4f", b.coh[p]);
        printf(",%.1f\n", us);
    }
    if (nch == N_MICS) fprintf(stderr, "bearing: %.1f us/frame mean over %d frames (host)\n", bear_total / nframes, nframes);
    return 0;
}

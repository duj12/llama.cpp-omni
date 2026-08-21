// Qwen3-Omni HTTP/WebSocket server — turn-based multimodal inference.
//
// Supports:
//   - Text-only mode (--text-only flag, no mmproj needed)
//   - Vision input (JPEG/PNG images via mmproj)
//   - Audio input (float32 PCM via mmproj -> WAV wrapper)
//   - Qwen3 ChatML template with <__media__> markers
//   - Streaming text output via WebSocket /backend protocol
//
// Build:   cmake --build . --target llama-qwen3omni-server
// Run:     ./llama-qwen3omni-server -m model.gguf --mmproj mmproj.gguf

#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"
#include "session.h"
#include "protocol.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <unordered_map>

#define CPPHTTPLIB_WEBSOCKET_MAX_PAYLOAD_LENGTH (128 * 1024 * 1024)
#include "httplib.h"
#include <nlohmann/json.hpp>

// protocol.h defines: using json = nlohmann::ordered_json;
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

// Replace invalid UTF-8 instead of throwing (nlohmann default is strict).
static std::string json_safe_dump(const json & j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// ============================================================================
// Helper: build WAV from float32 PCM (mono 16 kHz)
// ============================================================================

static std::vector<uint8_t> build_wav_from_pcm(const std::vector<float> & pcm) {
    const int n_samples     = (int)pcm.size();
    const int sample_rate   = 16000;
    const int n_channels    = 1;
    const int bits_per_samp = 32;
    const int byte_rate     = sample_rate * n_channels * bits_per_samp / 8;
    const int block_align   = n_channels * bits_per_samp / 8;
    const int data_size     = n_samples * block_align;
    const int file_size     = 36 + data_size;

    std::vector<uint8_t> wav(44 + data_size);
    auto w  = [&](int off, const void * s, int n) { memcpy(&wav[off], s, (size_t)n); };
    auto w4 = [&](int off, uint32_t v) { memcpy(&wav[off], &v, 4); };
    auto w2 = [&](int off, uint16_t v) { memcpy(&wav[off], &v, 2); };

    w(0, "RIFF", 4);  w4(4,  (uint32_t)file_size);
    w(8, "WAVE", 4);
    w(12, "fmt ", 4); w4(16, 16);
    w2(20, 3);                      // IEEE float
    w2(22, (uint16_t)n_channels);
    w4(24, (uint32_t)sample_rate);  w4(28, (uint32_t)byte_rate);
    w2(32, (uint16_t)block_align);  w2(34, (uint16_t)bits_per_samp);
    w(36, "data", 4); w4(40, (uint32_t)data_size);
    memcpy(&wav[44], pcm.data(), (size_t)data_size);
    return wav;
}

// ============================================================================
// Video extraction helper (ffmpeg)
// ============================================================================

static std::string shell_quote(const std::string & value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static bool file_nonempty(const std::string & path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::file_size(path, ec) > 0;
}

struct ExtractedVideoMedia {
    std::string video_path;
    std::string audio_path;
    std::vector<std::string> frame_paths;
};

// Decode base64 MP4 to temp file, extract N JPEG frames + audio WAV via ffmpeg.
static ExtractedVideoMedia extract_video_mp4_media(const std::string & video_b64,
                                                     const std::string & temp_dir,
                                                     int counter,
                                                     int stack_frames) {
    ExtractedVideoMedia out;
    auto raw = b64_decode(video_b64);
    if (raw.empty()) { return out; }

    // Cap at 4 frames (972 tokens each, leaves ~4K for generation).
    fs::path dir = fs::path(temp_dir) / ("video_" + std::to_string(counter));
    fs::create_directories(dir);

    out.video_path = (dir / "input.mp4").string();
    {
        std::ofstream f(out.video_path, std::ios::binary);
        if (!f) { return out; }
        f.write(reinterpret_cast<const char *>(raw.data()), raw.size());
        f.close();
    }
    if (!file_nonempty(out.video_path)) { return out; }

    // Get video duration via ffprobe
    double duration = 30.0;
    {
        std::string probe_cmd = "ffprobe -v error -show_entries format=duration -of csv=p=0 "
            + shell_quote(out.video_path);
        FILE * fp = popen(probe_cmd.c_str(), "r");
        if (fp) {
            char buf[64];
            if (fgets(buf, sizeof(buf), fp) != NULL) {
                duration = std::atof(buf);
            }
            pclose(fp);
        }
    }

    // Extract audio: mono 16kHz PCM WAV
    std::string audio_path = (dir / "audio.wav").string();
    std::string audio_cmd = "ffmpeg -y -hide_banner -loglevel error -i "
        + shell_quote(out.video_path)
        + " -vn -ac 1 -ar 16000 -c:a pcm_f32le "
        + shell_quote(audio_path);
    if (std::system(audio_cmd.c_str()) == 0 && file_nonempty(audio_path)) {
        out.audio_path = audio_path;
    }

    // Extract N frames evenly across the video duration (capped to 8).
    int n_frames = std::min((int)std::ceil(duration), 4);
    double fps_val = (double)n_frames / std::max(duration, 1.0);
    std::string frame_pattern = (dir / "frame_%03d.jpg").string();
    // Convert fps to a fraction string for ffmpeg
    char fps_str[32];
    snprintf(fps_str, sizeof(fps_str), "%.4f", fps_val);
    std::string frame_cmd = "ffmpeg -y -hide_banner -loglevel error -i "
        + shell_quote(out.video_path)
        + " -t " + std::to_string(duration)
        + " -an -vf fps=" + std::string(fps_str) + " -q:v 2 "
        + shell_quote(frame_pattern);
    if (std::system(frame_cmd.c_str()) == 0) {
        for (int i = 1; i <= n_frames; ++i) {
            char name[32];
            snprintf(name, sizeof(name), "frame_%03d.jpg", i);
            std::string frame_path = (dir / name).string();
            if (file_nonempty(frame_path)) {
                out.frame_paths.push_back(frame_path);
            }
        }
    }

    return out;
}

// ============================================================================
// Per-session state (stack-allocated per WebSocket connection)
// ============================================================================

struct Qwen3Session {
    llama_context  * ctx  = nullptr;
    common_sampler * smpl = nullptr;
    std::string sid;
    int resp_cnt = 0;
    int n_ctx = 0;
    int n_keep = 0;
    int n_past = 0;                    // persistent position (streaming / full_duplex)
    bool half_duplex = false;          // VAD+TurnSense worker-driven replies
    bool system_prompt_initialized = false;
    std::shared_ptr<std::atomic<bool>> interrupt = std::make_shared<std::atomic<bool>>(false); // barge-in flag
    ~Qwen3Session() {
        if (smpl) { common_sampler_free(smpl); smpl = nullptr; }
        if (ctx) { llama_free(ctx); ctx = nullptr; }
    }
};

// ============================================================================
// Shared mmproj — loaded once at startup
// ============================================================================

struct SharedMmproj {
    mtmd::context_ptr ctx;

    bool load(const std::string & path, const llama_model * text_model, int n_threads, bool use_gpu) {
        auto p = mtmd_context_params_default();
        p.use_gpu    = use_gpu;
        p.n_threads  = n_threads;
        p.warmup     = true;
        auto * raw = mtmd_init_from_file(path.c_str(), text_model, p);
        if (!raw) { return false; }
        ctx.reset(raw);
        LOG_INF("mmproj loaded: vision=%d audio=%d\n",
                (int)mtmd_support_vision(raw), (int)mtmd_support_audio(raw));
        return true;
    }

    bool has_vision() const { return ctx && mtmd_support_vision(ctx.get()); }
    bool has_audio()  const { return ctx && mtmd_support_audio(ctx.get());  }

    mtmd::bitmap_ptr bitmap_from_image_b64(const std::string & b64) const {
        auto raw = b64_decode(b64);
        if (raw.empty()) { return nullptr; }
        auto * bmp = mtmd_helper_bitmap_init_from_buf(ctx.get(), raw.data(), raw.size());
        return mtmd::bitmap_ptr(bmp);
    }

    mtmd::bitmap_ptr bitmap_from_audio_b64(const std::string & b64) const {
        auto pcm = b64_to_float32_pcm(b64);
        if (pcm.empty()) { return nullptr; }
        auto wav = build_wav_from_pcm(pcm);
        if (wav.empty()) { return nullptr; }
        auto * bmp = mtmd_helper_bitmap_init_from_buf(ctx.get(), wav.data(), wav.size());
        return mtmd::bitmap_ptr(bmp);
    }

    mtmd::bitmap_ptr bitmap_from_file(const std::string & path) const {
        auto * bmp = mtmd_helper_bitmap_init_from_file(ctx.get(), path.c_str());
        return mtmd::bitmap_ptr(bmp);
    }
};

// ============================================================================
// Shared text model
// ============================================================================

struct SharedModel {
    llama_model * model = nullptr;

    bool load(const std::string & path, const common_params & params) {
        LOG_INF("Loading GGUF: %s\n", path.c_str());
        auto mparams = common_model_params_to_llama(const_cast<common_params&>(params));
        model = llama_model_load_from_file(path.c_str(), mparams);
        if (!model) { LOG_ERR("load failed\n"); return false; }
        char buf[128];
        llama_model_desc(model, buf, sizeof(buf));
        LOG_INF("Loaded: %s\n", buf);
        return true;
    }

    void free() { if (model) { llama_model_free(model); model = nullptr; } }
    ~SharedModel() { free(); }
};

// ============================================================================
// Qwen3 ChatML prompt builder with <__media__> markers
// ============================================================================

static std::string build_qwen3_prompt(const std::vector<ParsedMessage> & msgs,
                                      const ParsedMessage * last_user,
                                      int n_media_markers)
{
    std::string prompt;

    bool has_system = false;
    for (const auto & m : msgs) {
        if (m.role == "system" && !m.text.empty()) {
            prompt += "<|im_start|>system\n" + m.text + "<|im_end|>\n";
            has_system = true;
            break;
        }
    }
    if (!has_system) {
        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }

    for (size_t i = 0; i < msgs.size(); i++) {
        const auto & m = msgs[i];
        if (&m == last_user) { break; }
        if (m.role == "system") { continue; }
        if (m.text.empty()) { continue; }
        prompt += "<|im_start|>" + m.role + "\n" + m.text + "<|im_end|>\n";
    }

    if (last_user) {
        prompt += "<|im_start|>user\n";
        for (int i = 0; i < n_media_markers; i++) { prompt += "<__media__>"; }
        if (!last_user->text.empty()) {
            if (n_media_markers > 0) { prompt += "\n"; }
            prompt += last_user->text;
        }
        prompt += "<|im_end|>\n";
    }

    prompt += "<|im_start|>assistant\n";
    return prompt;
}

// ============================================================================
// Text eval helpers (batch with position handling)
// ============================================================================

static bool eval_tokens(llama_context * ctx, const std::vector<llama_token> & tokens,
                        int n_batch, int * n_past)
{
    for (int i = 0; i < (int)tokens.size(); i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch) { n_eval = n_batch; }

        auto batch = llama_batch_get_one(const_cast<llama_token*>(&tokens[i]), n_eval);
        if (batch.pos == nullptr) {
            static thread_local std::vector<llama_pos> s_pos;
            s_pos.resize(n_eval);
            batch.pos = s_pos.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;
        }

        if (llama_decode(ctx, batch)) {
            LOG_ERR("eval_tokens: llama_decode failed at pos %d\n", *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

// ============================================================================
// UTF-8 safe streaming helpers (defined before usage in streaming generator)
// ============================================================================

static bool utf8_is_cont(unsigned char b) {
    return (b & 0xC0) == 0x80;
}

static std::vector<std::string> utf8_safe_chunks(const std::string & text, size_t target) {
    std::vector<std::string> chunks;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = std::min(pos + target, text.size());
        while (end < text.size() && utf8_is_cont((unsigned char)text[end])) { end++; }
        chunks.push_back(text.substr(pos, end - pos));
        pos = end;
    }
    return chunks;
}

// ============================================================================
// KV cache sliding window (keep system prompt, evict oldest tokens)
// ============================================================================

static void kv_cache_slide_window(llama_context * ctx, int n_ctx, int * n_past, int n_keep) {
    const int slack = 512;
    if (*n_past + 1 < n_ctx - slack) { return; }
    if (n_keep <= 0) { return; }

    int n_discard = std::max(16, *n_past - n_keep - (n_ctx / 4));
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
    }
    *n_past -= n_discard;
    LOG_INF("sliding window: n_past -> %d (keep=%d)\n", *n_past, n_keep);
}

// UTF-8 stream sanitizer (from ws_handler.cpp) — buffers incomplete multi-byte
// sequences across tokens so nlohmann::json::dump() never sees partial characters.
static std::string sanitize_utf8_stream(std::string & pending,
                                        const std::string & fragment,
                                        bool flush = false) {
    static const std::string replacement = "\xEF\xBF\xBD";
    std::string input = pending + fragment;
    pending.clear();
    std::string out;
    size_t i = 0;
    while (i < input.size()) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c < 0x80) { out.push_back(static_cast<char>(c)); i++; continue; }
        int need = 0;
        if (c >= 0xC2 && c <= 0xDF)       { need = 1; }
        else if (c >= 0xE0 && c <= 0xEF)  { need = 2; }
        else if (c >= 0xF0 && c <= 0xF4)  { need = 3; }
        else { out += replacement; i++; continue; }
        if (i + need >= input.size()) { pending = input.substr(i); break; }
        bool ok = true;
        for (int j = 1; j <= need; ++j) { ok = ok && utf8_is_cont((unsigned char)input[i + j]); }
        if (!ok) { out += replacement; i++; continue; }
        out.append(input, i, need + 1);
        i += need + 1;
    }
    if (flush && !pending.empty()) { out += replacement; pending.clear(); }
    return out;
}

// ============================================================================
// True streaming generation — sends text deltas inline as tokens decode.
// Returns accumulated text (with <|im_end|> stripped) for response.done.
// ============================================================================

static std::string generate_tokens_streaming(llama_context * ctx, common_sampler * smpl,
                                              int n_past, int n_ctx, int n_keep,
                                              int max_new, llama_token eos,
                                              const std::string & sid, const std::string & rid,
                                              httplib::ws::WebSocket & ws, bool streaming,
                                              std::atomic<bool> * interrupt = nullptr)
{
    common_sampler_reset(smpl);
    std::string full_text;
    std::string utf8_pending;
    std::string im_end_buf;  // accumulate token pieces to catch split <|im_end|>

    auto send_delta = [&](const std::string & text) {
        if (!streaming || text.empty()) { return; }
        for (const auto & ch : utf8_safe_chunks(text, 4)) {
            ws.send(json_safe_dump(make_text_delta(sid, rid, ch, ProtocolMetrics{})));
        }
    };

    for (int i = 0; i < max_new; ++i) {
        if (interrupt && interrupt->load()) { break; }  // barge-in

        kv_cache_slide_window(ctx, n_ctx, &n_past, n_keep);

        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (id == eos || id == 151645) { break; }

        std::string piece = common_token_to_piece(ctx, id);

        // <|im_end|> may be split across multiple tokens (e.g. text-form
        // "<|im" + "_end|>"). Buffer pieces; flush only what is confirmed not
        // to be part of the end marker, so generation stops exactly at
        // <|im_end|> without leaking it (or anything after) into the stream.
        im_end_buf += piece;
        auto pos = im_end_buf.find("<|im_end|>");
        if (pos != std::string::npos) {
            // Flush everything before the marker, then stop.
            std::string before = im_end_buf.substr(0, pos);
            std::string safe_before = sanitize_utf8_stream(utf8_pending, before);
            full_text += safe_before;
            send_delta(safe_before);
            break;
        }
        // Keep the buffer small: only the tail that could become <|im_end|>.
        if (im_end_buf.size() > 16) { im_end_buf = im_end_buf.substr(im_end_buf.size() - 16); }
        // How much of im_end_buf could still be a prefix of <|im_end|>?
        const std::string marker = "<|im_end|>";
        size_t prefix_len = 0;
        for (size_t k = 1; k <= im_end_buf.size() && k < marker.size(); k++) {
            if (marker.compare(0, k, im_end_buf, im_end_buf.size() - k, k) == 0) {
                prefix_len = k;
            }
        }
        // Flush everything except the possible marker prefix.
        size_t flush_len = im_end_buf.size() - prefix_len;
        if (flush_len > 0) {
            std::string flush = im_end_buf.substr(0, flush_len);
            im_end_buf.erase(0, flush_len);
            std::string safe = sanitize_utf8_stream(utf8_pending, flush);
            full_text += safe;
            send_delta(safe);
        }

        llama_token batch_tokens[] = {id};
        auto batch = llama_batch_get_one(batch_tokens, 1);
        if (batch.pos == nullptr) {
            static thread_local std::vector<llama_pos> s_pos(1);
            batch.pos = s_pos.data();
        }
        batch.pos[0] = n_past++;
        if (llama_decode(ctx, batch)) { break; }
    }

    // Flush remaining incomplete UTF-8
    std::string tail = sanitize_utf8_stream(utf8_pending, "", true);
    full_text += tail;
    send_delta(tail);

    // Strip <|im_end|> and ensure valid UTF-8 for json::dump()
    auto pos = full_text.find("<|im_end|>");
    if (pos != std::string::npos) { full_text.resize(pos); }
    {
        std::string dummy;
        full_text = sanitize_utf8_stream(dummy, full_text, true);
    }
    return full_text;
}

// ============================================================================
// Server state
// ============================================================================

struct ServerState {
    SharedModel  & shared_model;
    SharedMmproj & shared_mmproj;
    SessionManager & mgr;
    common_params_sampling sampling;
    bool text_only;
    int  n_batch;
    int  max_sessions;
    int  total_n_ctx;
    std::mutex mmproj_mtx;
    // Pre-allocated context pool to avoid CUDA fragmentation on second alloc.
    std::vector<llama_context *> ctx_pool;
    std::mutex ctx_pool_mtx;

    // Barge-in interrupt flags, keyed by session_id. Each handle_ws registers its
    // session's atomic (shared_ptr, so the route can find it even while the
    // session lives on the WS thread) and unregisters on disconnect.
    std::mutex interrupt_mtx;
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> interrupts;

    void register_interrupt(const std::string & sid,
                            const std::shared_ptr<std::atomic<bool>> & flag) {
        flag->store(false);
        std::lock_guard<std::mutex> lock(interrupt_mtx);
        interrupts[sid] = flag;
    }

    void set_interrupt(const std::string & sid) {
        std::lock_guard<std::mutex> lock(interrupt_mtx);
        auto it = interrupts.find(sid);
        if (it != interrupts.end()) { it->second->store(true); }
    }

    void unregister_interrupt(const std::string & sid) {
        std::lock_guard<std::mutex> lock(interrupt_mtx);
        interrupts.erase(sid);
    }

    ServerState(SharedModel & sm, SharedMmproj & mm, SessionManager & m,
                const common_params_sampling & sp, bool to, int nb, int ms, int tnc)
        : shared_model(sm), shared_mmproj(mm), mgr(m), sampling(sp),
          text_only(to), n_batch(nb), max_sessions(ms), total_n_ctx(tnc) {
        int per_session_n_ctx = total_n_ctx / max_sessions;
        if (per_session_n_ctx < 128) { per_session_n_ctx = 128; }
        // NOTE: no pre-allocation. Each session creates its own context via
        // llama_init_from_model in handle_ws and frees it on disconnect.
        // Pre-allocating max_sessions contexts doubles GPU VRAM (pool + per-session)
        // and caused "cudaMalloc failed: out of memory" on the 2nd concurrent init
        // with a large model (Qwen3-Omni 30B + mmproj already uses ~21GB).
        // ctx_pool stays empty; borrow_ctx()/return_ctx() are unused for now.
        ctx_pool.clear();
    }

    llama_context * borrow_ctx() {
        std::lock_guard<std::mutex> lock(ctx_pool_mtx);
        for (auto & ctx : ctx_pool) {
            if (ctx) { auto * r = ctx; ctx = nullptr; return r; }
        }
        return nullptr;
    }

    void return_ctx(llama_context * ctx) {
        if (!ctx) return;
        {
            llama_memory_t mem = llama_get_memory(ctx);
            if (mem) { llama_memory_seq_rm(mem, 0, 0, -1); }
        }
        std::lock_guard<std::mutex> lock(ctx_pool_mtx);
        for (auto & slot : ctx_pool) {
            if (slot == nullptr) { slot = ctx; return; }
        }
        llama_free(ctx);
    }

    ~ServerState() {
        for (auto ctx : ctx_pool) { if (ctx) llama_free(ctx); }
    }
};

// ============================================================================
// WebSocket /backend handler
// ============================================================================

static void handle_ws(httplib::ws::WebSocket & ws, ServerState & state) {
    std::string sid = state.mgr.allocate();
    if (sid.empty()) { ws.close(); return; }
    state.mgr.set_close_callback(sid, [&ws]() { ws.close(); });

    Qwen3Session sess;

    auto fail_fast = [&](const std::string & reason) {
        if (!sid.empty()) {
            ws.send(json_safe_dump(make_session_closed(sid, reason)));
        }
        state.mgr.close(sid);
        ws.close();
    };

    // ---- Read first message: session.init ----
    std::string raw;
    auto read_result = ws.read(raw);
    if (read_result != httplib::ws::ReadResult::Text) { return; }

    json first_msg;
    try { first_msg = json::parse(raw); } catch (...) { return; }
    auto parsed_init = parse_session_init(first_msg);
    if (!parsed_init.ok) { ws.close(); return; }

    sess.sid = sid;
    sess.half_duplex = (parsed_init.mode == "half_duplex" || parsed_init.mode == "full_duplex");
    state.register_interrupt(sid, sess.interrupt);

    // Per-session n_ctx = total_n_ctx / max_sessions (same as MiniCPM).
    // User can override via session.init payload.config.n_ctx.
    int per_session_n_ctx = state.total_n_ctx / state.max_sessions;
    if (per_session_n_ctx < 128) { per_session_n_ctx = 128; }
    int nc = parsed_init.config.contains("n_ctx") ? parsed_init.config["n_ctx"].get<int>()
             : per_session_n_ctx;
    LOG_INF("session %s: n_ctx=%d (total=%d max_sessions=%d)\n",
            sid.c_str(), nc, state.total_n_ctx, state.max_sessions);

    auto cp = llama_context_default_params();
    cp.n_ctx    = nc;
    cp.n_batch  = state.n_batch;
    cp.n_ubatch = state.n_batch;
    cp.n_seq_max = 1;
    sess.ctx = llama_init_from_model(state.shared_model.model, cp);
    if (!sess.ctx) { fail_fast("ctx_init_failed"); return; }
    sess.n_ctx = nc;

    sess.smpl = common_sampler_init(state.shared_model.model, state.sampling);
    if (!sess.smpl) { fail_fast("sampler_init_failed"); return; }

    state.mgr.activate(sid, nullptr, false);
    ws.send(json_safe_dump(make_session_created(sid, parsed_init.mode)));
    LOG_INF("session %s started (text_only=%d)\n", sid.c_str(), (int)state.text_only);

    // ---- Read loop: input.append ----
    while (true) {
        read_result = ws.read(raw);
        if (read_result != httplib::ws::ReadResult::Text) { break; }

        json msg;
        try { msg = json::parse(raw); } catch (...) { fail_fast("invalid_json"); return; }

        std::string msg_type;
        if (msg.contains("type") && msg["type"].is_string()) {
            msg_type = msg["type"].get<std::string>();
        }

        if (msg_type != "input.append") {
            fail_fast("unexpected_message_type"); return;
        }

        // Refresh idle timer — session is alive.
        state.mgr.touch(sid);

        auto parsed_input = parse_input_append(msg);
        if (!parsed_input.ok) { fail_fast("invalid_input"); return; }

        std::string rid = sid + "-" + std::to_string(++sess.resp_cnt);
        int max_new = parsed_input.max_new_tokens > 0 ? parsed_input.max_new_tokens : 512;

        // Determine the EOS token. For Qwen3 the correct EOS is <|im_end|> (151645).
        // llama_vocab_eos() may return the wrong token (</s> = 128247) if the GGUF
        // overrides it. Look up <|im_end|> in the vocabulary explicitly.
        llama_token eos_tok = -1;
        {
            auto vocab = llama_model_get_vocab(state.shared_model.model);
            auto im_end_toks = common_tokenize(vocab, "<|im_end|>", false, true);
            if (im_end_toks.size() == 1) {
                eos_tok = im_end_toks[0];
            }
        }
        if (eos_tok == -1) {
            eos_tok = llama_vocab_eos(llama_model_get_vocab(state.shared_model.model));
        }
        if (eos_tok == -1) { eos_tok = 151645; }

        // ====================================================================
        // Half-duplex / full-duplex streaming path (VAD+TurnSense worker-driven)
        // --------------------------------------------------------------------
        // The worker streams audio/video chunks with force_listen=true (prefill
        // only, accumulate into KV). When VAD+TurnSense says the user finished,
        // the worker sends a chunk WITHOUT force_listen → we decode a full reply.
        // Barge-in sets sess.interrupt via POST /interrupt; checked each token.
        // ====================================================================
        if (sess.half_duplex) {
            const bool use_mmproj_hd = !state.text_only && state.shared_mmproj.ctx != nullptr;
            const bool has_audio_hd  = !parsed_input.audio_b64.empty();
            const bool has_frames_hd = !parsed_input.video_frames_b64.empty();

            // ---- System prompt: prefill once at session start ----
            if (!sess.system_prompt_initialized) {
                std::string sys_prompt = parsed_init.system_prompt.empty()
                    ? "You are a helpful assistant."
                    : parsed_init.system_prompt;
                std::string sys_text = "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n";
                auto vocab_sys = llama_model_get_vocab(state.shared_model.model);
                auto sys_toks = common_tokenize(vocab_sys, sys_text, true, false);
                if (!eval_tokens(sess.ctx, sys_toks, state.n_batch, &sess.n_past)) {
                    fail_fast("eval_failed"); return;
                }
                sess.n_keep = sess.n_past;
                sess.system_prompt_initialized = true;
            }

            // ---- Stream a media chunk into KV (prefill) ----
            if (has_audio_hd || has_frames_hd) {
                if (!use_mmproj_hd) {
                    fail_fast("mmproj_required"); return;
                }

                // Build user-turn marker: N media items this chunk
                int n_media = (has_audio_hd ? 1 : 0) + (int)parsed_input.video_frames_b64.size();
                std::string user_text = "<|im_start|>user\n";
                for (int i = 0; i < n_media; i++) { user_text += "<__media__>"; }
                user_text += "<|im_end|>\n";

                // Build bitmaps: audio first (1), then frames
                std::vector<mtmd::bitmap_ptr> owned_bmps;
                std::vector<const mtmd_bitmap *> raw_bmps;
                owned_bmps.reserve((size_t)n_media);
                raw_bmps.reserve((size_t)n_media);

                if (has_audio_hd) {
                    auto bmp = state.shared_mmproj.bitmap_from_audio_b64(parsed_input.audio_b64);
                    if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
                }
                for (const auto & frame_b64 : parsed_input.video_frames_b64) {
                    auto bmp = state.shared_mmproj.bitmap_from_image_b64(frame_b64);
                    if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
                }

                mtmd_input_chunks * chunks_raw = mtmd_input_chunks_init();
                mtmd::input_chunks_ptr chunks(chunks_raw);
                mtmd_input_text txt;
                txt.text          = user_text.c_str();
                txt.add_special   = true;
                txt.parse_special = false;

                llama_pos new_n_past = sess.n_past;
                {
                    std::lock_guard<std::mutex> lock(state.mmproj_mtx);
                    int32_t ret = mtmd_tokenize(state.shared_mmproj.ctx.get(), chunks_raw,
                                                &txt, raw_bmps.data(), raw_bmps.size());
                    if (ret != 0) { fail_fast("mtmd_tokenize_failed"); return; }
                    // force_listen (accumulate): skip logits (cheaper).
                    // trigger (decode follows): keep last-token logits for the sampler.
                    const bool keep_logits = !parsed_input.force_listen;
                    ret = mtmd_helper_eval_chunks(state.shared_mmproj.ctx.get(), sess.ctx,
                                                  chunks_raw, new_n_past, 0, state.n_batch,
                                                  /*logits_last*/keep_logits, &new_n_past);
                    if (ret != 0) { fail_fast("mtmd_eval_failed"); return; }
                }
                sess.n_past = (int)new_n_past;
            }

            // Keep KV within budget while streaming: slide the window when the
            // accumulated stream (video frames + audio) would otherwise exhaust
            // the context. Evicts the oldest tokens (after the system prompt),
            // keeping the most recent stream content.
            // NOTE: no sliding window during prefill. Sliding here discards the
            // audio/video embedding tokens (keep=20 keeps only the system
            // prompt), corrupting the context — the model then emits garbage
            // (e.g. "xxxxx") and stops replying. Long audio should be bounded
            // with --max-audio-s instead. decode-time sliding (inside
            // generate_tokens_streaming) still applies as a last resort.

            // ---- force_listen: prefill only, no decode ----
            if (parsed_input.force_listen) {
                ws.send(json_safe_dump(make_listen_delta(sid, rid, ProtocolMetrics{})));
                continue;
            }

            // ---- Decode a full reply from current KV ----
            sess.interrupt->store(false);
            // Signal the model it's the assistant's turn to speak: eval the
            // assistant marker BEFORE decoding. Without this the model sees a
            // bare user turn and echoes it back instead of replying.
            {
                auto vocab_a = llama_model_get_vocab(state.shared_model.model);
                auto asst_toks = common_tokenize(vocab_a, "<|im_start|>assistant\n", true, false);
                if (!eval_tokens(sess.ctx, asst_toks, state.n_batch, &sess.n_past)) {
                    fail_fast("eval_failed"); return;
                }
            }
            auto text = generate_tokens_streaming(sess.ctx, sess.smpl, sess.n_past, sess.n_ctx,
                                                   sess.n_keep, max_new, eos_tok,
                                                   sid, rid, ws, parsed_input.streaming,
                                                   sess.interrupt.get());
            // generate_tokens_streaming advances a local n_past; restore the
            // authoritative position from the KV cache.
            {
                llama_memory_t mem = llama_get_memory(sess.ctx);
                if (mem) {
                    llama_pos used = llama_memory_seq_pos_max(mem, 0);
                    if (used >= 0) { sess.n_past = (int)used + 1; }
                }
            }
            // Ensure the assistant turn boundary for the next user chunk
            {
                auto vocab_a = llama_model_get_vocab(state.shared_model.model);
                auto asst_toks = common_tokenize(vocab_a, "<|im_start|>assistant\n<|im_end|>\n", true, false);
                if (!eval_tokens(sess.ctx, asst_toks, state.n_batch, &sess.n_past)) {
                    fail_fast("eval_failed"); return;
                }
            }
            ws.send(json_safe_dump(make_response_done(sid, rid, text, "", "turn_end", ProtocolMetrics{})));

            // 每段回复完成后清理 KV，下一段从干净上下文开始。
            // 否则多段 turn-based 对话的 KV 持续累积（音频+视频+回复），
            // 超过 per-session n_ctx 后 "failed to find a memory slot"，
            // 后续 prefill 失败、连接被关。
            {
                llama_memory_t mem = llama_get_memory(sess.ctx);
                if (mem) { llama_memory_seq_rm(mem, 0, 0, -1); }
            }
            sess.n_past = 0;
            sess.n_keep = 0;
            sess.system_prompt_initialized = false;  // 下段重新 prefill 系统 prompt
            continue;
        }

        auto msgs = parse_messages_array(parsed_input.messages);
        if (msgs.empty()) { fail_fast("empty_messages"); return; }

        const ParsedMessage * last_user = nullptr;
        for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
            if (it->role == "user") { last_user = &(*it); break; }
        }
        if (!last_user) { fail_fast("no_user_message"); return; }

        const bool has_images = !last_user->image_b64s.empty();
        const bool has_audio  = !last_user->audio_b64s.empty();
        const bool has_video  = !last_user->video_b64s.empty();
        const bool use_mmproj = !state.text_only && state.shared_mmproj.ctx != nullptr;

        // Turn-based rebuilds the full prompt from the message history every
        // input. Clear the KV cache first so positions restart cleanly from 0;
        // otherwise the previous turn's tokens collide (decode: failed to
        // initialize batch / eval_tokens: llama_decode failed at pos 0).
        {
            llama_memory_t mem = llama_get_memory(sess.ctx);
            if (mem) { llama_memory_seq_rm(mem, 0, 0, -1); }
        }

        int n_past = 0;

        if (use_mmproj && (has_images || has_audio || has_video)) {
            // ======== MULTIMODAL PATH via mtmd ========

            // ---- Step 1: extract video frames + audio (if any) ----
            std::vector<std::string> video_frame_paths;
            std::string video_audio_path;
            if (has_video) {
                const std::string temp_dir = (fs::temp_directory_path() / "qwen3omni_video").string();
                fs::create_directories(temp_dir);
                thread_local int video_counter = 0;
                ++video_counter;
                for (const auto & vid_b64 : last_user->video_b64s) {
                    auto vid = extract_video_mp4_media(vid_b64, temp_dir, video_counter,
                                                       /*stack_frames*/2);
                    if (!vid.frame_paths.empty()) {
                        for (auto & fp : vid.frame_paths) {
                            video_frame_paths.push_back(std::move(fp));
                        }
                        // Use first video's audio (most common case: single video per turn)
                        if (video_audio_path.empty() && !vid.audio_path.empty()) {
                            video_audio_path = vid.audio_path;
                        }
                    }
                }
            }

            // ---- Step 2: count total media items ----
            int n_user_images = (int)last_user->image_b64s.size();
            int n_video_frames = (int)video_frame_paths.size();
            int n_total_audio = (int)last_user->audio_b64s.size()
                              + (video_audio_path.empty() ? 0 : 1);
            int n_media = n_user_images + n_video_frames + n_total_audio;

            std::string prompt = build_qwen3_prompt(msgs, last_user, n_media);

            // ---- Step 3: build bitmaps array ----
            std::vector<mtmd::bitmap_ptr> owned_bmps;
            std::vector<const mtmd_bitmap *> raw_bmps;
            owned_bmps.reserve((size_t)n_media);
            raw_bmps.reserve((size_t)n_media);

            // Images first
            for (const auto & img_b64 : last_user->image_b64s) {
                auto bmp = state.shared_mmproj.bitmap_from_image_b64(img_b64);
                if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
            }
            // Video frames (as images)
            for (const auto & fpath : video_frame_paths) {
                auto bmp = state.shared_mmproj.bitmap_from_file(fpath);
                if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
            }
            // User audio
            for (const auto & aud_b64 : last_user->audio_b64s) {
                auto bmp = state.shared_mmproj.bitmap_from_audio_b64(aud_b64);
                if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
            }
            // Video audio
            if (!video_audio_path.empty()) {
                auto bmp = state.shared_mmproj.bitmap_from_file(video_audio_path);
                if (bmp) { raw_bmps.push_back(bmp.get()); owned_bmps.push_back(std::move(bmp)); }
            }

            // ---- Step 4: mtmd tokenize + eval (locked) ----
            mtmd_input_chunks * chunks_raw = mtmd_input_chunks_init();
            mtmd::input_chunks_ptr chunks(chunks_raw);

            mtmd_input_text txt;
            txt.text          = prompt.c_str();
            txt.add_special   = true;
            txt.parse_special = false;

            llama_pos new_n_past = 0;
            {
                std::lock_guard<std::mutex> lock(state.mmproj_mtx);
                int32_t ret = mtmd_tokenize(state.shared_mmproj.ctx.get(), chunks_raw,
                                            &txt, raw_bmps.data(), raw_bmps.size());
                if (ret != 0) { fail_fast("mtmd_tokenize_failed"); return; }

                ret = mtmd_helper_eval_chunks(state.shared_mmproj.ctx.get(), sess.ctx,
                                              chunks_raw, 0, 0, state.n_batch, true, &new_n_past);
                if (ret != 0) { fail_fast("mtmd_eval_failed"); return; }
            }
            n_past = (int)new_n_past;
            if (sess.n_keep == 0) { sess.n_keep = std::min(n_past, 512); }

            // Cleanup temp files (best-effort)
            if (has_video) {
                for (const auto & fp : video_frame_paths) { fs::remove(fp); }
                if (!video_audio_path.empty()) { fs::remove(video_audio_path); }
            }

        } else {
            // ======== TEXT PATH (with or without mmproj) ========
            std::string prompt = build_qwen3_prompt(msgs, last_user, 0);
            auto vocab = llama_model_get_vocab(state.shared_model.model);
            auto toks = common_tokenize(vocab, prompt, true, false);
            if (!eval_tokens(sess.ctx, toks, state.n_batch, &n_past)) {
                fail_fast("eval_failed"); return;
            }
            if (sess.n_keep == 0) { sess.n_keep = std::min(n_past, 512); }
        }

        // ---- Generate tokens (true streaming — deltas sent inline) ----
        auto text = generate_tokens_streaming(sess.ctx, sess.smpl, n_past, sess.n_ctx, sess.n_keep,
                                               max_new, eos_tok,
                                               sid, rid, ws, parsed_input.streaming);

        // Send response.done
        ws.send(json_safe_dump(make_response_done(sid, rid, text, "", "turn_end", ProtocolMetrics{})));
    }

    // ---- Cleanup on disconnect ----
    LOG_INF("session %s disconnected\n", sid.c_str());
    state.unregister_interrupt(sid);
    ws.send(json_safe_dump(make_session_closed(sid, "client_disconnected")));
    state.mgr.close(sid);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    // Strip custom flags before common_params_parse
    bool text_only = false;
    {
        int wi = 1;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--text-only") == 0) { text_only = true; }
            else { argv[wi++] = argv[i]; }
        }
        argv[wi] = nullptr;
        argc = wi;
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) { return 1; }
    if (params.model.path.empty()) { LOG_ERR("use -m <gguf>\n"); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Load text model
    SharedModel shared_model;
    if (!shared_model.load(params.model.path, params)) { llama_backend_free(); return 1; }

    // Load mmproj (optional)
    SharedMmproj shared_mmproj;
    if (!text_only && !params.mmproj.path.empty()) {
        int n_threads = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : 4;
        if (!shared_mmproj.load(params.mmproj.path, shared_model.model,
                                n_threads, params.mmproj_use_gpu))
        {
            LOG_WRN("mmproj load failed, falling back to text-only\n");
        }
    } else if (!text_only) {
        LOG_INF("No --mmproj specified; text-only mode\n");
    }

    int max_sessions = params.n_parallel > 0 ? params.n_parallel : 4;
    int n_batch      = params.n_batch > 0 ? params.n_batch : 512;
    // Total KV cache: from -c flag. Per-session = total / max_sessions.
    int total_n_ctx = params.n_ctx;
    if (total_n_ctx <= 0) {
        total_n_ctx = 16384;
    }
    LOG_INF("Server: max_sessions=%d total_n_ctx=%d (per-session=%d)\n",
            max_sessions, total_n_ctx, total_n_ctx / max_sessions);

    // HTTP server
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif
    svr.set_read_timeout(params.timeout_read);

    SessionManager mgr(max_sessions);
    ServerState state(shared_model, shared_mmproj, mgr, params.sampling, text_only, n_batch, max_sessions, total_n_ctx);

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"status","ok"},{"engine","qwen3omni"},{"sessions",mgr.active_count()}};
        res.set_content(j.dump(), "application/json");
    });
    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"status","ok"},{"engine","qwen3omni"},{"sessions",mgr.active_count()}};
        res.set_content(j.dump(), "application/json");
    });

    svr.WebSocket("/backend", [&](const httplib::Request &, httplib::ws::WebSocket & ws) {
        handle_ws(ws, state);
    });

    svr.Post("/sessions/:session_id/close", [&](const httplib::Request & req, httplib::Response & res) {
        auto sid = req.path_params.at("session_id");
        // Close the session and the transport WS. The worker's backend_to_client
        // will see ConnectionClosed and handle it gracefully (no error sent to frontend).
        if (mgr.get(sid)) {
            mgr.request_transport_close(sid);
            mgr.close(sid);
        }
        json j = {{"ok",true},{"session_id",sid},{"closed",true}};
        res.set_content(j.dump(), "application/json");
    });

    // Barge-in: stop current generation, keep session alive.
    svr.Post("/sessions/:session_id/interrupt", [&](const httplib::Request & req, httplib::Response & res) {
        auto sid = req.path_params.at("session_id");
        state.set_interrupt(sid);
        json j = {{"ok",true},{"session_id",sid},{"interrupted",true}};
        res.set_content(j.dump(), "application/json");
    });

    LOG_INF("Qwen3-Omni server on 0.0.0.0:%d (sessions=%d text_only=%d)\n",
            params.port, max_sessions, (int)text_only);

    // Idle-session reaper: periodically close sessions inactive too long
    // (e.g. client WS died without a clean close). With max_sessions=1 a
    // single leaked session blocks all new connections.
    // 300s: turn-based replies can be long; 60s reclaimed sessions mid-reply.
    const double idle_timeout_s = 300.0;
    std::atomic<bool> reaper_stop{false};
    std::thread reaper([&]() {
        while (!reaper_stop.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(15));
            auto reaped = mgr.reap_idle(idle_timeout_s);
            if (!reaped.empty()) {
                LOG_INF("Reaped %zu idle session(s) (timeout %.0fs)\n",
                        reaped.size(), idle_timeout_s);
            }
        }
    });

    svr.listen("0.0.0.0", params.port);

    LOG_INF("Shutting down\n");
    reaper_stop.store(true);
    if (reaper.joinable()) { reaper.join(); }
    mgr.shutdown();
    shared_mmproj.ctx.reset();
    shared_model.free();
    llama_backend_free();
    return 0;
}

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

#define CPPHTTPLIB_WEBSOCKET_MAX_PAYLOAD_LENGTH (128 * 1024 * 1024)
#include "httplib.h"
#include <nlohmann/json.hpp>

// protocol.h defines: using json = nlohmann::ordered_json;
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

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

    const int n_frames = std::max(1, std::min(stack_frames, 8));
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

    // Extract audio: mono 16kHz PCM WAV
    std::string audio_path = (dir / "audio.wav").string();
    std::string audio_cmd = "ffmpeg -y -hide_banner -loglevel error -i "
        + shell_quote(out.video_path)
        + " -vn -ac 1 -ar 16000 -c:a pcm_f32le "
        + shell_quote(audio_path);
    if (std::system(audio_cmd.c_str()) == 0 && file_nonempty(audio_path)) {
        out.audio_path = audio_path;
    }

    // Extract N JPEG frames at even intervals
    std::string frame_pattern = (dir / "frame_%03d.jpg").string();
    std::string frame_cmd = "ffmpeg -y -hide_banner -loglevel error -i "
        + shell_quote(out.video_path)
        + " -an -frames:v " + std::to_string(n_frames)
        + " -q:v 2 " + shell_quote(frame_pattern);
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

    ~Qwen3Session() {
        if (smpl) { common_sampler_free(smpl); smpl = nullptr; }
        if (ctx)  { llama_free(ctx);            ctx  = nullptr; }
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

static std::string generate_tokens(llama_context * ctx, common_sampler * smpl,
                                    int n_past, int max_new, llama_token eos)
{
    common_sampler_reset(smpl);
    std::string out;
    for (int i = 0; i < max_new; ++i) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        // Qwen3 uses <|im_end|> (151645) as EOS. Accept both the lookup-based
        // eos and the hardcoded Qwen3 token ID, plus detect the rendered text.
        if (id == eos || id == 151645) { break; }

        std::string piece = common_token_to_piece(ctx, id);
        if (piece == "<|im_end|>") { break; }
        out += piece;

        llama_token batch_tokens[] = {id};
        auto batch = llama_batch_get_one(batch_tokens, 1);
        if (batch.pos == nullptr) {
            static thread_local std::vector<llama_pos> s_pos(1);
            batch.pos = s_pos.data();
        }
        batch.pos[0] = n_past++;
        if (llama_decode(ctx, batch)) { break; }
    }
    return out;
}

// ============================================================================
// UTF-8 safe streaming helpers
// ============================================================================

// Return true if byte b is a UTF-8 continuation byte (10xxxxxx)
static bool utf8_is_cont(unsigned char b) {
    return (b & 0xC0) == 0x80;
}

// Split text into chunks of at most `target` bytes, only at UTF-8 character
// boundaries.  nlohmann::json::dump() throws type_error.316 on incomplete
// UTF-8, so we must never produce a chunk that ends in the middle of a
// multi-byte character (common with CJK text which is 3 bytes per char).
static std::vector<std::string> utf8_safe_chunks(const std::string & text, size_t target) {
    std::vector<std::string> chunks;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = std::min(pos + target, text.size());
        // If end lands on a continuation byte, extend to the next char boundary
        while (end < text.size() && utf8_is_cont((unsigned char)text[end])) {
            end++;
        }
        chunks.push_back(text.substr(pos, end - pos));
        pos = end;
    }
    return chunks;
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
    std::mutex mmproj_mtx;

    ServerState(SharedModel & sm, SharedMmproj & mm, SessionManager & m,
                const common_params_sampling & sp, bool to, int nb)
        : shared_model(sm), shared_mmproj(mm), mgr(m), sampling(sp), text_only(to), n_batch(nb) {}
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
            ws.send(make_session_closed(sid, reason).dump());
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
    // Default to model's maximum context length (capped to 16384 to save VRAM)
    // which is safe for multi-frame video (~1530 tokens per frame) + images + audio + text.
    int max_ctx = llama_model_n_ctx_train(state.shared_model.model);
    int nc = parsed_init.config.contains("n_ctx") ? parsed_init.config["n_ctx"].get<int>()
             : std::min(max_ctx > 0 ? max_ctx : 16384, 16384);
    LOG_INF("session %s: n_ctx=%d (model_train=%d)\n", sid.c_str(), nc, max_ctx);

    auto cp = llama_context_default_params();
    cp.n_ctx    = nc;
    cp.n_batch  = state.n_batch;
    cp.n_ubatch = state.n_batch;
    cp.n_seq_max = 1;
    sess.ctx = llama_init_from_model(state.shared_model.model, cp);
    if (!sess.ctx) { fail_fast("ctx_init_failed"); return; }

    sess.smpl = common_sampler_init(state.shared_model.model, state.sampling);
    if (!sess.smpl) { fail_fast("sampler_init_failed"); return; }

    state.mgr.activate(sid, nullptr, false);
    ws.send(make_session_created(sid, parsed_init.mode).dump());
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

        auto parsed_input = parse_input_append(msg);
        if (!parsed_input.ok) { fail_fast("invalid_input"); return; }

        const auto t_start = std::chrono::high_resolution_clock::now();
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
                                                       /*stack_frames*/4);
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

            // ---- Step 4: mtmd tokenize + eval ----
            mtmd_input_chunks * chunks_raw = mtmd_input_chunks_init();
            mtmd::input_chunks_ptr chunks(chunks_raw);

            mtmd_input_text txt;
            txt.text          = prompt.c_str();
            txt.add_special   = true;
            txt.parse_special = false;

            int32_t ret = mtmd_tokenize(state.shared_mmproj.ctx.get(), chunks_raw,
                                        &txt, raw_bmps.data(), raw_bmps.size());
            if (ret != 0) { fail_fast("mtmd_tokenize_failed"); return; }

            llama_pos new_n_past = 0;
            {
                std::lock_guard<std::mutex> lock(state.mmproj_mtx);
                ret = mtmd_helper_eval_chunks(state.shared_mmproj.ctx.get(), sess.ctx,
                                              chunks_raw, 0, 0, state.n_batch, true, &new_n_past);
            }
            if (ret != 0) { fail_fast("mtmd_eval_failed"); return; }
            n_past = (int)new_n_past;

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
        }

        // ---- Generate tokens ----
        auto gen_start = std::chrono::high_resolution_clock::now();
        auto text = generate_tokens(sess.ctx, sess.smpl, n_past, max_new, eos_tok);
        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_ms  = std::chrono::duration<double,std::milli>(gen_end - gen_start).count();
        double wall_ms = std::chrono::duration<double,std::milli>(gen_end - t_start).count();

        // ---- Send response ----
        // Safety: strip everything after the first <|im_end|> (in case the
        // model generates past the EOS token despite our checks above)
        std::string clean_text = text;
        {
            auto pos = clean_text.find("<|im_end|>");
            if (pos != std::string::npos) {
                clean_text.resize(pos);
            }
        }

        {
            json m;
            m["backend"]       = "qwen3omni";
            m["generate_ms"]   = gen_ms;
            m["wall_clock_ms"] = wall_ms;

            if (parsed_input.streaming && !clean_text.empty()) {
                // Use UTF-8-safe chunking to avoid nlohmann type_error.316 on CJK
                auto chunks = utf8_safe_chunks(clean_text, 4);
                for (const auto & chunk : chunks) {
                    ws.send(make_text_delta(sid, rid, chunk, ProtocolMetrics{}).dump());
                }
            } else if (!clean_text.empty()) {
                ws.send(make_text_delta(sid, rid, clean_text, ProtocolMetrics{}).dump());
            }

            ws.send(make_response_done(sid, rid, clean_text, "", "turn_end", ProtocolMetrics{}).dump());
        }
    }

    // ---- Cleanup on disconnect ----
    LOG_INF("session %s disconnected\n", sid.c_str());
    ws.send(make_session_closed(sid, "client_disconnected").dump());
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

    // HTTP server
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif
    svr.set_read_timeout(params.timeout_read);

    SessionManager mgr(max_sessions);
    ServerState state(shared_model, shared_mmproj, mgr, params.sampling, text_only, n_batch);

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

    LOG_INF("Qwen3-Omni server on 0.0.0.0:%d (sessions=%d text_only=%d)\n",
            params.port, max_sessions, (int)text_only);

    svr.listen("0.0.0.0", params.port);

    LOG_INF("Shutting down\n");
    mgr.shutdown();
    shared_mmproj.ctx.reset();
    shared_model.free();
    llama_backend_free();
    return 0;
}

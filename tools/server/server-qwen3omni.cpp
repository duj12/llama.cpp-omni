// Qwen3-Omni HTTP/WebSocket server — turn-based multimodal inference.
//
// Uses the shared batch executor (server-context.cpp) so multiple WebSocket
// sessions share one llama_context + one mtmd_context and are batched into
// the same llama_decode calls. Each session pins a server_slot (seq_id) and
// posts SERVER_TASK_TYPE_OMNI_STREAM tasks; results stream back over the
// /backend WebSocket protocol.
//
// Supports:
//   - Text-only mode (--text-only flag, no mmproj needed)
//   - Vision input (JPEG/PNG images via mmproj)
//   - Audio input (float32 PCM via mmproj -> WAV wrapper)
//   - Qwen3 ChatML template with <__media__> markers
//   - Streaming text output via WebSocket /backend protocol
//   - half_duplex / full_duplex (VAD+TurnSense) force_listen prefill + trigger decode
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

#include "server-context.h"
#include "server-task.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <unordered_map>
#include <map>
#include <set>
#include <thread>

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

static std::vector<uint8_t> read_file_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { return {}; }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

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
// Per-session state
// ============================================================================

struct Qwen3Session {
    std::string sid;
    int slot = -1;                       // pinned server_slot (seq_id)
    int resp_cnt = 0;
    bool half_duplex = false;            // VAD+TurnSense worker-driven replies
    std::shared_ptr<std::atomic<bool>> interrupt = std::make_shared<std::atomic<bool>>(false); // barge-in flag

    // half-duplex: cumulative prompt text + media bytes so far (for this turn).
    // Each force_listen task carries the full cumulative prompt so the shared
    // executor's prefix-reuse (chunk-hash) only evals the newly added media.
    // 累积式上下文：回复后不清空，把 assistant 回复也追加进累积 prompt，
    // 下一句靠 executor 前缀复用看到完整历史（跨分句记忆）。
    std::string cumulative_prompt;
    std::vector<raw_buffer> cumulative_files;
    bool pending_reset = true;           // next task clears the slot KV (start of a fresh turn)

    std::string last_reply;              // 最近一次 DONE 的完整回复文本（用于累积进上下文）
    int last_n_past = 0;                 // 最近一次 DONE 时该 slot 的 KV 已用 position 数
    int slot_n_ctx = 0;                  // per-slot context size（KV 降级阈值判断）
};

// ============================================================================
// Server state
// ============================================================================

struct ServerState {
    server_context ctx_server;
    SessionManager & mgr;
    common_params & params;
    bool text_only;
    int max_sessions;

    // session_id -> pinned slot index
    std::mutex slot_mtx;
    std::map<std::string, int> session_slots;

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

    // Allocate a stable slot for a session. Returns -1 if all slots are taken.
    // Uses the first unoccupied slot (not size()), so a slot freed by a previous
    // session is correctly reused even if session_slots still holds stale entries.
    int alloc_slot(const std::string & sid) {
        std::lock_guard<std::mutex> lock(slot_mtx);
        if (session_slots.find(sid) != session_slots.end()) {
            return session_slots[sid];
        }
        std::set<int> used;
        for (const auto & [_, s] : session_slots) { used.insert(s); }
        for (int i = 0; i < max_sessions; i++) {
            if (used.find(i) == used.end()) {
                session_slots[sid] = i;
                return i;
            }
        }
        return -1;
    }

    void free_slot(const std::string & sid) {
        std::lock_guard<std::mutex> lock(slot_mtx);
        session_slots.erase(sid);
    }

    ServerState(SessionManager & m, common_params & p, bool to)
        : mgr(m), params(p), text_only(to) {
        max_sessions = p.n_parallel > 0 ? p.n_parallel : 4;
    }
};

// ============================================================================
// Prompt builders
// ============================================================================

// Qwen3 ChatML prompt with <__media__> markers for the shared executor.
// NOTE: the executor tokenizes with parse_special=true (canonical 151644/151645),
// which the model was trained on. The old per-session server split these into
// byte tokens (parse_special=false) which degraded generation.
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
// Shared executor task helpers
// ============================================================================

// Post an OMNI_STREAM task to the shared executor and stream its events back
// to the /backend WS as protocol events. Returns true if the turn completed
// normally (DONE/PREFILL_DONE), false on error.
//
// force_listen = true  -> prefill only (PREFILL_DONE event, no generation)
// force_listen = false -> decode a full reply (TEXT_DELTA... then DONE)
static bool run_omni_task(httplib::ws::WebSocket & ws, ServerState & state,
                          Qwen3Session & sess, const std::string & rid,
                          std::string prompt, std::vector<raw_buffer> files,
                          bool force_listen, int max_new, bool streaming)
{
    server_task task(SERVER_TASK_TYPE_OMNI_STREAM);
    task.id_slot   = sess.slot;
    task.cli       = true;
    task.cli_prompt = std::move(prompt);
    task.cli_files  = std::move(files);

    task.params               = task_params();      // defaults
    task.params.sampling      = state.params.sampling;
    task.params.stream        = streaming;
    task.params.n_predict     = max_new;
    task.params.force_listen  = force_listen;
    task.params.reset_kv      = sess.pending_reset;
    sess.pending_reset        = false;

    server_response_reader rd = state.ctx_server.get_response_reader();
    task.id = rd.get_new_id();                     // required: queue.post() asserts id != -1
    rd.post_task(std::move(task));

    bool done   = false;
    bool error  = false;
    std::string partial_text;

    while (auto res = rd.next([&]() { return sess.interrupt->load(); })) {
        if (res->is_error()) {
            error = true;
            json err = res->to_json();
            ws.send(json_safe_dump(make_session_closed(sess.sid,
                "executor_error", err.contains("message") ? err["message"].get<std::string>() : err.dump())));
            break;
        }
        auto omni = dynamic_cast<server_task_result_omni_stream *>(res.get());
        if (!omni) { continue; }
        switch (omni->event) {
            case server_task_result_omni_stream::Event::PREFILL_DONE:
                // force_listen 任务完成：只累积 prefill，无回复。
                ws.send(json_safe_dump(make_listen_delta(sess.sid, rid, ProtocolMetrics{})));
                done = true;
                break;
            case server_task_result_omni_stream::Event::TEXT_DELTA:
                partial_text += omni->text_delta;
                ws.send(json_safe_dump(make_text_delta(sess.sid, rid, omni->text_delta, ProtocolMetrics{})));
                break;
            case server_task_result_omni_stream::Event::DONE:
                // 记录本轮回复 + KV 占用，供累积式上下文的跨分句记忆与降级判断。
                sess.last_reply   = omni->full_text;
                sess.last_n_past  = omni->n_past;
                ws.send(json_safe_dump(make_response_done(sess.sid, rid, omni->full_text, "", "turn_end", ProtocolMetrics{})));
                done = true;
                break;
            default:
                break;
        }
        if (done || error) { break; }
    }

    if (!done && !error) {
        // 循环退出但未收到 DONE → barge-in 中断。reader 析构时会 post CANCEL
        // 释放 slot；这里向客户端补发 response.done（reason=interrupted），
        // 下一轮 input 从干净 KV 开始。
        LOG_INF("session %s: task interrupted (rid=%s)\n", sess.sid.c_str(), rid.c_str());
        sess.pending_reset = true;
        ws.send(json_safe_dump(make_response_done(sess.sid, rid, partial_text, "", "interrupted", ProtocolMetrics{})));
    }

    return !error;
}

// ============================================================================
// WebSocket /backend handler
// ============================================================================

static void handle_ws(httplib::ws::WebSocket & ws, ServerState & state) {
    std::string sid = state.mgr.allocate();
    if (sid.empty()) {
        LOG_WRN("session allocation rejected: max_sessions=%d reached (all sessions busy)\n", state.max_sessions);
        ws.close();
        return;
    }
    state.mgr.set_close_callback(sid, [&ws]() { ws.close(); });

    Qwen3Session sess;
    sess.sid = sid;
    sess.slot = state.alloc_slot(sid);
    if (sess.slot < 0) {
        LOG_ERR("session %s: no free executor slot (max_sessions=%d)\n", sid.c_str(), state.max_sessions);
        ws.send(json_safe_dump(make_session_closed(sid, "no_free_slot")));
        state.mgr.close(sid);
        ws.close();
        return;
    }

    auto fail_fast = [&](const std::string & reason) {
        if (!sid.empty()) {
            ws.send(json_safe_dump(make_session_closed(sid, reason)));
        }
        state.mgr.close(sid);
        state.free_slot(sid);
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

    sess.half_duplex = (parsed_init.mode == "half_duplex" || parsed_init.mode == "full_duplex");
    state.register_interrupt(sid, sess.interrupt);

    LOG_INF("session %s started (slot=%d, mode=%s, text_only=%d)\n",
            sid.c_str(), sess.slot, parsed_init.mode.c_str(), (int) state.text_only);

    state.mgr.activate(sid, nullptr, false);

    // per-slot context 大小（用于累积式上下文降级阈值判断）。
    sess.slot_n_ctx = state.ctx_server.get_meta().slot_n_ctx;
    if (sess.slot_n_ctx <= 0) { sess.slot_n_ctx = 8192; }  // fallback

    // 关键修复：slot 释放绑定到 SessionManager::close 的 cleanup_fn。
    // reaper（reap_idle → close）或 /sessions/:id/close API 关闭 session 时，
    // handle_ws 的 ws.read() 仍可能阻塞（full_duplex 长连接），cleanup 里的
    // free_slot 不会执行 → session_slots 泄漏、新连接 no free executor slot。
    // cleanup_fn 在 SessionManager::close 的锁外调用，这里把 slot 释放挂上去，
    // 任何关闭路径都会同步释放 slot（free_slot 幂等，与 cleanup 重复调用安全）。
    if (auto * osess = state.mgr.get(sid)) {
        osess->cleanup_fn = [&state, sid]() {
            state.free_slot(sid);
            state.unregister_interrupt(sid);
        };
    }

    ws.send(json_safe_dump(make_session_created(sid, parsed_init.mode)));

    // System prompt block, prefilled once per turn (included in every
    // cumulative-prompt task; prefix-reuse keeps it cached in KV).
    const std::string sys_prompt = parsed_init.system_prompt.empty()
        ? "You are a helpful assistant."
        : parsed_init.system_prompt;
    const std::string sys_text = "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n";
    sess.cumulative_prompt = sys_text;  // every turn starts with the system block

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

        // ====================================================================
        // Half-duplex / full-duplex streaming path (VAD+TurnSense worker-driven)
        // --------------------------------------------------------------------
        // The worker streams audio/video chunks with force_listen=true (prefill
        // only, accumulate into shared KV via prefix-reuse). When VAD+TurnSense
        // says the user finished, the worker sends a chunk WITHOUT force_listen
        // → we decode a full reply. Barge-in sets sess.interrupt via POST
        // /interrupt; checked by server_response_reader.next()'s should_stop.
        // ====================================================================
        if (sess.half_duplex) {
            const bool has_audio_hd  = !parsed_input.audio_b64.empty();
            const bool has_frames_hd = !parsed_input.video_frames_b64.empty();

            // ---- Append the new media chunk to the cumulative prompt ----
            int n_media = (has_audio_hd ? 1 : 0) + (int)parsed_input.video_frames_b64.size();
            std::string user_text;
            if (n_media > 0) {
                user_text = "<|im_start|>user\n";
                for (int i = 0; i < n_media; i++) { user_text += "<__media__>"; }
                user_text += "<|im_end|>\n";
            }

            // media bytes, in marker order: audio first, then frames
            std::vector<raw_buffer> new_files;
            if (has_audio_hd) {
                auto pcm = b64_to_float32_pcm(parsed_input.audio_b64);
                if (pcm.empty()) { fail_fast("invalid_audio"); return; }
                new_files.push_back(build_wav_from_pcm(pcm));
            }
            for (const auto & frame_b64 : parsed_input.video_frames_b64) {
                auto raw_img = b64_decode(frame_b64);
                if (raw_img.empty()) { fail_fast("invalid_frame"); return; }
                new_files.push_back(std::move(raw_img));
            }

            if (user_text.empty()) {
                // force_listen 且无媒体：无可累积，跳过。
                // force_listen=false 且无媒体：纯文本 trigger（VAD 分句点）。
                if (parsed_input.force_listen) { continue; }
            } else {
                sess.cumulative_prompt += user_text;
                for (auto & f : new_files) { sess.cumulative_files.push_back(std::move(f)); }
            }

            // ---- Build the cumulative prompt for this task ----
            std::string prompt = sess.cumulative_prompt;
            if (!parsed_input.force_listen) {
                // Trigger decode: mark the assistant turn.
                prompt += "<|im_start|>assistant\n";
            }

            std::vector<raw_buffer> files = sess.cumulative_files; // copy

            bool ok = run_omni_task(ws, state, sess, rid,
                                    std::move(prompt), std::move(files),
                                    /*force_listen*/ parsed_input.force_listen,
                                    max_new, parsed_input.streaming);
            if (!ok) {
                // error already sent to the WS; close the session
                return;
            }

            // After a completed reply turn, KEEP the context for cross-turn memory.
            // Append the assistant reply to the cumulative prompt so the next
            // utterance sees the full history (prefix-reuse only evals new media).
            if (!parsed_input.force_listen) {
                if (!sess.last_reply.empty()) {
                    sess.cumulative_prompt += "<|im_start|>assistant\n" + sess.last_reply + "<|im_end|>\n";
                    sess.last_reply.clear();
                }
                // KV 接近 slot 上限时降级：reset + 从 system prompt 重建。
                // M-RoPE 无法滑窗，只能整段重建；短对话不受影响（只在长对话
                // context 快满时才牺牲历史，避免 "failed to find a memory slot")。
                // 用比例阈值（85%）而非固定值，兼容小 slot_ctx（如 -c 16384/np10
                // → slot_ctx=1792 时固定 2048 余量会是负数，导致每次触发都误降级）。
                if (sess.last_n_past > 0 && sess.slot_n_ctx > 0 &&
                    sess.last_n_past >= sess.slot_n_ctx * 85 / 100) {
                    LOG_INF("session %s: context budget nearly full (n_past=%d/slot_ctx=%d), degrading KV\n",
                            sid.c_str(), sess.last_n_past, sess.slot_n_ctx);
                    sess.pending_reset = true;
                    sess.cumulative_prompt = sys_text;  // 只保留 system prompt 作为新前缀
                    sess.cumulative_files.clear();
                }
            }
            continue;
        }

        // ====================================================================
        // Turn-based path
        // ====================================================================
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

        if (!state.text_only && (has_images || has_audio || has_video)) {
            // ---- Step 1: extract video frames + audio (if any) ----
            std::vector<std::string> video_frame_paths;
            std::string video_audio_path;
            if (has_video) {
                // 每 session 唯一临时目录，避免并发 session 写同一 video_N/ 目录
                // 导致帧文件互相覆盖/删除 → 图像 embedding 损坏 → 问号乱码。
                const std::string temp_dir = (fs::temp_directory_path() / ("qwen3omni_video_" + sid)).string();
                fs::create_directories(temp_dir);
                for (const auto & vid_b64 : last_user->video_b64s) {
                    auto vid = extract_video_mp4_media(vid_b64, temp_dir, 0, /*stack_frames*/2);
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

            // ---- Step 2: count total media items + build files (marker order) ----
            int n_user_images   = (int)last_user->image_b64s.size();
            int n_video_frames  = (int)video_frame_paths.size();
            int n_total_audio   = (int)last_user->audio_b64s.size()
                                + (video_audio_path.empty() ? 0 : 1);
            int n_media = n_user_images + n_video_frames + n_total_audio;

            std::string prompt = build_qwen3_prompt(msgs, last_user, n_media);

            std::vector<raw_buffer> files;
            // Images first
            for (const auto & img_b64 : last_user->image_b64s) {
                files.push_back(b64_decode(img_b64));
            }
            // Video frames (as images)
            for (const auto & fpath : video_frame_paths) {
                files.push_back(read_file_bytes(fpath));
            }
            // User audio
            for (const auto & aud_b64 : last_user->audio_b64s) {
                files.push_back(build_wav_from_pcm(b64_to_float32_pcm(aud_b64)));
            }
            // Video audio
            if (!video_audio_path.empty()) {
                files.push_back(read_file_bytes(video_audio_path));
            }

            // Cleanup temp files (best-effort)
            if (has_video) {
                for (const auto & fp : video_frame_paths) { fs::remove(fp); }
                if (!video_audio_path.empty()) { fs::remove(video_audio_path); }
            }

            bool ok = run_omni_task(ws, state, sess, rid,
                                    std::move(prompt), std::move(files),
                                    /*force_listen*/ false,
                                    max_new, parsed_input.streaming);
            if (!ok) { return; }
            // Turn-based: reset KV every turn (fresh context), matching the
            // old per-session full-KV-reset behavior.
            sess.pending_reset = true;
            continue;
        }

        // ---- Text-only path ----
        std::string prompt = build_qwen3_prompt(msgs, last_user, 0);
        bool ok = run_omni_task(ws, state, sess, rid,
                                std::move(prompt), {},
                                /*force_listen*/ false,
                                max_new, parsed_input.streaming);
        if (!ok) { return; }
        sess.pending_reset = true;
    }

    // ---- Cleanup on disconnect ----
    LOG_INF("session %s disconnected\n", sid.c_str());
    // 清理本 session 唯一的视频临时目录（session 结束后不再被任何并发 session 引用）。
    std::error_code ec;
    fs::remove_all((fs::temp_directory_path() / ("qwen3omni_video_" + sid)), ec);
    state.unregister_interrupt(sid);
    state.free_slot(sid);
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

    int max_sessions = params.n_parallel > 0 ? params.n_parallel : 4;
    LOG_INF("Server: max_sessions=%d total_n_ctx=%d (per-slot=%d)\n",
            max_sessions, params.n_ctx, params.n_ctx / max_sessions);

    // HTTP server
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif
    svr.set_read_timeout(params.timeout_read);

    SessionManager mgr(max_sessions);
    ServerState state(mgr, params, text_only);

    // ---- Load model into the shared executor ----
    // Neutralize settings that would break the half-duplex prefix-reuse design:
    //  - cache_idle_slots clears a slot's KV after every task (kills accumulation)
    //  - n_ctx_checkpoints adds mtmd-incompatible checkpoint machinery
    if (!params.mmproj.path.empty() && text_only) {
        // --text-only flag with --mmproj: ignore the mmproj
        params.mmproj.path.clear();
    }
    params.cache_idle_slots = false;
    params.n_ctx_checkpoints = 0;

    // The executor initializes its mtmd_context with get_media_marker(), which is
    // a RANDOM string per default. Our WS protocol uses the canonical <__media__>
    // marker. Pin the env var so get_media_marker() returns exactly <__media__>.
    if (getenv("LLAMA_MEDIA_MARKER") == nullptr) {
        setenv("LLAMA_MEDIA_MARKER", "<__media__>", 0);
    }

    LOG_INF("loading model into shared batch executor...\n");
    if (!state.ctx_server.load_model(params)) {
        LOG_ERR("%s", "failed to load model\n");
        llama_backend_free();
        return 1;
    }
    LOG_INF("%s", "shared batch executor ready\n");

    // Run the executor loop on a background thread.
    std::atomic<bool> loop_stop{false};
    std::thread executor_thread([&state]() {
        state.ctx_server.start_loop();
    });

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
    state.ctx_server.terminate();
    if (executor_thread.joinable()) { executor_thread.join(); }
    mgr.shutdown();
    llama_backend_free();
    return 0;
}

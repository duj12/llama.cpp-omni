// Omni streaming HTTP server — multi-session omni API endpoints
// Based on the old server.cpp omni handlers, adapted for multi-session llama.cpp

#include "omni.h"
#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"
#include "session.h"
#include "ws_handler.h"

#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <string>

#include "httplib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static json format_error_response(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}};
}

template<typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key)) {
        try {
            return body.at(key).get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

static void res_ok(httplib::Response & res, const json & data) {
    res.set_content(data.dump(), "application/json");
}

static void res_error(httplib::Response & res, const json & err) {
    res.status = json_value(err, "code", 500);
    res.set_content(err.dump(), "application/json");
}

static bool server_sent_event(httplib::DataSink & sink, const json & ev) {
    std::string str = "data: " + ev.dump() + "\n\n";
    return sink.write(str.data(), str.size());
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static bool ensure_omni_model_paths_from_llm(common_params & params) {
    if (params.model.path.empty()) {
        return false;
    }
    const std::string root = parent_dir(params.model.path);
    if (root.empty()) {
        return false;
    }
    if (params.vpm_model.empty()) {
        params.vpm_model = root + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    }
    if (params.apm_model.empty()) {
        params.apm_model = root + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    }
    if (params.tts_model.empty()) {
        params.tts_model = root + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    }
    if (params.tts_bin_dir.empty()) {
        params.tts_bin_dir = root + "/tts";
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Multi-session: default to 4 concurrent sessions, or use --n-parallel.
    int max_sessions = params.n_parallel;
    if (max_sessions < 1) {
        max_sessions = 4;
    }
    params.n_parallel = max_sessions; // ensure n_seq_max is sufficient

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("Omni HTTP server starting (multi-session, max_sessions=%d)...\n", max_sessions);

    // Auto-detect omni model paths
    ensure_omni_model_paths_from_llm(params);

    // HTTP server setup
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif

    // ========================================================================
    // Load all shared models ONCE at startup
    // ========================================================================
    omni_shared_models * shared_models = omni_shared_init(&params, params.tts_bin_dir,
                                                           /*tts_gpu_layers*/99,
                                                           /*token2wav_device*/"gpu:0");
    if (!shared_models) {
        LOG_ERR("Failed to initialize shared models. Exiting.\n");
        llama_backend_free();
        return 1;
    }
    LOG_INF("All shared models loaded. Ready for multi-session connections.\n");

    // Session manager — one per process
    SessionManager session_mgr(max_sessions);

    // ========================================================================
    // Routes
    // ========================================================================

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}, {"sessions", session_mgr.active_count()}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}, {"sessions", session_mgr.active_count()}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    //
    // Backend Protocol (WebSocket + HTTP unary)
    //
    svr.WebSocket("/backend", [&](const httplib::Request &, httplib::ws::WebSocket & ws) {
        handle_ws_backend(ws, session_mgr, params,
                          shared_models);
    });

    svr.Post("/sessions/:session_id/close", [&](const httplib::Request & req, httplib::Response & res) {
        std::string session_id = req.path_params.at("session_id");
        LOG_INF("Close session requested: %s\n", session_id.c_str());

        auto * session = session_mgr.get(session_id);
        if (!session || session->state != SessionState::ACTIVE) {
            res_error(res, format_error_response("session not found", "not_found"));
            res.status = 404;
            return;
        }

        session_mgr.request_transport_close(session_id);

        // close is a completion primitive: signal break, clear queues,
        // then release the per-session context and remove from manager.
        {
            auto * closing = session_mgr.get(session_id);
            if (closing && closing->octx) {
                closing->octx->break_event = true;
                {
                    std::lock_guard<std::mutex> lk(closing->octx->text_mtx);
                    closing->octx->text_queue.clear();
                    closing->octx->text_done_flag = true;
                }
                closing->octx->text_cv.notify_all();
            }
            session_mgr.close(session_id);
        }

        json resp;
        resp["ok"] = true;
        resp["session_id"] = session_id;
        resp["closed"] = true;
        res_ok(res, resp);
    });

    // start server
    LOG_INF("Omni HTTP server listening on 0.0.0.0:%d\n", params.port);
    svr.listen("0.0.0.0", params.port);

    // cleanup
    LOG_INF("Server shutting down\n");
    session_mgr.shutdown();
    omni_shared_free(shared_models);
    llama_backend_free();

    return 0;
}

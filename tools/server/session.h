#pragma once

#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <functional>
#include <unordered_map>

struct omni_context;

enum class SessionMode {
    FULL_DUPLEX,
    TURN_BASED,
};

enum class SessionState {
    UNINITIALIZED,
    ACTIVE,
    CLOSED,
};

struct OmniSession {
    std::string session_id;
    SessionState state = SessionState::UNINITIALIZED;
    SessionMode mode = SessionMode::FULL_DUPLEX;
    omni_context * octx = nullptr;
    bool owns_octx = false;
    double created_at = 0.0;
    std::function<void()> close_ws;
    // Cleanup callback invoked by SessionManager::close().
    // Allows decoupling model-specific cleanup (e.g. omni_free, llama_free)
    // from the generic session manager, so it can be reused across servers.
    std::function<void()> cleanup_fn;
};

// SessionManager — manages multiple concurrent backend sessions.
// Default max_sessions = 4.
class SessionManager {
public:
    SessionManager(int max_sessions = 4);
    ~SessionManager();

    // Allocate a new session (UNINITIALIZED state). Returns session_id.
    // Returns empty string if max_sessions reached.
    std::string allocate();

    // Activate a session after successful init. Sets octx pointer.
    bool activate(const std::string & session_id, omni_context * octx, bool owns_octx);

    // Get a session by id. Returns nullptr if not found.
    OmniSession * get(const std::string & session_id);

    // Register/trigger transport close without holding the manager lock while
    // invoking the callback.
    void set_close_callback(const std::string & session_id, std::function<void()> cb);
    void request_transport_close(const std::string & session_id);

    // Close and forget a session. Releases omni_context if owned.
    void close(const std::string & session_id);

    // Handle WS disconnect — close all sessions with expired WS.
    void on_disconnect();

    // Number of active sessions.
    int active_count() const;

    // Close all sessions (shutdown cleanup).
    void shutdown();

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::unique_ptr<OmniSession>> sessions_;
    int max_sessions_ = 4;
    std::string generate_uuid();
};

#include "session.h"

#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

SessionManager::SessionManager(int max_sessions)
    : max_sessions_(max_sessions) {}

SessionManager::~SessionManager() {
    shutdown();
}

static double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

std::string SessionManager::generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;

    uint64_t a = dist(gen);
    uint64_t b = dist(gen);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << a
        << std::setw(16) << b;
    return oss.str();
}

std::string SessionManager::allocate() {
    std::lock_guard<std::mutex> lock(mtx_);

    // Count active sessions (exclude CLOSED)
    int active = 0;
    for (const auto & [id, s] : sessions_) {
        if (s->state != SessionState::CLOSED) {
            active++;
        }
    }
    if (active >= max_sessions_) {
        return ""; // max concurrent sessions reached
    }

    auto session = std::make_unique<OmniSession>();
    session->session_id = generate_uuid();
    session->state = SessionState::UNINITIALIZED;
    session->created_at = now_seconds();
    session->last_active_at = session->created_at;

    std::string id = session->session_id;
    sessions_[id] = std::move(session);
    return id;
}

bool SessionManager::activate(const std::string & session_id, omni_context * octx, bool owns_octx) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }
    if (it->second->state != SessionState::UNINITIALIZED) {
        return false; // already active or closed
    }

    it->second->octx = octx;
    it->second->owns_octx = owns_octx;
    it->second->state = SessionState::ACTIVE;
    it->second->last_active_at = now_seconds();
    return true;
}

void SessionManager::touch(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->last_active_at = now_seconds();
    }
}

std::vector<std::string> SessionManager::reap_idle(double idle_timeout_s) {
    // Collect idle session ids (ACTIVE but inactive too long), then close them
    // outside the manager lock.
    std::vector<std::string> to_close;
    double now = now_seconds();
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (const auto & [id, s] : sessions_) {
            if (s->state == SessionState::ACTIVE &&
                (now - s->last_active_at) > idle_timeout_s) {
                to_close.push_back(id);
            }
        }
    }
    for (const auto & id : to_close) {
        close(id);
    }
    return to_close;
}

OmniSession * SessionManager::get(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return nullptr;
    }
    return it->second.get();
}

void SessionManager::set_close_callback(const std::string & session_id, std::function<void()> cb) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return;
    }
    it->second->close_ws = std::move(cb);
}

void SessionManager::request_transport_close(const std::string & session_id) {
    std::function<void()> close_ws;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return;
        }
        close_ws = it->second->close_ws;
    }

    if (close_ws) {
        close_ws();
    }
}

void SessionManager::close(const std::string & session_id) {
    std::unique_ptr<OmniSession> to_free;
    {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return; // not found
        }

        if (it->second->state == SessionState::ACTIVE) {
            it->second->state = SessionState::CLOSED;
        }

        to_free = std::move(it->second);
        sessions_.erase(it);
    }

    // Run the session's cleanup callback outside the manager lock. The callback
    // may free model contexts, join threads, or release encoder resources, so
    // holding mtx_ here could race with WS cleanup on other threads.
    if (to_free && to_free->cleanup_fn) {
        to_free->cleanup_fn();
    }
}

void SessionManager::on_disconnect() {
    // Not used in multi-session architecture: WS disconnect cleanup is handled
    // directly in handle_ws_backend via session_mgr.close(session_id).
    // This is a no-op stub retained for API compatibility.
}

int SessionManager::active_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    int count = 0;
    for (const auto & [id, s] : sessions_) {
        if (s->state == SessionState::ACTIVE) {
            count++;
        }
    }
    return count;
}

void SessionManager::shutdown() {
    std::vector<std::unique_ptr<OmniSession>> to_free_list;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto & [id, s] : sessions_) {
            if (s->state != SessionState::CLOSED) {
                s->state = SessionState::CLOSED;
            }
            to_free_list.push_back(std::move(s));
        }
        sessions_.clear();
    }

    for (auto & to_free : to_free_list) {
        if (to_free && to_free->cleanup_fn) {
            to_free->cleanup_fn();
        }
    }
}
